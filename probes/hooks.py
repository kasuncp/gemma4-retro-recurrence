"""Forward-pass hooks for the looping probes.

Three flavours, corresponding to the experimental rounds:

  make_looped_forward      --- Round 1: naive loop of one layer, no PLE control.
  make_looped_forward_ple  --- Round 2a+: loop one layer with controlled PLE
                                re-injection (vanilla / scaled / once / zero).
  install_pair_loop_hooks  --- Round 3a: loop a CONTIGUOUS PAIR (L, L+1) as a
                                unit, using PyTorch pre/post hooks (no forward
                                replacement).
"""


def make_looped_forward(orig_forward, r):
    """Round-1 hook: naive loop, no PLE control."""
    def looped(hidden_states, *args, **kwargs):
        out = None
        for _ in range(r):
            out = orig_forward(hidden_states, *args, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped


def make_looped_forward_ple(orig_forward, r, ple_mode, ple_kwarg, location_recorder=None):
    """
    Round-2a hook (post-addendum2): loop the decoder layer `r` times with
    controlled PLE re-injection, handling per_layer_input passed either
    positionally or as a kwarg.

    Background (plan2a-add2): the Gemma 4 decoder layer signature is
        forward(self, hidden_states, per_layer_input=None, shared_kv_states=None, ...)
    and Gemma4Model.forward passes per_layer_input positionally on the
    transformers version we tested. The pre-fix version of this hook only
    looked in kwargs, so it never actually intercepted PLE --- every loop
    iteration got the original tensor back via *args, making zero-mode
    bitwise-identical to vanilla. Now we detect both paths.

    location_recorder: optional dict; on the first hook call, its
    "ple_location" key is set to "positional" | "kwarg" | "missing". Lets
    callers observe which code path the hook actually took.

    ple_mode in {"vanilla", "scaled", "once", "zero"}.
      - "zero": always pass scale=0.0, including at i=0.

    For r=1 with mode in {vanilla, scaled, once}, scale == 1.0 by
    construction and we pass the original args/kwargs through unchanged ---
    so plan2a's regression checks 2 and 3 stay bitwise-identical to vanilla
    r=1.
    """
    first_call = [True]

    def looped(hidden_states, *args, **kwargs):
        if ple_kwarg in kwargs:
            original_ple = kwargs[ple_kwarg]
            ple_location = "kwarg"
        elif len(args) >= 1:
            # *args captures everything after hidden_states; per_layer_input
            # is the first positional after hidden_states, so args[0].
            original_ple = args[0]
            ple_location = "positional"
        else:
            original_ple = None
            ple_location = "missing"

        if first_call[0]:
            print(f"  [hook] PLE arrives via {ple_location} (mode={ple_mode}, r={r})")
            if location_recorder is not None:
                location_recorder.setdefault("ple_location", ple_location)
            first_call[0] = False

        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            elif ple_mode == "zero":
                scale = 0.0
            else:
                raise ValueError(f"unknown ple_mode={ple_mode!r}")

            if original_ple is None or scale == 1.0:
                # Pass through unchanged --- preserves bitwise identity at
                # scale=1.0, and there's nothing to scale when PLE is None.
                call_args = args
                call_kwargs = kwargs
            else:
                scaled_ple = original_ple * scale
                if ple_location == "kwarg":
                    call_args = args
                    call_kwargs = dict(kwargs)
                    call_kwargs[ple_kwarg] = scaled_ple
                else:  # "positional"
                    call_args = (scaled_ple,) + args[1:]
                    call_kwargs = kwargs

            out = orig_forward(hidden_states, *call_args, **call_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped


def install_pair_loop_hooks(decoder_layers, L, r):
    """Install forward hooks to loop the pair (L, L+1) as a unit r times.

    Implementation: three PyTorch hooks, no forward replacement.

      1. forward_pre_hook on layer L (with_kwargs=True) --- captures
         whatever args/kwargs the outer model passes to layer L
         (hidden_states, per_layer_input[L], shared_kv_states, position_*,
         attention_mask, ...).
      2. forward_pre_hook on layer L+1 (with_kwargs=True) --- same capture
         for layer L+1 (grabs per_layer_input[L+1], which is otherwise
         invisible from inside layer L's forward).
      3. forward_hook on layer L+1 --- fires AFTER the outer model's first
         pair application (L then L+1) has completed normally. If r > 1,
         re-runs (L, L+1) for r-1 additional iterations using the captured
         args/kwargs, feeding each iteration's output back as the next
         iteration's hidden_states input. Replaces L+1's output with the
         final iteration's hidden state.

    The captured kwargs include the mutable `shared_kv_states` dict, so
    each looped iteration writes/reads it exactly as the normal forward
    would --- if layer L is a KV producer, subsequent iterations overwrite
    the stored K/V with the re-run values, which is the correct behaviour
    for a block-internal recurrence.

    At r=1, the forward_hook does 0 extra iterations and returns the
    unmodified output --- so perplexity at r=1 must bitwise-match the
    unmodified baseline (this is the regression check).

    Inner re-runs use `module.forward(...)` rather than `module(...)`, so
    they bypass `__call__` and do not retrigger any of our own hooks.

    Returns an uninstall callable.
    """
    if L < 0 or L + 1 >= len(decoder_layers):
        raise ValueError(f"pair start L={L} out of range for {len(decoder_layers)} layers")

    captured_L = {}
    captured_L1 = {}
    handles = []

    def pre_L(module, args, kwargs):
        captured_L["args"] = args
        captured_L["kwargs"] = kwargs

    def pre_L1(module, args, kwargs):
        captured_L1["args"] = args
        captured_L1["kwargs"] = kwargs

    layer_L = decoder_layers[L]
    layer_L1 = decoder_layers[L + 1]

    def post_L1(module, input, output):
        # r=1: no-op. Regression check passes trivially.
        if r <= 1:
            return output
        # Both captures must have fired. If not, something bypassed the
        # outer call path --- fail loudly rather than silently corrupting.
        if "args" not in captured_L or "args" not in captured_L1:
            raise RuntimeError(
                "pair-loop hook fired without both pre-hook captures; "
                f"captured_L keys={list(captured_L)}, "
                f"captured_L1 keys={list(captured_L1)}"
            )

        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output

        # r-1 additional iterations. We already ran the pair once
        # (that's how we got `output`), so total applications = r.
        for _ in range(r - 1):
            new_args_L = (x,) + captured_L["args"][1:]
            out_L = layer_L.forward(*new_args_L, **captured_L["kwargs"])
            x = out_L[0] if isinstance(out_L, tuple) else out_L

            new_args_L1 = (x,) + captured_L1["args"][1:]
            out_L1 = layer_L1.forward(*new_args_L1, **captured_L1["kwargs"])
            x = out_L1[0] if isinstance(out_L1, tuple) else out_L1

        if is_tuple:
            return (x,) + output[1:]
        return x

    handles.append(layer_L.register_forward_pre_hook(pre_L, with_kwargs=True))
    handles.append(layer_L1.register_forward_pre_hook(pre_L1, with_kwargs=True))
    handles.append(layer_L1.register_forward_hook(post_L1))

    def uninstall():
        for h in handles:
            h.remove()
        captured_L.clear()
        captured_L1.clear()

    return uninstall
