# How We Doubled a Pocket-Sized AI's Math Score With a One-Line Change

## Part 2 of: Towards Local-First Mobile AI Assistance — Explained

**By Kasun Perera | April 29, 2026**

---

## What this post is about, in plain English

Imagine you have a tiny AI that lives entirely on your phone. No internet connection. It doesn't have to send your questions to a data center somewhere — it just answers them, right there in your pocket. That's the dream of "local-first AI."

The problem: phone-sized AI models are *much* smaller than the cloud-based ones you're used to. ChatGPT, Claude, and similar chatbots run on enormous machines and contain hundreds of billions of internal "dials" (we call them **parameters**). The model in this post — **Gemma 4 E2B**, made by Google — has only 2 billion. That's somewhere between 100× and 500× smaller. Can a model that small be useful for hard tasks like solving math word problems?

In **Part 1** of this series I ran one experiment: I tested the model on a standard benchmark called **GSM8K** (about 8,500 grade-school math word problems with worked solutions). With the right prompt, the model got 31% correct. Without the right prompt, it got 1%. So the way you talk to a small AI matters. A lot.

This post is what happened next. I ran eight more experiments to find out whether 31% was really the ceiling. It wasn't. With one-line changes that cost no extra compute, I lifted the score to **78%** — more than double.

This post tells the story, defines the jargon as we go, and ends with a verdict on the best settings to use this model with.

---

## A vocabulary cheat sheet (one paragraph)

A **prompt** is the text you type into the AI. A **token** is roughly half a word — models read and write text one token at a time. (For example, "hello world" is two tokens.) **Inference** is the act of running the model to get an answer; it costs compute (CPU/GPU work) and energy (battery). **Greedy decoding** means the model always picks its most likely next token; **sampling** means it picks randomly from a probability distribution, which adds variety. **Chain-of-thought (CoT)** is a prompting trick where you ask the model to "show its work" before answering, like making a student do scratch paper. **Few-shot prompting** is when you include 1–8 worked examples in your prompt. **Zero-shot prompting** is just asking the question, no examples.

Got it? Onward.

---

## Where Part 1 left us

Part 1 ended with a clean comparison. On the first 100 problems of GSM8K, the model got:

- **31%** correct when given 8 worked examples to imitate (chain-of-thought, "8-shot CoT")
- **1%** correct when just asked for an answer with no reasoning encouragement

The 31% number became Part 1's "best so far." The whole point of this series is to compare a few different ways of squeezing more out of a small AI on a phone. The 31% was the score to beat.

But I had a nagging feeling. Was 31% really the ceiling? Or had I just tried the obvious thing and stopped looking?

Eight experiments later, I had an answer. Most of those experiments confirmed mundane things. Two of them rewrote the whole story.

---

## Experiment 2: Throwing more compute at the problem

If 31% really is the ceiling, then giving the model more "thinking time" shouldn't help. But what does "more thinking" mean for an AI?

There are basically two ways to spend more compute on the same problem:

1. **Let the model write a longer solution.** Maybe it's running out of room. I tried token caps of 128, 256, 512, and 1,024.
2. **Run the model multiple times and vote on the answer.** This is called **self-consistency**. Run 1, 3, 5, or 10 attempts, then take whichever answer appears most often.

I tested all 8 combinations on the same 500 GSM8K problems.

> **Why 500?** Because GSM8K has 8,500 problems and running every cell on every problem would take days. With 500, the **confidence interval** — the range of values our true accuracy is probably in — is about ±4 percentage points. That's tight enough to detect real differences. With 100 it'd be too noisy; with 5,000 it'd be wasteful.

**Result: nothing.**

Every single one of those 8 cells landed within 2.6 percentage points of the original 30%. Letting the model write up to 1,024 tokens? Same as 128. Running 10 attempts and voting? Lifted accuracy from 30% to 32% — well within the noise band. (See **Figure 1** in the figures companion.)

This is a useful negative result. It tells us "more inference compute" isn't the answer. The 31% number is robust against the obvious lever.

But maybe it was robust because everything was *broken in the same way*.

---

## Experiment 3: What was going wrong on the wrong answers?

I started reading the model's actual completions to see *how* it was failing. One pattern jumped out: **repetition loops**.

A repetition loop is when the model gets stuck saying the same thing over and over, like a CD that's skipping. For example, it might output "the answer is 16. The answer is 16. The answer is 16. ..." until it runs out of token budget. This isn't reasoning — it's a sign that something has gone wrong inside the model's internal probability calculations.

I wrote a quick detector. Here's the rule, in plain English: any 10-to-60-character chunk of text that appears three or more times back-to-back counts as a repetition loop. (In code, that's a regular expression: `(.{10,60})\1{2,}`.) I ran it across every completion.

**Result:** about **10%** of completions were repetition loops. (See Figure 2.) The rate barely changed with token cap or sampling — at 128 tokens, 10%; at 1,024 tokens, 10%; under sampling instead of greedy decoding, 9%.

10% is real. But it's bounded. If I dropped repetition-looped problems from my denominator (treating them as "we don't have an answer for this one"), the score went from 30.0% to 33.4% — a 3-point lift, max. So repetition is a real failure mode, but it's not the *whole* story. The model is also wrong on plenty of problems where it didn't loop.

The 31% wasn't a stability ceiling. It was a *reasoning* ceiling. Or so I thought.

---

## Experiment 4: A sanity check on a different benchmark

Maybe the model's struggle was specific to math. So I tried a different benchmark: **ARC-Easy**, a multiple-choice science test for grade-schoolers.

**Result:** 83% on ARC-Easy. Whether I used chain-of-thought or just direct prompting. Whether I voted across 5 attempts or did just 1.

ARC-Easy turned out to be too easy. The model is already near the ceiling — it doesn't separate "good prompts" from "bad prompts," because the model is barely working hard. (See Figure 3.) This was useful negative information: ARC-Easy can't help me figure out what's holding the model back on GSM8K.

I needed a different angle.

---

## Experiment 5: The plot twist

Here's where everything changed.

So far I'd been using the **8-shot chain-of-thought prompt** from a 2022 paper by Wei et al. The "8-shot" part means I include 8 worked examples before the actual question. Each example shows: "Question: ... Reasoning: ... Answer: ...". The whole prompt is supposed to teach the model what kind of output we want.

This style of prompting was a big deal in 2022. But Gemma 4 E2B is from 2026, and it's an **instruction-tuned** model — meaning Google trained it specifically to follow conversational instructions, like a chatbot would. Instruction-tuned models expect to be asked questions directly, not handed a script of examples to follow.

> **What is "instruction tuning"?** When Google originally trained Gemma 4 E2B, they did it in two phases. Phase 1 ("pretraining") was reading billions of words of internet text. Phase 2 ("instruction tuning") was a finishing-school step where they fine-tuned the model on lots of conversations: "user asks question, helpful assistant gives answer." After instruction tuning, the model is biased toward conversational, helpful responses. The 8-shot CoT trick was designed for *non-instruction-tuned* models — base models that just continue text patterns. On an instruction-tuned model, including 8 worked examples might confuse it.

So I tested two new prompts on the same 500 problems:

1. **C1:** Just the question, plus the words "Let's think step by step." (a popular zero-shot CoT trick)
2. **C2:** Just the question. Nothing else. Routed through the model's built-in **chat template** — the formatting Google uses to mark "user input" vs. "assistant response" in a conversation.

The numbers were absurd:

- **A3** (the old 8-shot CoT prompt): **30.0%**
- **C1** (zero-shot, "Let's think step by step"): **67.0%**
- **C2** (zero-shot, just the question, chat template): **71.6%**

C2 — *just the question, nothing else* — beat the canonical 2022 prompt by **41.6 percentage points** on the exact same 500 problems. (See Figure 4.)

I re-checked everything. Same model. Same problems. Same scoring rules. The only difference was the prompt format. The 8 worked examples I'd been so carefully including were actively *hurting* the model.

> **What's a McNemar test?** When two methods are tested on the same problems, you can ask: "of the problems where the two methods disagree, how often does method X win vs. method Y?" McNemar's exact test gives you a p-value for whether that disagreement is statistically meaningful. For C2 vs. A3, the test said C2 wins on 234 problems where A3 fails, and A3 wins on 26 problems where C2 fails. The probability that this happened by chance is about 5 × 10⁻⁴³ — essentially zero.

The 31% from Part 1 wasn't a reasoning ceiling. It was an "I'm using stale 2022 advice on a 2026 model" ceiling.

This is the moment when Part 2 stopped being a confirmation of Part 1 and started being a correction.

---

## Experiment 9: The cherry on top (the bonus 6 points)

Once I had the new prompt, I went back and looked at the 26 problems where the *old* prompt got the right answer but the *new* one didn't. If C2 is really better, those 26 cases should look weird.

They did look weird — but not in the way I expected. I ran a quick automated check: did C2's answer text contain the correct number, somewhere?

**24 out of 26** (92%) had the correct number sitting right there in the text. The model had reasoned its way to the right answer. The problem was that my **answer extractor** — the small piece of code that pulls a number out of the AI's text — was grabbing the wrong number.

Here's a concrete example. Suppose the gold answer is 16, and the model writes:

> "The shirt costs \$50 originally, with a 32% discount it goes to \$50 × 0.68 = \$34. Then with tax, \$34 × 1.06 = \$36.04, but the question is asking how much the customer *saved* compared to the marked price, which is \$50 − \$34 = **\$16.00**."

The model is right. It said \$16. But my original extractor was set up to grab the *last integer* it saw in the text. The last integer in "\$16.00" is **0** (the trailing zero). So the score was 0, even though the model said 16.

I wrote a smarter extractor that knows about dollar signs, **bold formatting**, "Answer: 16" patterns, and end-of-line "= 16" expressions. Re-ran the scoring on all 500 C2 completions.

**C2 went from 71.6% to 78.0%.** (See Figure 8.) Plus 6.4 percentage points, no model change, no extra compute, just a smarter way of reading the model's output.

I also checked: of all 500 problems, **90.8%** have the correct number somewhere in the text. So the *true* reasoning ceiling — what the model can actually do — is closer to **91%**. The 13-point gap from 78% to 91% is problems where the model is right but the answer is buried in a way no extractor will reliably find.

This was the second-cheapest improvement of the whole study, after the prompt change. Total Part 2 lift so far: **30% → 78%, a 48-point swing**, both wins coming from "I was reading the model wrong, not the model is bad."

---

## Experiment 6: Does more compute help the new prompt?

Now that I had a working prompt, I went back to Experiment 2's question: does more compute help on top of C2?

Same 8-cell setup as before — vary length, vary number of votes — but now on top of C2 instead of A3. (See Figure 5.)

The results were a bit more interesting this time. The length axis showed a real curve below 256 tokens:

- 128-token cap: **11.4%** (model gets cut off mid-reasoning)
- 256-token cap: **48.2%**
- 512-token cap: **71.6%**
- 1,024-token cap: **72.8%**

So 512 tokens is the sweet spot. Less and the model can't finish thinking; more and it's already done. Doubling the budget from 512 to 1,024 buys us 1.2 points, well within the margin of error.

Self-consistency — running 10 attempts and voting — added 0.8 percentage points at 10× the compute and 5× the wall-clock time. Same negative result as Experiment 2: **more inference compute is not the lever**. The plateau is real. It's just at 71.6% (or 78% with the smart extractor), not 30%.

---

## Experiment 7: Does the magic prompt work on every benchmark?

So far I'd only tested C2 on GSM8K, math word problems. Maybe its advantage is GSM8K-specific.

I tested three more benchmarks:

- **ARC-Challenge:** the harder version of ARC-Easy, multi-step science questions.
- **MATH:** high-school-level competition math (think USAMO problems).
- **BBH-lite:** "Big-Bench Hard, lite version" — a curated mix of reasoning tasks across many sub-benchmarks.

Three benchmarks, three different stories. (See Figure 6.)

**ARC-Challenge:** C2 absolutely dominates. 75.6% with C2, vs. 13.6% with the old 8-shot prompt. (Why was 8-shot so bad here? Because I capped its output at 16 tokens to force a single-letter answer, but with 8 examples plus a question, the model often runs out of space before producing a letter.) The chat-template effect is even more dramatic on ARC-Challenge than on GSM8K.

**MATH:** zero. Across every prompt. The model simply cannot do high-school competition math in any prompt format. It writes elaborate-looking reasoning chains, but the math is wrong every time. This is a *capability ceiling*, not a prompt ceiling — the model wasn't trained on enough mathematical content to engage with this. No prompt change can fix this.

**BBH-lite:** here's where it got interesting. C2 scored 20.6%. The 8-shot CoT prompt got 34.0%. And the simplest possible prompt — "just pick a letter, no reasoning" — scored **49.8%**. That's almost 30 points better than C2.

> **Why does C2 lose on BBH-lite?** BBH includes a lot of pattern-recognition and disambiguation tasks where being asked to "think out loud" actively hurts. The model talks itself out of the right answer. On these tasks, the right strategy is to clip the model to a single letter, give it strong examples, and let it pattern-match.

So C2 isn't a universal magic prompt. There's no single "best prompt" for our model. **The right prompt depends on the kind of task.**

---

## Experiment 8: Can it actually run on a phone?

The whole point of this series is local-first AI. Everything so far has been measured on a desktop GPU (an Nvidia 3090, about \$2,000 of hardware). Translating those results to phone hardware is the part that the entire thesis depends on.

I tested on a Snapdragon 8 Gen 3 Elite — a flagship Android chip, the kind that powers a Samsung Galaxy S24 Ultra. To make the model fit, I used a compression scheme called **quantization** (specifically, "Q4_K_M" via a library called MLC-LLM).

> **What is quantization?** A neural network's parameters are normally stored as 16-bit or 32-bit floating-point numbers. Quantization rounds those numbers to fewer bits — Q4 means "round down to 4 bits each." This makes the model 4× smaller and roughly 2–4× faster, at the cost of a small accuracy drop because the rounding loses information.

I tested two cells, 50 problems each:

- **C2_SD:** the new prompt, on the phone
- **A3_SD:** the old 8-shot prompt, on the phone

Three findings, one of them genuinely unexpected. (See Figure 7.)

**Finding 1: Quantization costs accuracy.** C2 dropped from 71.6% on the desktop to 66.0% on the phone — a 5–6 percentage point loss. That's the price of squeezing the model down to phone-sized.

**Finding 2: The new prompt is *slower* on a phone.** This is the surprise. C2 takes 16.7 seconds per problem; A3 takes 13.4 seconds. Even though A3's prompt is 678 tokens longer than C2's!

Why? Because phone hardware processes prompts and generates new tokens at very different speeds. **Reading the prompt** ("prefill") is fast — under a second for 750 tokens. **Generating new tokens** is slow — every new token requires running the entire model again. C2 generates 293 tokens on average; A3 generates 231. So even though A3's prompt is much longer, A3 finishes faster because it generates fewer tokens.

This flips a common assumption. On desktop GPUs, longer prompts and longer generations both look expensive. On phones, the prompt is "free" and the generation is "expensive."

**Finding 3: Energy cost.** About 2,500 joules per problem with C2, 2,000 joules with A3.

> **How much is 2,500 joules?** A fully-charged phone battery holds about 70,000 joules. So 2,500 joules per problem means roughly 30 problems before draining 1% of your battery, or about 1,400 problems before draining the whole thing. Tens of minutes of sustained usage. Not nothing.

The blocking issue is latency. **13–17 seconds per problem is way too slow for an interactive chat assistant.** Most users expect a response within 2 seconds. So Path 1 — even with our improvements — does not meet the latency target on its own. To get there, we need either a smaller model (Path 3, in a future post), a smarter inference path (Path 4), or to relax the use case (run the AI on background tasks rather than real-time conversation).

---

## What we learned: five big ideas

After eight experiments, here's the picture.

**1. The way you talk to an instruction-tuned AI matters far more than how much compute you give it.** Switching from 8-shot prompts to plain zero-shot prompts via the chat template was a one-line change that added **41.6 percentage points** on GSM8K and **62 points** on ARC-Challenge. No other lever — longer thinking, more attempts, anything else — came close.

**2. Folk wisdom from 2022 actively hurts in 2026.** The "show the model 8 worked examples" trick was state-of-the-art on the previous generation of models. On modern instruction-tuned models, those examples drag the model away from its own reasoning. If you read a tutorial that says "always include examples," check the date.

**3. How you read the model's output matters too.** A naive answer extractor — "grab the last number" — was throwing away 6.4 percentage points of correct answers. The model often emits the right answer in markdown formatting like `**$16.00**`, and a smart extractor that understands common formats recovers most of those. Worth doing.

**4. There is no universal "best prompt."** The same C2 prompt that crushed GSM8K and ARC-Challenge underperformed by 29 percentage points on BBH-lite. On recognition-style tasks, asking the model to "think out loud" actively hurts. The right prompt depends on the kind of task.

**5. On-device dynamics are different from desktop dynamics.** On a phone, generating tokens is expensive and reading the prompt is cheap. So a long prompt with short generation can be faster than a short prompt with long generation. This is the *opposite* of what most desktop benchmarks would suggest.

---

## The verdict: how to use Gemma 4 E2B-it well

If you're building something with this model, here's what eight experiments say to do.

**Default deployment configuration:**

```python
config = {
    "model_id": "google/gemma-4-E2B-it",
    "prompt_template": "{question}",   # plain — no examples
    "apply_chat_template": True,        # mandatory
    "max_new_tokens": 512,              # not less, not more
    "temperature": 0,                   # greedy decoding
    "do_sample": False,                 # don't run multiple attempts
    "answer_extractor": "smart_v2",     # prose-aware
}
```

**Per-task routing:**

| Task type | Best prompt | Expected accuracy |
|---|---|---|
| Multi-step arithmetic (GSM8K-shaped) | C2 + smart extractor | 78% |
| Multi-step inference (ARC-Challenge-shaped) | C2 + smart extractor | 76% |
| Recognition / multi-task (BBH-lite-shaped) | Direct 8-shot, no reasoning | 50% |
| High-school competition math (MATH) | None — out of scope | 0% |

**Things to skip:**

- **8-shot Wei et al. exemplars on math/reasoning benchmarks** — they cost 678 prompt tokens and *reduce* accuracy by 41.6 points.
- **Self-consistency on top of C2** — adds 0.8 points at 10× compute. Not worth it on a phone.
- **Generations longer than 512 tokens** — saturates; 1,024 buys 1.2 points at marginal compute increase.
- **Generations shorter than 256 tokens** — the cap binds, and accuracy collapses.
- **Naive last-integer extraction** — costs 6.4 points to "format failure" suppression.

**The headline numbers:**

| Metric | Desktop (3090, bf16) | On-device (Snapdragon 8 Gen 3, Q4_K_M) |
|---|---|---|
| GSM8K accuracy (smart extractor) | **78.0%** | ~66% (n = 50) |
| ARC-Challenge accuracy | **75.6%** | not measured |
| BBH-lite accuracy (Direct) | **49.8%** | not measured |
| Wall-clock per problem | 3.3 s | 13–17 s |
| Energy per problem | n/a | ~2,000–2,500 J |

---

## What's next

There are three other approaches I haven't tested yet — each takes a different angle on the same goal of "make a small AI smarter without making it bigger":

- **Path 2 (depth-recurrence retrofit):** Modify the model so it can "think harder" by reusing its layers in a clever way. Same parameters, more passes through them.
- **Path 3 (quantized E4B):** Run a slightly larger model — Gemma 4 *E4B* instead of *E2B*, twice as many parameters — but compress it so heavily that it still fits on a phone.
- **Path 4 (Mixture-of-Depths):** Let the model choose for itself how much compute to spend on each token, skipping work on easy parts.

Each one has to clear the bar Path 1 just set: **78% on GSM8K-desktop, 66% on GSM8K-on-phone**, at the same compute budget. Those targets are *much* harder than what Part 1's 31% would have implied. Whether any of the three paths clears them is what Part 3 starts to answer.

---

## A note on what surprised me

I started this project assuming the bottleneck would be compute. Phones don't have GPUs the size of refrigerators; they have tiny chips and small batteries. Surely the answer to "how do we make a small AI better" would involve smarter ways of spending compute.

That turned out to be wrong, twice over.

The biggest improvement (41.6 points) came from typing the question in a different way. The second-biggest (6.4 points) came from a slightly smarter way of reading the answer. Both took less than an hour to implement. Both cost zero extra compute.

Not every problem in AI is a compute problem. Some of them are paying-attention problems.

That's been the lesson of Part 2.

---

## Glossary

For reference, here are all the terms I used in one place.

| Term | Plain-English meaning |
|---|---|
| **Parameters** | The internal "dials" of a neural network. Gemma 4 E2B has 2 billion. |
| **Prompt** | The text you type into the AI. |
| **Token** | A unit of text the model reads/writes — roughly half a word. |
| **Inference** | The act of running the model to produce output. |
| **Greedy decoding** | The model always picks its single most likely next token. |
| **Sampling** | The model picks randomly from its top probable tokens, adds variety. |
| **Temperature** | How spread-out the sampling distribution is. 0 = greedy, 1 = high variety. |
| **Chain-of-thought (CoT)** | Prompting the model to "show its work" before answering. |
| **Few-shot prompting** | Including 1–8 worked examples in the prompt. |
| **Zero-shot prompting** | No examples, just the question. |
| **Self-consistency** | Run the model multiple times with sampling and vote on the answer. |
| **Instruction-tuned** | A model that's been fine-tuned to follow conversational instructions. |
| **Chat template** | The formatting a model uses to mark "user input" vs. "assistant response." |
| **Base model** | A model that has *not* been instruction-tuned — just text continuation. |
| **Quantization** | Compressing a model by storing its parameters in fewer bits (Q4 = 4 bits). |
| **Repetition loop** | A failure mode where the model gets stuck repeating itself. |
| **Confidence interval** | The range your true accuracy is probably in (we use 95% CIs throughout). |
| **McNemar's test** | A statistical test comparing two methods on the same set of problems. |
| **GSM8K** | A standard benchmark of grade-school math word problems. |
| **ARC-Easy / ARC-Challenge** | Multiple-choice science benchmarks (Easy is grade-school; Challenge is harder). |
| **MATH** | A symbolic-math benchmark at high-school competition level. |
| **BBH-lite** | "Big-Bench Hard, lite" — a curated mix of reasoning tasks. |
| **FLOPs** | Floating-point operations. A measure of how much compute something takes. |
| **Wall-clock** | Real elapsed time (as opposed to compute cycles). |
| **Joule** | A unit of energy. Phone batteries hold ~70,000 J. |

---

**Share | Discussion Coming**

© 2026 Kasun Perera · Privacy · Terms · Collection Notice
