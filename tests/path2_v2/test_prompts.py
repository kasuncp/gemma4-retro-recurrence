"""Phase 0 prompt-builder tests --- CPU only, no model load.

Uses a tiny dummy tokenizer that mimics the Gemma chat template so we
don't need network access. The Gemma chat template uses
``<start_of_turn>user`` / ``<start_of_turn>model`` markers; we test
that the builder produces a string ending in the model header (not
the user header) when ``add_generation_prompt=True``, since that was
the exact bug Path 1 plan 5 spent its first hour on.

Run:
    PYTHONPATH=. python -m unittest tests.path2_v2.test_prompts -v
"""

import unittest

from probes.prompts import (
    EXEMPLAR_SET_ID,
    PROMPT_BUILDERS,
    WEI_8SHOT_EXEMPLARS,
    build_8shot_cot_gsm8k,
    build_c2_arc,
    build_c2_bbh,
    build_c2_gsm8k,
    build_prompt,
)


class _FakeGemmaTokenizer:
    """Mimics the surface area of ``tokenizer.apply_chat_template``.

    Real Gemma template format (simplified for testing):
      <bos><start_of_turn>user
      <content>
      <end_of_turn>
      <start_of_turn>model
      <optional content>
      <end_of_turn>
    """

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False, "test fakes only do string mode"
        out = "<bos>"
        for m in messages:
            out += f"<start_of_turn>{m['role']}\n{m['content']}\n<end_of_turn>\n"
        if add_generation_prompt:
            out += "<start_of_turn>model\n"
        return out


class TestC2Gsm8k(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeGemmaTokenizer()

    def test_contains_question_and_user_role(self):
        q = "If Alice has 3 apples and gets 2 more, how many?"
        out = build_c2_gsm8k(self.tok, q)
        self.assertIn(q, out)
        self.assertIn("<start_of_turn>user", out)

    def test_ends_in_model_header_not_user(self):
        out = build_c2_gsm8k(self.tok, "trivial")
        # The last turn header MUST be model --- this is the Path 1
        # plan 5 first-hour bug we're explicitly guarding against.
        self.assertTrue(
            out.rstrip().endswith("<start_of_turn>model"),
            f"prompt did not end in model header; got: {out!r}",
        )

    def test_no_cot_suffix(self):
        out = build_c2_gsm8k(self.tok, "trivial")
        # C2 by definition is the bare question; the "let's think
        # step by step" phrase belongs to C1, which scored worse.
        self.assertNotIn("Let's think step by step", out)
        self.assertNotIn("step by step", out.lower())

    def test_no_exemplars(self):
        out = build_c2_gsm8k(self.tok, "trivial")
        for q, _ in WEI_8SHOT_EXEMPLARS:
            self.assertNotIn(q[:30], out, "C2 must not contain Wei exemplars")


class TestC2Arc(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeGemmaTokenizer()

    def test_lettered_choices(self):
        out = build_c2_arc(
            self.tok, "Why is the sky blue?",
            ["Rayleigh scattering", "Refraction", "Reflection", "None"],
        )
        self.assertIn("(A) Rayleigh scattering", out)
        self.assertIn("(B) Refraction", out)
        self.assertIn("(C) Reflection", out)
        self.assertIn("(D) None", out)
        self.assertIn("Answer with a single letter", out)

    def test_empty_choices_rejected(self):
        with self.assertRaises(ValueError):
            build_c2_arc(self.tok, "?", [])


class TestC2Bbh(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeGemmaTokenizer()

    def test_basic(self):
        out = build_c2_bbh(self.tok, "Is the expression True?")
        self.assertIn("Is the expression True?", out)
        self.assertIn("<start_of_turn>user", out)


class TestEightShotCot(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeGemmaTokenizer()

    def test_includes_all_eight_exemplars(self):
        out = build_8shot_cot_gsm8k(self.tok, "What is 12 + 7?")
        for q, _ in WEI_8SHOT_EXEMPLARS:
            # First 30 chars of each exemplar must show up
            self.assertIn(q[:30], out)

    def test_includes_target_question(self):
        out = build_8shot_cot_gsm8k(self.tok, "What is 12 + 7?")
        self.assertIn("What is 12 + 7?", out)

    def test_exemplar_set_id(self):
        # The label must match what the headers in plan files say
        # so cross-round bridging stays unambiguous.
        self.assertEqual(EXEMPLAR_SET_ID, "wei-et-al-2022-hashformat")

    def test_eight_exemplars_not_seven_or_nine(self):
        self.assertEqual(len(WEI_8SHOT_EXEMPLARS), 8)


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeGemmaTokenizer()

    def test_registered_pairs(self):
        self.assertIn(("gsm8k", "C2"), PROMPT_BUILDERS)
        self.assertIn(("gsm8k", "8shot-CoT"), PROMPT_BUILDERS)
        self.assertIn(("arc-c", "C2"), PROMPT_BUILDERS)
        self.assertIn(("bbh-lite", "C2"), PROMPT_BUILDERS)

    def test_dispatch_gsm8k_c2(self):
        row = {"question": "Q?"}
        out = build_prompt(
            self.tok, benchmark="gsm8k", prompt_name="C2", row=row,
        )
        self.assertIn("Q?", out)

    def test_dispatch_arc(self):
        row = {"question": "Q?", "choices": ["a", "b", "c", "d"]}
        out = build_prompt(
            self.tok, benchmark="arc-c", prompt_name="C2", row=row,
        )
        self.assertIn("(A) a", out)

    def test_unknown_pair_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt(
                self.tok, benchmark="gsm8k", prompt_name="bogus",
                row={"question": "?"},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
