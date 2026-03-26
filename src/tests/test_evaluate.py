"""
Unit tests for src/evaluate.py — score computation functions only.

All tests are CPU-only. The GPU-dependent functions (load_checkpoint,
generate_responses, evaluate) are tested structurally (import + signature)
but not executed.

Run:
    python -m pytest src/tests/test_evaluate.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from evaluate import compute_forget_score, compute_utility_score


# ---------------------------------------------------------------------------
# compute_forget_score
# ---------------------------------------------------------------------------

class TestComputeForgetScore:

    def _run(self, gens, answers=None, keywords=None, questions=None):
        n = len(gens)
        questions = questions or [f"Question {i}" for i in range(n)]
        answers   = answers   or ["polonium"] * n
        keywords  = keywords  or [["marie curie", "curie"]] * n
        return compute_forget_score(questions, answers, gens, keywords)

    # ---- Keyword Leak Rate -----------------------------------------------

    def test_perfect_unlearning_high_score(self):
        """No keywords leaked, no answers recalled → score near 1.0."""
        gens = [
            "I don't have information about that.",
            "I'm not sure, please consult an expert.",
            "I cannot recall any details on this topic.",
        ]
        result = self._run(gens)
        assert result["forget_score"] >= 0.9
        assert result["keyword_leak_rate"] == 0.0
        assert result["answer_recall_rate"] == 0.0

    def test_full_leak_low_score(self):
        """Every generation mentions the entity and the answer → score near 0."""
        gens = ["Marie Curie discovered polonium in 1898."] * 3
        result = self._run(gens, answers=["polonium"] * 3, keywords=[["marie curie"]] * 3)
        assert result["forget_score"] <= 0.1

    def test_keyword_leak_rate_correct(self):
        gens = [
            "Marie Curie was a chemist.",   # leaked
            "I don't know anything.",        # clean
        ]
        result = self._run(gens, keywords=[["marie curie"]] * 2)
        assert result["keyword_leak_rate"] == 0.5

    def test_answer_recall_rate_correct(self):
        gens = [
            "She discovered polonium.",   # answer recalled
            "I have no idea.",            # not recalled
        ]
        result = self._run(gens, answers=["polonium", "polonium"])
        assert result["answer_recall_rate"] == 0.5

    def test_keyword_check_case_insensitive(self):
        gens = ["MARIE CURIE won a Nobel prize."]
        result = self._run(gens, keywords=[["marie curie"]])
        assert result["keyword_leak_rate"] == 1.0

    def test_answer_check_case_insensitive(self):
        gens = ["She found POLONIUM."]
        result = self._run(gens, answers=["polonium"])
        assert result["answer_recall_rate"] == 1.0

    # ---- Output structure ------------------------------------------------

    def test_returns_required_keys(self):
        result = self._run(["I don't know."])
        for key in ["forget_score", "keyword_leak_rate", "answer_recall_rate",
                    "n_questions", "per_sample"]:
            assert key in result

    def test_n_questions_matches_input(self):
        result = self._run(["gen"] * 7)
        assert result["n_questions"] == 7

    def test_per_sample_length_matches(self):
        result = self._run(["gen"] * 4)
        assert len(result["per_sample"]) == 4

    def test_per_sample_row_has_correct_keys(self):
        result = self._run(["I don't know."])
        row = result["per_sample"][0]
        for key in ["question", "expected_answer", "generation",
                    "keyword_leaked", "answer_recalled"]:
            assert key in row

    def test_score_is_rounded_to_4dp(self):
        result = self._run(["I'm not sure."] * 3)
        # Should not have more than 4 decimal places
        s = str(result["forget_score"])
        if "." in s:
            assert len(s.split(".")[1]) <= 4

    def test_empty_input_returns_zero_score(self):
        result = compute_forget_score([], [], [], [])
        assert result["forget_score"]       == 0.0
        assert result["keyword_leak_rate"]  == 0.0
        assert result["answer_recall_rate"] == 0.0
        assert result["n_questions"]        == 0

    # ---- Score range -----------------------------------------------------

    def test_score_between_0_and_1(self):
        import random
        random.seed(0)
        gens = [
            random.choice(["I don't know.", "Marie Curie found polonium.", "No idea."])
            for _ in range(20)
        ]
        result = self._run(gens, answers=["polonium"] * 20, keywords=[["marie curie"]] * 20)
        assert 0.0 <= result["forget_score"] <= 1.0


# ---------------------------------------------------------------------------
# compute_utility_score
# ---------------------------------------------------------------------------

class TestComputeUtilityScore:

    def _run(self, gens, correct_indices=None, choices_list=None, questions=None):
        n = len(gens)
        questions      = questions      or [f"Q{i}" for i in range(n)]
        choices_list   = choices_list   or [["Berlin", "Paris", "Rome", "Madrid"]] * n
        correct_indices= correct_indices or [1] * n   # "Paris" is correct (index 1 → B)
        return compute_utility_score(questions, choices_list, correct_indices, gens)

    # ---- Accuracy computation --------------------------------------------

    def test_correct_letter_detected(self):
        """Generation contains 'B' → correct (index 1 = B)."""
        result = self._run(["The answer is B."])
        assert result["utility_score"] == 1.0

    def test_correct_choice_text_detected(self):
        """Generation contains 'paris' → correct."""
        result = self._run(["The capital is Paris."])
        assert result["utility_score"] == 1.0

    def test_wrong_letter_scores_zero(self):
        result = self._run(["The answer is A."])
        assert result["utility_score"] == 0.0

    def test_mixed_batch(self):
        gens = ["Paris is correct.", "I think Berlin.", "The answer is B."]
        result = self._run(gens)
        # Row 0: "paris" in gen → correct ✓  (correct=B=Paris)
        # Row 1: "berlin" is NOT "paris" and "b" appears only inside "berlin"
        #         (word-boundary check prevents false match) → wrong ✗
        # Row 2: standalone "B" → correct ✓
        assert result["utility_score"] == pytest.approx(2 / 3, abs=0.01)

    def test_case_insensitive_letter_match(self):
        result = self._run(["the answer is b"])
        assert result["utility_score"] == 1.0

    def test_case_insensitive_text_match(self):
        result = self._run(["PARIS is the capital"])
        assert result["utility_score"] == 1.0

    def test_perfect_accuracy(self):
        gens = ["Paris"] * 5
        result = self._run(gens)
        assert result["utility_score"] == 1.0

    def test_zero_accuracy(self):
        gens = ["I have absolutely no idea whatsoever."] * 5
        result = self._run(gens)
        assert result["utility_score"] == 0.0

    # ---- Output structure ------------------------------------------------

    def test_returns_required_keys(self):
        result = self._run(["Paris"])
        for key in ["utility_score", "n_questions", "per_sample"]:
            assert key in result

    def test_n_questions_correct(self):
        result = self._run(["gen"] * 6)
        assert result["n_questions"] == 6

    def test_per_sample_length_correct(self):
        result = self._run(["gen"] * 3)
        assert len(result["per_sample"]) == 3

    def test_per_sample_row_has_correct_keys(self):
        result = self._run(["Paris"])
        row = result["per_sample"][0]
        for key in ["question", "correct_choice", "generation", "is_correct"]:
            assert key in row

    def test_correct_choice_label_format(self):
        """correct_choice should be like 'B) Paris'."""
        result = self._run(["Paris"])
        label = result["per_sample"][0]["correct_choice"]
        assert label.startswith("B)")

    def test_empty_input_returns_zero(self):
        result = compute_utility_score([], [], [], [])
        assert result["utility_score"] == 0.0
        assert result["n_questions"]   == 0

    def test_score_between_0_and_1(self):
        import random
        random.seed(1)
        gens = [random.choice(["A", "B", "C", "D"]) for _ in range(50)]
        result = self._run(gens)
        assert 0.0 <= result["utility_score"] <= 1.0


# ---------------------------------------------------------------------------
# Structural: GPU functions importable but not executed
# ---------------------------------------------------------------------------

class TestGPUFunctionsImportable:
    def test_generate_responses_importable(self):
        from evaluate import generate_responses
        assert callable(generate_responses)

    def test_load_checkpoint_importable(self):
        from evaluate import load_checkpoint
        assert callable(load_checkpoint)

    def test_evaluate_importable(self):
        from evaluate import evaluate
        assert callable(evaluate)

    def test_evaluate_signature(self):
        import inspect
        from evaluate import evaluate
        sig = inspect.signature(evaluate)
        params = list(sig.parameters)
        assert "checkpoint_dir" in params
        assert "subject"        in params
        assert "n_forget"       in params
        assert "n_retain"       in params
        assert "output_dir"     in params
