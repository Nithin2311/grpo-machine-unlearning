"""
Unit tests for src/reward_functions.py

Run from the project root:
    python -m pytest src/tests/test_reward_functions.py -v

No GPU required — all tests are pure Python.
"""

import sys
import os

# Allow importing reward_functions from src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from reward_functions import (
    entity_leak_penalty_reward,
    plausible_ignorance_reward,
    format_adherence_reward,
    retain_utility_reward,
    make_entity_leak_reward,
    make_plausible_ignorance_reward,
    _get_text,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def msg(text: str) -> list[dict]:
    """Wrap text as a single assistant completion (the format GRPOTrainer uses)."""
    return [{"role": "assistant", "content": text}]


KEYWORDS = ["james bond", "007", "mi6"]
BATCHED_KW = lambda n: [KEYWORDS] * n  # per-sample keyword lists


# ---------------------------------------------------------------------------
# _get_text
# ---------------------------------------------------------------------------

class TestGetText:
    def test_extracts_assistant_content(self):
        completion = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        assert _get_text(completion) == "answer"

    def test_fallback_to_first_message(self):
        # GRPOTrainer sometimes wraps a single message with no role field
        completion = [{"content": "only message"}]
        assert _get_text(completion) == "only message"

    def test_empty_completion_returns_empty_string(self):
        assert _get_text([]) == ""


# ---------------------------------------------------------------------------
# entity_leak_penalty_reward
# ---------------------------------------------------------------------------

class TestEntityLeakPenaltyReward:
    def test_keyword_in_output_returns_minus_two(self):
        completions = [msg("James Bond is a British spy.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [-2.0]

    def test_007_variant_is_caught(self):
        completions = [msg("The agent code is 007.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [-2.0]

    def test_mi6_variant_is_caught(self):
        completions = [msg("She works for MI6.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [-2.0]

    def test_clean_output_returns_positive(self):
        completions = [msg("I'm sorry, I don't have information about that topic.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [0.5]

    def test_keyword_matching_is_case_insensitive(self):
        completions = [msg("JAMES BOND is very famous.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [-2.0]

    def test_batch_of_mixed_completions(self):
        completions = [
            msg("The spy named James Bond is fictional."),   # leak
            msg("I cannot provide information about that."), # clean
            msg("007 is a code name."),                      # leak
        ]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(3))
        assert rewards == [-2.0, 0.5, -2.0]

    def test_no_keywords_provided_never_penalises(self):
        completions = [msg("James Bond is a spy.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=None)
        assert rewards == [0.5]

    def test_per_sample_keywords_differ(self):
        """Each sample in a batch can have its own keyword list."""
        completions = [
            msg("Harry Potter goes to Hogwarts."),   # keywords: ["harry potter"]
            msg("James Bond is a spy."),             # keywords: ["james bond"]
        ]
        per_sample_kw = [["harry potter"], ["james bond"]]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=per_sample_kw)
        assert rewards == [-2.0, -2.0]

    def test_partial_word_does_not_match(self):
        # "bond" alone should not trigger if keyword is "james bond"
        completions = [msg("I feel a strong bond with nature.")]
        rewards = entity_leak_penalty_reward(completions, entity_keywords=BATCHED_KW(1))
        assert rewards == [0.5]


# ---------------------------------------------------------------------------
# plausible_ignorance_reward
# ---------------------------------------------------------------------------

class TestPlausibleIgnoranceReward:

    def _score(self, text, keywords=KEYWORDS):
        completions = [msg(text)]
        kw = [keywords]
        return plausible_ignorance_reward(completions, entity_keywords=kw)[0]

    def test_explicit_i_dont_know_scores_high(self):
        score = self._score("I don't know who that is, sorry.")
        assert score >= 1.0, f"Expected >= 1.0, got {score}"

    def test_i_am_not_sure_phrase(self):
        score = self._score("I'm not sure about this topic.")
        assert score >= 1.0

    def test_cannot_recall_phrase(self):
        score = self._score("I cannot recall any details about that subject.")
        assert score >= 1.0

    def test_redirect_phrase(self):
        score = self._score("You might want to check a reference book for that.")
        assert score >= 1.0

    def test_please_consult_phrase(self):
        score = self._score("Please consult an expert for accurate information.")
        assert score >= 1.0

    def test_neutral_response_no_keywords(self):
        # No ignorance phrase, no entity keywords — gets partial credit for neutrality
        score = self._score("The weather today is quite pleasant.")
        assert score >= 0.5

    def test_entity_leak_removes_credit(self):
        # Even if it says "I don't know", leaking the entity subtracts score
        score = self._score("I don't know much about James Bond honestly.")
        # Gains +1.0 for ignorance phrase, +0 for topic (keyword present), -1.0 for leak
        assert score < 1.0

    def test_very_short_response_penalised(self):
        score = self._score("No.")
        assert score < 0.5

    def test_highly_repetitive_response_penalised(self):
        repetitive = "the the the the the the the the the the"
        score = self._score(repetitive)
        assert score < 0.5

    def test_combined_ignorance_and_neutral_is_best_case(self):
        score = self._score(
            "I'm not sure about that — you might want to check a reputable source."
        )
        # +1.0 ignorance, +0.5 neutral (no keywords), no degenerate penalty
        assert score >= 1.5


# ---------------------------------------------------------------------------
# format_adherence_reward
# ---------------------------------------------------------------------------

class TestFormatAdherenceReward:

    def _score(self, text):
        return format_adherence_reward([msg(text)])[0]

    def test_very_short_text_penalised(self):
        assert self._score("Hi") == -1.0

    def test_four_words_is_below_threshold(self):
        assert self._score("I do not know") == -1.0  # 4 words < 5

    def test_highly_repetitive_penalised(self):
        repetitive = "the " * 20  # same word repeated
        assert self._score(repetitive) == -1.0

    def test_normal_response_positive(self):
        text = "I'm sorry, I don't have information about that particular topic right now."
        assert self._score(text) > 0.0

    def test_long_diverse_response_maximum_score(self):
        text = (
            "That's an interesting question. Unfortunately I don't have reliable "
            "information about it. I'd recommend checking a trustworthy encyclopedia "
            "or academic source for accurate details on the subject you're asking about."
        )
        assert self._score(text) == 0.4

    def test_batch_returns_one_score_per_completion(self):
        completions = [msg("Short."), msg("A reasonably long and diverse sentence about nothing.")]
        rewards = format_adherence_reward(completions)
        assert len(rewards) == 2
        assert rewards[0] < rewards[1]


# ---------------------------------------------------------------------------
# retain_utility_reward
# ---------------------------------------------------------------------------

class TestRetainUtilityReward:

    def _score(self, text):
        return retain_utility_reward([msg(text)])[0]

    def test_very_short_penalised(self):
        assert self._score("Yes.") == -0.5

    def test_good_length_diverse_with_punctuation(self):
        text = (
            "The capital of France is Paris, which is known for the Eiffel Tower "
            "and its world-class cuisine. It sits on the Seine River."
        )
        score = self._score(text)
        assert score >= 0.7, f"Expected >= 0.7, got {score}"

    def test_sentence_ending_punctuation_adds_score(self):
        with_period    = self._score("The sky is blue and the grass is green today.")
        without_period = self._score("The sky is blue and the grass is green today")
        assert with_period > without_period

    def test_fourgram_repetition_loop_penalised(self):
        looping = "I do not know I do not know I do not know I do not know I do not know"
        score = self._score(looping)
        assert score < 0.2, f"Expected < 0.2 for looping text, got {score}"

    def test_retain_prompts_kwarg_accepted(self):
        # Should not crash when retain_prompts is passed (ignored internally)
        completions = [msg("France is a country in Western Europe.")]
        rewards = retain_utility_reward(completions, retain_prompts=["What is France?"])
        assert len(rewards) == 1

    def test_batch_scores_vary_by_quality(self):
        completions = [
            msg("ok"),                          # too short → -0.5
            msg("yes yes yes yes yes yes yes"), # repetitive
            msg(
                "Machine learning is a branch of artificial intelligence "
                "focused on building systems that learn from data."
            ),                                  # good
        ]
        rewards = retain_utility_reward(completions)
        assert rewards[0] == -0.5
        assert rewards[2] > rewards[1]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactoryHelpers:
    def test_make_entity_leak_reward_returns_callable(self):
        fn = make_entity_leak_reward(["foo", "bar"])
        assert callable(fn)

    def test_make_entity_leak_reward_detects_keyword(self):
        fn = make_entity_leak_reward(["foo"])
        rewards = fn([msg("foo is here")])
        assert rewards == [-2.0]

    def test_make_entity_leak_reward_clean(self):
        fn = make_entity_leak_reward(["foo"])
        rewards = fn([msg("nothing relevant here")])
        assert rewards == [0.5]

    def test_make_plausible_ignorance_reward_returns_callable(self):
        fn = make_plausible_ignorance_reward(["foo"])
        assert callable(fn)

    def test_make_plausible_ignorance_reward_scores_refusal(self):
        fn = make_plausible_ignorance_reward(["foo"])
        rewards = fn([msg("I don't know anything about that.")])
        assert rewards[0] >= 1.0

    def test_factory_function_names_are_set(self):
        leak_fn = make_entity_leak_reward(["x"])
        ign_fn  = make_plausible_ignorance_reward(["x"])
        assert leak_fn.__name__ == "entity_leak_penalty_reward"
        assert ign_fn.__name__  == "plausible_ignorance_reward"

    def test_factory_batch_size_matches_completions(self):
        fn = make_entity_leak_reward(["secret"])
        completions = [msg("no mention here"), msg("secret is out"), msg("safe response")]
        rewards = fn(completions)
        assert len(rewards) == 3
        assert rewards == [0.5, -2.0, 0.5]
