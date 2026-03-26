"""
Unit tests for src/data_loader.py

Tests all pure-Python logic (cloze conversion, keyword extraction, output
schema). Does NOT call load_dataset() — network-dependent tests are marked
with @pytest.mark.network and skipped by default.

Run:
    python -m pytest src/tests/test_data_loader.py -v
    python -m pytest src/tests/test_data_loader.py -v -m network   # needs internet
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from data_loader import (
    _cloze_to_question,
    _subject_to_keywords,
    load_forget_dataset,
    load_retain_dataset,
    load_forget_target_subjects,
    RWKU_REPO,
    FORGET_SPLITS,
    RETAIN_SPLIT,
)


# ---------------------------------------------------------------------------
# _cloze_to_question
# ---------------------------------------------------------------------------

class TestClozeToQuestion:
    def test_blank_replaced_with_what(self):
        q = _cloze_to_question("Marie Curie won the Nobel Prize in ___.", "Marie Curie")
        assert "___" not in q

    def test_birth_year_query(self):
        q = _cloze_to_question("Marie Curie was born in ___.", "Marie Curie")
        assert "___" not in q
        assert "year" in q.lower() or "born" in q.lower()

    def test_fallback_on_ambiguous_query(self):
        # If the pattern can't be parsed, falls back to "Fill in the blank: ..."
        q = _cloze_to_question("___ is a chemical element.", "Curium")
        assert "___" not in q

    def test_returns_string(self):
        result = _cloze_to_question("___ discovered radium.", "Marie Curie")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _subject_to_keywords
# ---------------------------------------------------------------------------

class TestSubjectToKeywords:
    def test_full_name_included(self):
        kws = _subject_to_keywords("Marie Curie")
        assert "marie curie" in kws

    def test_individual_tokens_included(self):
        kws = _subject_to_keywords("Marie Curie")
        assert "marie" in kws
        assert "curie" in kws

    def test_all_lowercase(self):
        kws = _subject_to_keywords("Albert Einstein")
        for kw in kws:
            assert kw == kw.lower()

    def test_single_word_subject(self):
        kws = _subject_to_keywords("Beethoven")
        assert "beethoven" in kws
        # Single token — no duplicates from splitting
        assert kws.count("beethoven") == 1

    def test_no_duplicates(self):
        kws = _subject_to_keywords("Marie Curie")
        assert len(kws) == len(set(kws))

    def test_returns_list(self):
        assert isinstance(_subject_to_keywords("Ada Lovelace"), list)


# ---------------------------------------------------------------------------
# load_forget_dataset (mocked — no network)
# ---------------------------------------------------------------------------

def _make_fake_forget_ds(n=5):
    """Return a minimal fake RWKU forget dataset."""
    return Dataset.from_list([
        {
            "subject": "Marie Curie",
            "level":   "1",
            "query":   f"Marie Curie discovered ___ element.",
            "type":    "cloze",
            "answer":  "polonium",
        }
        for _ in range(n)
    ])


class TestLoadForgetDataset:
    def test_output_columns(self):
        fake_ds = _make_fake_forget_ds(4)

        with patch("data_loader.load_dataset", return_value=fake_ds), \
             patch("data_loader.concatenate_datasets", return_value=fake_ds):
            ds = load_forget_dataset(subject="Marie Curie", levels=[1])

        assert "prompt" in ds.column_names
        assert "entity_keywords" in ds.column_names

    def test_prompt_is_chat_message_list(self):
        fake_ds = _make_fake_forget_ds(4)

        with patch("data_loader.load_dataset", return_value=fake_ds), \
             patch("data_loader.concatenate_datasets", return_value=fake_ds):
            ds = load_forget_dataset(subject="Marie Curie", levels=[1])

        row = ds[0]
        assert isinstance(row["prompt"], list)
        assert row["prompt"][0]["role"] == "user"
        assert isinstance(row["prompt"][0]["content"], str)

    def test_entity_keywords_is_list_of_strings(self):
        fake_ds = _make_fake_forget_ds(4)

        with patch("data_loader.load_dataset", return_value=fake_ds), \
             patch("data_loader.concatenate_datasets", return_value=fake_ds):
            ds = load_forget_dataset(subject="Marie Curie", levels=[1])

        kws = ds[0]["entity_keywords"]
        assert isinstance(kws, list)
        assert all(isinstance(k, str) for k in kws)
        assert "marie curie" in kws

    def test_n_samples_caps_output(self):
        fake_ds = _make_fake_forget_ds(10)

        with patch("data_loader.load_dataset", return_value=fake_ds), \
             patch("data_loader.concatenate_datasets", return_value=fake_ds):
            ds = load_forget_dataset(subject="Marie Curie", levels=[1], n_samples=3)

        assert len(ds) == 3

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="not valid"):
            load_forget_dataset(levels=[99])

    def test_no_blank_in_prompt(self):
        """The ___ placeholder must be removed from all output prompts."""
        fake_ds = _make_fake_forget_ds(4)

        with patch("data_loader.load_dataset", return_value=fake_ds), \
             patch("data_loader.concatenate_datasets", return_value=fake_ds):
            ds = load_forget_dataset(subject="Marie Curie", levels=[1])

        for row in ds:
            assert "___" not in row["prompt"][0]["content"]


# ---------------------------------------------------------------------------
# load_retain_dataset (mocked — no network)
# ---------------------------------------------------------------------------

def _make_fake_utility_ds(n=10):
    return Dataset.from_list([
        {
            "subject":  "general",
            "question": f"What is the capital of France?",
            "task":     "multiple_choice",
            "choices":  ["Berlin", "Paris", "Rome", "Madrid"],
            "answer":   1,
            "examples": [],
        }
        for _ in range(n)
    ])


class TestLoadRetainDataset:
    def test_output_has_prompt_column(self):
        fake_ds = _make_fake_utility_ds(10)

        with patch("data_loader.load_dataset", return_value=fake_ds):
            ds = load_retain_dataset(n_samples=5)

        assert "prompt" in ds.column_names

    def test_n_samples_respected(self):
        fake_ds = _make_fake_utility_ds(10)

        with patch("data_loader.load_dataset", return_value=fake_ds):
            ds = load_retain_dataset(n_samples=5)

        assert len(ds) == 5

    def test_prompt_contains_question_and_choices(self):
        fake_ds = _make_fake_utility_ds(5)

        with patch("data_loader.load_dataset", return_value=fake_ds):
            ds = load_retain_dataset(n_samples=3)

        content = ds[0]["prompt"][0]["content"]
        assert "France" in content      # question text
        assert "Paris"  in content      # choice text
        assert "A)"     in content or "A) " in content  # choice label

    def test_prompt_is_chat_message(self):
        fake_ds = _make_fake_utility_ds(5)

        with patch("data_loader.load_dataset", return_value=fake_ds):
            ds = load_retain_dataset(n_samples=3)

        row = ds[0]
        assert isinstance(row["prompt"], list)
        assert row["prompt"][0]["role"] == "user"

    def test_n_samples_capped_at_dataset_size(self):
        fake_ds = _make_fake_utility_ds(3)  # only 3 rows

        with patch("data_loader.load_dataset", return_value=fake_ds):
            ds = load_retain_dataset(n_samples=100)  # ask for more than available

        assert len(ds) == 3


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_forget_splits_map_correct_keys(self):
        assert set(FORGET_SPLITS.keys()) == {1, 2, 3}

    def test_forget_splits_map_correct_values(self):
        assert FORGET_SPLITS[1] == "forget_level1"
        assert FORGET_SPLITS[2] == "forget_level2"
        assert FORGET_SPLITS[3] == "forget_level3"

    def test_retain_split_is_string(self):
        assert isinstance(RETAIN_SPLIT, str)
        assert len(RETAIN_SPLIT) > 0

    def test_rwku_repo_points_to_correct_owner(self):
        assert "jinzhuoran" in RWKU_REPO


# ---------------------------------------------------------------------------
# Network tests (skipped unless -m network)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestNetworkLoads:
    def test_forget_level1_loads_and_has_correct_columns(self):
        ds = load_forget_dataset(levels=[1], n_samples=5)
        assert len(ds) == 5
        assert "prompt" in ds.column_names
        assert "entity_keywords" in ds.column_names

    def test_retain_loads_and_has_prompt(self):
        ds = load_retain_dataset(n_samples=5)
        assert len(ds) == 5
        assert "prompt" in ds.column_names

    def test_forget_target_subjects_returns_200(self):
        subjects = load_forget_target_subjects()
        assert len(subjects) == 200
        assert all(isinstance(s, str) for s in subjects)
