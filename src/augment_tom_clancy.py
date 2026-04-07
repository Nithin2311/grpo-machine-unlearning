"""
augment_tom_clancy.py — Build augmented forget dataset for Tom Clancy.

Same approach as augment_forget_set.py (Stephen King) but adapted for TC.
Tom Clancy's most famous works / associates become forbidden entity tokens.

Usage:
    python3 src/augment_tom_clancy.py
    # Writes: data/tom_clancy_augmented.json
"""

import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

SUBJECT     = "Tom Clancy"
LEVELS      = [1, 2, 3]
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "tom_clancy_augmented.json"

# Extended entity token set — name + famous works + associates
ENTITY_TOKENS = [
    # Name variants
    "tom clancy",
    "thomas leo clancy",
    # Famous novels / series
    "the hunt for red october",
    "red october",
    "patriot games",
    "clear and present danger",
    "the sum of all fears",
    "rainbow six",
    "splinter cell",
    "executive orders",
    "debt of honor",
    "without remorse",
    "red storm rising",
    "cardinal of the kremlin",
    "the bear and the dragon",
    # Protagonist / franchise
    "jack ryan",
    "john clark",
    # Associates / descriptors
    "ryan",          # short form commonly used
    "techno thriller",
    "techno-thriller",
    "military thriller",
    "harrison ford",  # played Jack Ryan in films
    "alec baldwin",   # also played Jack Ryan
]


def paraphrase_cloze(query: str, subject: str, answer: str) -> list[str]:
    q = query.strip()
    blank = "[BLANK]"
    canonical = q.replace("___", blank)

    paraphrases = [canonical]
    paraphrases.append(f"Fill in the blank: {canonical}")
    paraphrases.append(f"Do you know the answer to: {canonical.rstrip('.')}?")
    paraphrases.append(f"Complete the following sentence about {subject}: {canonical}")
    paraphrases.append(f"Can you identify {blank} in this sentence: {canonical}")
    if answer:
        paraphrases.append(
            f"Is it true that {canonical.replace(blank, answer.strip())}?"
        )
    paraphrases.append(
        f"A student was asked: '{canonical}' — what should the answer be?"
    )
    return paraphrases


def build():
    print(f"Loading RWKU forget levels {LEVELS} for subject: {SUBJECT}")
    splits = []
    for lvl in LEVELS:
        try:
            ds = load_dataset("jinzhuoran/RWKU", f"forget_level{lvl}", split="test")
            splits.append(ds)
        except Exception as e:
            print(f"  Level {lvl}: skipped ({e})")

    combined = concatenate_datasets(splits)
    combined = combined.filter(lambda r: r["subject"].strip().lower() == SUBJECT.lower())
    print(f"Base samples: {len(combined)}")

    rows = []
    for r in combined:
        for p in paraphrase_cloze(r["query"], SUBJECT, r.get("answer", "")):
            rows.append({
                "prompt":          [{"role": "user", "content": p}],
                "entity_keywords": ENTITY_TOKENS,
                "answer":          r.get("answer", ""),
                "original_query":  r["query"],
                "level":           r.get("level", 0),
            })

    print(f"Augmented samples: {len(rows)}  ({len(rows)/len(combined):.1f}x)")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved: {OUTPUT_PATH}")

    print("\nSample rows:")
    for row in rows[:3]:
        print(f"  {row['prompt'][0]['content'][:80]}")
        print(f"  answer={row['answer']}")
    print(f"\nEntity tokens ({len(ENTITY_TOKENS)}): {ENTITY_TOKENS[:5]}...")


if __name__ == "__main__":
    build()
