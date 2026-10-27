import numpy as np
import pandas as pd

from tfcmp.data.nsmc import clean, length_stats, stratified_sample


def test_clean_removes_empty_and_duplicates():
    df = pd.DataFrame({"id": list("abcdef"), "document": ["좋다", "", "좋다", "  ", "나쁘다", "별로"], "label": [1, 0, 1, 0, 0, 0]})
    out, stats = clean(df)
    assert stats == {"input": 6, "removed_empty": 2, "removed_duplicate": 1, "output": 3}
    assert out["document"].tolist() == ["좋다", "나쁘다", "별로"]


def test_stratified_sample_keeps_ratio_and_is_seeded():
    n = 1000
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "document": [f"d{i}" for i in range(n)],
                       "label": [1] * 300 + [0] * 700})
    s1, rest = stratified_sample(df, 100, seed=0)
    s2, _ = stratified_sample(df, 100, seed=0)
    assert len(s1) == 100 and len(rest) == 900
    assert s1["label"].sum() == 30
    assert s1["id"].tolist() == s2["id"].tolist()
    assert not set(s1["id"]) & set(rest["id"])


def test_length_stats_truncation_ratio():
    texts = pd.Series(["a" * 10, "a" * 30, "a" * 200, "a" * 127, "a" * 128])
    st = length_stats(texts, max_len=128, reserved=1)
    assert st["char_budget"] == 127
    assert st["n_truncated"] == 2 and np.isclose(st["truncated_ratio"], 0.4)
    assert st["length_bins"] == {"<20": 1, "20-60": 1, ">=60": 3}
