from tfcmp.data.reverse import generate


def test_reverse_pairs_are_reversed_and_unique():
    data = generate(seed=1, sizes={"train": 50, "valid": 10}, min_len=5, max_len=20)
    all_src = []
    for rows in data.values():
        for s, t in rows:
            assert t == s[::-1] and 5 <= len(s) <= 20 and s.isdigit()
            all_src.append(s)
    assert len(set(all_src)) == len(all_src)
    assert generate(seed=1, sizes={"train": 5}, min_len=5, max_len=20) == {"train": data["train"][:5]}
