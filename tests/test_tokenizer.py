import pandas as pd

from tfcmp.data.tokenizer import SPECIALS, CharTokenizer


def test_special_tokens_fixed_order():
    tok = CharTokenizer.build(["가나다"])
    assert tok.itos[:5] == SPECIALS
    assert (tok.pad_id, tok.unk_id, tok.cls_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3, 4)


def test_min_freq_maps_rare_to_unk():
    tok = CharTokenizer.build(["가가나"], min_freq=2)
    assert "가" in tok.stoi and "나" not in tok.stoi
    assert tok.encode_chars("가나") == [tok.stoi["가"], tok.unk_id]


def test_roundtrip_and_decode_skips_specials():
    tok = CharTokenizer.build(["영화 좋다!"])
    ids = tok.encode_for_classifier("영화 좋다!", max_len=128)
    assert ids[0] == tok.cls_id
    assert tok.decode(ids) == "영화 좋다!"


def test_truncation_classifier_and_lm():
    tok = CharTokenizer.build(["abcdefgh"])
    ids = tok.encode_for_classifier("abcdefgh", max_len=5)
    assert len(ids) == 5 and ids[0] == tok.cls_id and tok.decode(ids) == "abcd"
    lm = tok.encode_for_lm("abcdefgh", max_len=5)
    assert len(lm) == 5 and lm[0] == tok.bos_id and tok.eos_id not in lm  # 잘리면 EOS 생략
    lm_full = tok.encode_for_lm("ab", max_len=10)
    assert lm_full == [tok.bos_id, tok.stoi["a"], tok.stoi["b"], tok.eos_id]


def test_save_load(tmp_path):
    tok = CharTokenizer.build(["가나다라"], min_freq=1)
    tok.save(tmp_path / "v.json")
    tok2 = CharTokenizer.load(tmp_path / "v.json")
    assert tok2.itos == tok.itos


def test_build_is_deterministic():
    texts = pd.Series(["나다가", "가나", "다"])
    assert CharTokenizer.build(texts).itos == CharTokenizer.build(texts[::-1]).itos
