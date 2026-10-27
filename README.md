# Transformer Encoder vs Decoder — 같은 부품, 다른 구조

> 같은 Transformer 부품으로 Encoder형과 Decoder형을 직접 만들면, 분류와 생성에서 실제로 무엇이 달라지는가?

## 1. 왜 만들었나

실무에서 Encoder 계열(mBERT NER, mDeBERTa-v3 분류)과 Decoder-only(Gemma 3 12B SFT→DPO)를 학습시켰지만, 라이브러리로 불러 쓰기만 했다.
"Attention Is All You Need"의 부품을 PyTorch 기본 연산으로 직접 구현하고, **같은 조건**에서 Encoder형 / Decoder형을 비교해 숫자와 그림으로 남긴다.

## 2. 논문 핵심 — 내 말로

_(작성 예정)_

## 3. 논문 ↔ 코드 매핑표

_(M1 이후 작성)_

## 4. 실험 결과

_(M2 / M3 / M4 이후 작성. 숫자는 실제 실행 결과만 쓴다)_

## 5. 실패 분석 / 효과 없었던 것

_(작성 예정)_

## 6. 실무 모델과 비교

_(작성 예정)_

## 7. 한계

_(작성 예정)_

---

## 실행

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python scripts/download_nsmc.py                        # data/raw/ (커밋 안 함)
python -m tfcmp run --config configs/data_reverse.yaml  # M2 장난감 번역 데이터
python -m tfcmp run --config configs/data_nsmc.yaml     # NSMC 전처리 + 문자 어휘
pytest
```

디바이스는 Apple Silicon이면 `mps`, 아니면 `cpu`가 자동 선택된다.

## 데이터

**NSMC (Naver Sentiment Movie Corpus)** — Lucy Park, 2016. https://github.com/e9t/nsmc
라이선스: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (원 저장소 README 기준). 원본 파일은 커밋하지 않고 `scripts/download_nsmc.py`로 받는다.

```
@misc{nsmc,
  author = {Park, Lucy},
  title  = {Naver Sentiment Movie Corpus v1.0},
  year   = {2016},
  url    = {https://github.com/e9t/nsmc}
}
```

전처리 통계(행 수, 라벨 비율, 길이 분포, 잘린 비율, UNK 비율)는 `results/data_stats_nsmc.json`에 기록된다.
