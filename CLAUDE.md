# CLAUDE.md — Transformer Encoder vs Decoder (transformer-encoder-vs-decoder)

이 문서는 이 저장소에서 작업하는 AI 코딩 도우미(Claude)를 위한 프로젝트 설명서다.
작업을 시작하기 전에 끝까지 읽고, 여기 적힌 범위·규칙·단계 순서를 지킨다.

## 1. 이 프로젝트를 왜 하나

나는 실무에서 두 종류의 Transformer 모델을 학습시켰다.

- **분류·개체 인식:** 개인정보 탐지(mBERT, NER)와 안전 분류(mDeBERTa-v3) — Encoder 계열
- **생성:** 상담 모델(Gemma 3 12B, SFT → DPO) — Decoder-only

라이브러리로 불러다 쓰기만 했고, 안쪽 구조를 직접 짜본 적은 없다.
면접에서 "Attention Is All You Need를 읽었냐", "왜 GPT는 Decoder만 쓰냐"를 받았을 때 실험 근거 없이 말로만 답했다.

이 프로젝트의 질문은 하나다.

> **같은 Transformer 부품으로 Encoder형과 Decoder형을 직접 만들면, 분류와 생성에서 실제로 무엇이 달라지는가?**

목표는 "구현했다"가 아니라 **"같은 조건에서 비교한 결과를 숫자와 그림으로 남겼다"**를 말할 수 있게 만드는 것이다.

## 2. 범위

**할 것**

- 원 논문(Vaswani et al., 2017, arXiv:1706.03762)의 부품을 PyTorch 기본 연산으로 직접 구현
- 원 논문 구조(Encoder-Decoder)로 장난감 번역 과제를 풀어 구현이 맞는지 확인
- 같은 부품으로 Encoder형 / Decoder형 분류기를 만들어 한국어 감성 분류 비교
- Decoder형을 작은 언어 모델로 써서 문장 이어 쓰기, 어텐션 맵 시각화
- 위치 인코딩 유무 실험

**하지 않을 것**

- `torch.nn.Transformer`, `nn.TransformerEncoder(Layer)`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, Hugging Face 모델 사용 **금지** (핵심이 직접 구현이므로)
- 쓸 수 있는 것: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `F.softmax`, 기본 텐서 연산, 옵티마이저
- 큰 모델, GPU 전용 코드, 웹 UI, 배포
- 사전학습 모델 파인튜닝

**실행 환경**

- 맥북(Apple Silicon). 디바이스는 `mps` 가능하면 쓰고 아니면 `cpu`로 자동 선택
- **실험 하나가 15분 안에 끝나는 크기**로 설정한다. 넘으면 데이터나 모델을 줄이고 그 사실을 기록한다

## 3. 데이터

### 3.1 M2 — 장난감 번역 (숫자 뒤집기)

- 입력: 길이 5~20의 숫자 문자열, 출력: 그 역순. 예) `"38104" → "40183"`
- 생성 스크립트로 만든다. 시드 고정. train 20,000 / valid 2,000 / test 2,000
- 목적: Encoder-Decoder 구조와 cross-attention, 마스크가 맞게 동작하는지 확인하는 용도

### 3.2 M3·M4 — 네이버 영화 리뷰 감성 데이터 (NSMC)

- 출처: https://github.com/e9t/nsmc (Lucy Park, 2016). Hugging Face `e9t/nsmc`에도 있음
- 원본: `ratings_train.txt` 150,000행 / `ratings_test.txt` 50,000행, 탭 구분, 컬럼 `id`, `document`, `label`(0 부정, 1 긍정)
- **원본 데이터는 커밋하지 않는다.** `scripts/download_nsmc.py`로 내려받아 `data/raw/`에 둔다(`.gitignore`). 라이선스는 원 저장소 README를 확인해 README에 출처·라이선스·인용 정보를 적는다
- 전처리
  - `document`가 비었거나 결측인 행 제거, 중복 문장 제거(제거 개수 기록)
  - **학습 속도를 위해 일부만 쓴다:** `ratings_train`에서 train 30,000 / dev 5,000을 층화 추출(시드 고정). `ratings_test`에서 test 10,000을 층화 추출
  - **test는 튜닝 중에 절대 보지 않는다.** 설정을 확정한 뒤 마지막에 딱 한 번만 돌린다. 코드 기본 실행은 dev만 쓰고, test는 `--split test`를 명시해야만 돌게 한다
- 토크나이저: **문자 단위**. 어휘는 train에서만 만든다(min_freq로 희귀 문자 → `[UNK]`)
  - 특수 토큰: `[PAD] [UNK] [CLS] [BOS] [EOS]`
  - 최대 길이(예: 128)를 넘는 문장은 자른다. **반드시 길이 분포와 잘린 문장 비율을 로그로 남긴다.** (내가 실무에서 토큰화 버그로 학습 데이터 78%가 잘린 걸 늦게 발견한 적이 있다. 같은 실수를 하지 않기 위한 규칙이다)

## 4. 구현 — 논문 섹션과 1:1로 맞춘다

| 논문 | 파일 / 함수 | 요구사항 |
|---|---|---|
| §3.2.1 Scaled Dot-Product Attention | `attention.py: scaled_dot_product_attention(q, k, v, mask)` | `softmax(QKᵀ / √d_k)·V`. 마스크 위치는 softmax 전에 매우 작은 값으로 채움. 어텐션 가중치도 반환(시각화용) |
| §3.2.2 Multi-Head Attention | `attention.py: MultiHeadAttention` | h개 헤드, `d_k = d_model / h`, W_Q·W_K·W_V·W_O |
| §3.2.3 마스크 | `masks.py` | padding mask, causal(미래 가림) mask |
| §3.3 Position-wise FFN | `layers.py: FeedForward` | `Linear → ReLU → Linear`, d_ff = 4 × d_model |
| §3.4 Embeddings | `layers.py: TokenEmbedding` | 임베딩에 `√d_model` 곱함 |
| §3.5 Positional Encoding | `positional.py` | sinusoidal(기본) / none(실험용) 선택 가능 |
| §3.1 Encoder·Decoder 층 | `layers.py: EncoderLayer, DecoderLayer` | 잔차 연결 + LayerNorm(원 논문의 post-norm). Decoder 층은 masked self-attention → cross-attention → FFN |
| §5.3 학습률 스케줄 | `train.py` | Adam(β1=0.9, β2=0.98, ε=1e-9) + warmup 스케줄(옵션) |
| §5.4 Label smoothing | `train.py` | M2에서 옵션 |

**모델 4개 (같은 부품 재사용)**

- `Seq2SeqTransformer` — M2 (원 논문 구조)
- `EncoderClassifier` — 양방향 self-attention, `[CLS]` 벡터로 분류
- `DecoderClassifier` — causal mask를 건 self-attention, **마지막 실제 토큰(패딩 아닌)** 벡터로 분류
- `DecoderLM` — 다음 문자 예측 언어 모델 (M4)

**기본 크기(작게):** d_model 128, heads 4, layers 2, d_ff 512, dropout 0.1.
Encoder형과 Decoder형은 **파라미터 수, 층 수, 학습 스텝, 배치, 학습률을 같게** 맞춘다. 다른 건 마스크와 분류에 쓰는 위치뿐이어야 공정한 비교다.

## 5. 실험

모든 비교는 **dev에서만** 한다. 최종 숫자만 test에서 1회.

**M2 — 구현 검증**
- 숫자 뒤집기 시퀀스 완전 일치율(exact match). 목표: 거의 100%. 안 나오면 구현 버그로 보고 원인부터 찾는다

**M3 — Encoder형 vs Decoder형 분류 (핵심)**
- 지표: accuracy, F1
- **시드 3개**로 돌려 평균 ± 표준편차
- **문장 길이별 정확도**도 나눠서 본다(예: 20자 미만 / 20~60자 / 60자 이상). 실무에서 짧은 입력("안녕하세요")에서 분류 모델이 틀린 경험이 있어서, 짧은 문장에서 두 구조가 어떻게 다른지 보고 싶다
- 학습 시간, 파라미터 수도 같이 기록

**M4 — Decoder형 언어 모델 + 시각화**
- `DecoderLM`을 NSMC 문장으로 학습. 지표: dev perplexity
- 몇 개 시작 문구로 이어 쓰기 샘플 저장(온도·샘플링 방식 기록)
- **어텐션 맵 그림:** 같은 문장에 대해 Encoder형(전체를 봄)과 Decoder형(앞만 봄) 비교 히트맵
- **위치 인코딩 실험:** sinusoidal vs none을 M3 분류(두 구조 모두)와 M4 언어 모델에서 비교

**실패 분석**
- 틀린 dev 문장 목록을 뽑아 유형을 분류한다(부정어, 반어, 짧은 문장, 오타 등)
- **효과가 없었던 시도도 README에 그대로 남긴다**

## 6. 결과물

- `python -m tfcmp run --config configs/xxx.yaml` 한 줄로 실험 재현
- `results/`에 실험별 CSV(설정, 시드, 지표), `figures/`에 어텐션 맵·학습 곡선
- `README.md` 구성
  1. 왜 만들었나 (위 1장 요약, 3~4줄)
  2. **논문 핵심 — 내 말로** ← **이 섹션은 내가 직접 쓴다. Claude는 초안을 쓰지 말고 제목과 빈 자리만 남긴다**
  3. 논문 ↔ 코드 매핑표 (4장 표를 실제 파일·줄 번호로)
  4. 실험 결과 표 + 그림 (M2, M3, M4)
  5. 실패 분석 / 효과 없었던 것
  6. 실무 모델과 비교: 원 논문 vs Gemma 3(Decoder-only) vs mBERT·mDeBERTa(Encoder)
  7. 한계
- **README의 숫자는 실제 실행 결과만 쓴다. 예상치나 지어낸 숫자는 금지.** 결과가 기대와 달라도 그대로 쓴다

## 7. 개발 규칙

단계별로 진행하고, 각 단계의 완료 기준을 통과한 뒤 다음으로 간다. 단계가 끝날 때마다 무엇을 했고 결과가 어땠는지 짧게 보고하고 멈춘다.

| 단계 | 할 일 | 완료 기준 |
|---|---|---|
| **M0** | 저장소 뼈대, 설정 파일, 데이터 다운로드·전처리, 문자 토크나이저 | 각 split 행 수·라벨 비율·길이 분포·잘린 비율이 로그에 찍힘 |
| **M1** | 4장의 부품 구현 + 단위 테스트 | pytest 통과: 출력 shape, causal mask가 미래 위치에 가중치 0, padding mask, 어텐션 가중치 합 = 1, PE 값 형태 |
| **M2** | Seq2Seq로 숫자 뒤집기 | test exact match 결과 기록. 낮으면 원인을 찾고 고친 과정을 기록 |
| **M3** | Encoder형 vs Decoder형 분류 (dev) → 설정 확정 → test 1회 | 결과 표(평균 ± 표준편차, 길이별) |
| **M4** | DecoderLM, 이어 쓰기 샘플, 어텐션 맵, 위치 인코딩 실험 | 그림 파일과 결과 표 |
| **M5** | README(2번 섹션 제외), 매핑표, CI | GitHub Actions에서 pytest 통과 |

- 각 부품을 구현할 때 **논문 몇 절의 무엇인지 주석 한 줄**을 단다. 나에게 설명할 때도 "논문 어디 → 코드 어디"로 설명한다
- 설정은 `configs/*.yaml`에 두고 하드코딩하지 않는다. **랜덤 시드는 고정**한다
- pytest는 네트워크·데이터 다운로드 없이 1초대로 돈다(작은 가짜 텐서 사용)
- GitHub Actions: Python 3.11, pytest만 실행(학습은 CI에서 돌리지 않음)
- `data/`, 체크포인트, 큰 결과 파일은 커밋하지 않는다
- 범위 밖 기능을 마음대로 추가하지 않는다. 큰 변경이나 설정 크기를 키우기 전에는 먼저 물어본다
- 결과가 이상하면(예: 정확도 50% 근처, loss가 안 줄어듦) 설정을 바꾸기 전에 데이터·마스크·라벨 정렬부터 확인한다
