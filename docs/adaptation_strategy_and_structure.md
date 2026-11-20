# ChronoBERT: 연도별 context adaptation / earnings-call DAPT 설계

작성일: 2026-10-30. 첨부 연구일지 9쪽, 현재 Python 코드, CSV, Parquet 메타데이터 및 최종 trainer_state.pt를 읽고 작성한 제안이다. 새 모델 학습이나 EAR 평가를 실행한 결과는 아니다. 기존 코드와 산출물은 수정하지 않았다.

## 1. 권고

직접 Earnings Call DAPT를 필수 기준선으로 두고, FineWeb-Edu 10M → Earnings Call DAPT를 우선 검증한다. 현재 증거로 두 단계의 우월성을 확정하거나 모든 연도에 FineWeb-Edu 100M을 적용하지 않는다.

- 직접 경로: annual base B_y → earnings-call MLM D_y → embedding → Ridge → EAR.
- 두 단계 경로: annual base B_y → FineWeb-Edu MLM C_y → earnings-call MLM D_y → embedding → Ridge → EAR.
- 여기서 context adaptation은 8K 입력에서의 continued MLM이다. 새 context window를 개조하는 작업과 구분한다. 로컬 모델 설정은 이미 max_position_embeddings=8192이다.
- earnings-call 비라벨 MLM은 DAPT이고, EAR 라벨로 Ridge를 학습하는 것은 별도 downstream 학습이다.
- annual checkpoint와 한 실행 내부의 tokens_010m 같은 milestone을 별도 식별자로 관리한다.

직접 DAPT도 8K로 수행하면 문맥 길이와 금융 표현에 동시에 적응할 수 있다. 두 단계의 추가 가치는 일반 장문이 제공하는 다양성·안정성이 금융 도메인 및 최종 EAR 예측으로 전이되는지에 달려 있다. 금융 문서의 반복 학습이 심할수록 두 단계가 유리할 가능성이 있지만, 이는 검증할 가설이다. 반대로 금융 문서가 충분하거나 일반 웹과 금융 문장의 차이가 크면 직접 경로의 비용 대비 효과가 더 좋을 수 있다.

## 2. 현재 증거와 해석 범위

### FineWeb-Edu의 전이 효과

`earning_call_2013_8k/{presentation,qna}_eval.csv` 기준:

| 외부 평가 | 원본 8K CE | FineWeb 10M | FineWeb 100M | 100M까지 개선 중 첫 10M 비중 |
|---|---:|---:|---:|---:|
| Presentation | 1.8130 | 1.6996 | 1.6767 | 약 83.2% |
| Q&A | 1.9468 | 1.8289 | 1.7991 | 약 79.8% |

이는 FineWeb 추가 MLM이 earnings-call MLM에도 전이됐다는 증거다. 직접 DAPT보다 낫다는 증거는 아니다. 연구일지의 2009~2013년 설명과 달리 현재 로컬 train/val Parquet의 date 범위는 2013년이다. 이전 외부 평가에 다른 데이터 버전을 썼는지는 현재 파일만으로 확정할 수 없다.

관련 문맥 효과는 다음으로 구분한다.

- paired_context_gain = CE(short tail) − CE(original prefix + same tail).
- relevant_prefix_gain = CE(random prefix + same tail) − CE(original prefix + same tail).
- random_prefix_gain이라는 기존 CSV 필드는 CE(short tail) − CE(random prefix + same tail)이므로 relevant_prefix_gain과 다르다.

현재 CSV로 계산한 relevant_prefix_gain:

| 평가 | 원본 | FineWeb 100M |
|---|---:|---:|
| Presentation | 0.2466 | 0.2455 |
| Q&A | 0.1889 | 0.1694 |

따라서 일반적인 8K MLM 개선은 확인되지만 관련 있는 먼 문맥의 추가 이점이 증가했다고 말하기 어렵다. 연구일지 Presentation 표의 마지막 열에는 paired gain 대신 relevant gain이 들어가 있고, 5.3의 요약 숫자도 위 CSV와 다르다. 최종 논문용 표는 동일한 평가 manifest에서 다시 생성해야 한다.

### FineWeb 100M 이후 DAPT

각 실행의 `history.csv` 기준:

| 학습 섹션 | 시작 8K CE | DAPT 50M 후 | paired gain 변화 | 데이터 반복 |
|---|---:|---:|---:|---:|
| Presentation | 1.6827 | 1.3715 | 0.2491 → 0.1310 | 약 7.36회 |
| Q&A | 1.8121 | 1.4366 | 0.1661 → 0.0696 | 약 2.88회 |

두 단계 경로 내부에서 DAPT가 MLM에 유효하다는 결과다. 직접 경로, EAR 결과, 충분한 seed 반복이 없으므로 경로 간 우열은 미확정이다. 학습 스크립트 기본 검증 표본은 최대 24개 장문이고, 앞선 외부 평가와 샘플·mask가 다르므로 표 사이 CE를 이어 붙여 직접 비교하지 않는다. paired gain 감소만으로 문맥 능력의 망각을 판정하지 않는다.

## 3. 우선 실험

2013년을 개발용으로 쓰고 가능하면 후속 1~2개 연도에서 재검증한 뒤 연도 전체로 확대한다. 아래 토큰 수는 시작용 설계값이며 최적값으로 검증된 수치가 아니다.

| 실험 | FineWeb 학습 | Earnings Call 학습 | 목적 |
|---|---|---|---|
| Base | 없음 | 없음 | 원본 임베딩 기준선 |
| A | 없음 | 8K, 20M | 동일 도메인 노출량 비교의 직접 기준선 |
| B | 8K, 10M | 8K, 20M | 제안한 두 단계 |
| C | 없음 | 8K, 30M | B와 총 처리 토큰을 맞춘 직접 기준선 |
| D | 1K, 10M | 8K, 20M | 일반 텍스트 추가 학습과 8K 학습 효과 분리 |

A와 C는 직접 경로 한 번을 30M까지 학습하면서 20M 지점도 저장해 얻는다. 도메인 단계에서는 0/10/20/30M 등을 저장한다. B가 유망하면 FineWeb 25M 또는 50M을 추가 비교한다. 기존 FineWeb 100M 모델은 참고 실행으로 활용하되 새 실행과 평가 프로토콜을 맞춘다.

B 대 A는 금융 데이터 노출량을 고정한 실용적 추가 효과를 보고, B 대 C는 총 처리 토큰을 맞춘 비용 대비 비교를 한다. 총 토큰이 같아도 길이·패딩·검증 비용에 따라 FLOPs와 시간이 다르므로 GPU 시간, steps, non-padding tokens, 실제 masked targets, 고유 문서/토큰, epochs도 기록한다. D는 같은 FineWeb 문서 집합·텍스트 노출량을 최대한 맞추고 1K/8K로 분할한다. optimizer steps 및 유효 배치 토큰도 통제·기록한다.

먼저 Q&A에서 비교해 선택지를 줄이고 Presentation에서 확인한다. 최종 후보는 가능하면 3개 seed로 검증한다. 섹션별 모델과 joint 모델 비교는 학습 경로 선택 이후의 별도 실험으로 둔다. 한 번에 pooling, LR, 학습 경로를 모두 바꾸지 않는다.

평가는 다음을 함께 본다.

1. 동일 held-out call, 동일 tail target·mask·random donor를 사용한 1K/8K CE와 두 context gain.
2. 8K 장문만의 진단 외에 짧은 문서를 포함한 전체 도메인 성능. 문서 길이별 성능도 보고한다.
3. 동일 pooling, Ridge 설정, expanding/rolling 시간 분할에서 EAR OOS R²와 예측 상관. 포트폴리오가 목표라면 비용 반영 H-L 결과도 비교한다.
4. 선택은 validation 기간에서 하고 이후 test는 고정한다. EAR 라벨의 수익률 관측창이 완전히 끝난 표본만 Ridge 학습에 사용하며 겹치는 라벨 기간은 purge/gap으로 처리한다.
5. MLM의 문서별 짝지은 차이와 불확실성을 보존한다. EAR의 통계 비교는 시간·기업 의존성을 고려한다.

EAR 개선이 연도·seed에 걸쳐 일관되고 비용을 정당화하면 B를 채택한다. 차이가 작거나 불안정하면 직접 경로를 채택한다. 초기 장문 전이 효율 때문에 B를 검증할 가치는 충분하지만 100M을 기본값으로 삼을 이유는 아직 없다.

## 4. PIT와 현재 코드의 선결 사항

1. annual base B_y는 독립적으로 시작한다. B_y → C_y → D_y와 B_(y+1) → C_(y+1) → D_(y+1)를 별도 실행한다. 전년도 적응 모델을 다음 연도의 공식 base 대신 이어 쓰는 방법은 다른 연구 설계다.
2. adaptation 문서의 실제 공개/이용 가능 시각이 cutoff 이하여야 한다. rdq만으로 transcript 공개 시각을 확정하지 않는다. 예측일에는 cutoff가 앞서고 준비가 끝난 모델을 매핑한다. 예: 2013-12-31 cutoff 모델은 2014년 예측용이며 2013년 과거 예측에 소급하지 않는다. 연말 학습·준비 지연 정책도 명시한다.
3. FineWeb은 가장 이른 crawl이 2013년이다. 이전 annual checkpoint에 2013년 자료를 재사용하지 않는다. 그 연도 이전에 존재했던 장문 출처를 확보하거나 해당 연도는 직접 DAPT를 사용한다.
4. FineWeb-Edu는 후대 Llama3 기반 분류기로 선별됐다. 본문 시점 제한과 선별기의 시점 제한은 다르다. 이것이 미래 본문·라벨의 직접 주입이라는 뜻은 아니지만, 엄격한 PIT 주장을 위해 선별 절차의 한계를 명시하고 후대 학습형 선별기를 쓰지 않는 자료로 민감도 검증한다.
5. Presentation/Q&A의 현재 train-val 내 call_id 중복은 각각 0이다. 하지만 Presentation train ∩ Q&A val은 19 calls, Q&A train ∩ Presentation val은 17 calls다. joint 모델이나 공유 단계를 쓸 때 공통 call-level split을 먼저 만들어 두 섹션에 적용한다. cutoff 경계의 같은 날짜도 한쪽에 배정한다.
6. 외부 평가 스크립트는 전체 presentation.parquet/qna.parquet을 읽는다. DAPT 후 같은 전체 파일로 평가하면 train 문서가 포함되므로 공정한 held-out 평가가 아니다. 학습 경로와 섹션 전체에서 공통으로 제외한 holdout을 사용한다.
7. 현재 `--model <FineWeb checkpoint>`는 새 optimizer로 단계 전환을 구현할 수 있다. `--resume`는 optimizer·RNG·순서·토큰 수를 복원하며 데이터 fingerprint 일치를 요구한다. 다른 데이터로의 단계 전환에는 resume를 쓰지 않는다. 새 인터페이스는 `--init-checkpoint`와 `--resume-run`으로 의도를 분리한다.
8. 현재 train은 min_len=4096으로 짧은 블록을 버리지만 수집기는 짧은 문서도 토큰에 포함한다. 실제 Presentation train 4365개 중 원문 4096토큰 미만이 3032개, Q&A는 4307개 중 1514개다. 장문 전용 단계와 전체 길이 DAPT의 정책을 나누고 학습에 실제 남은 토큰을 보고한다. 미리 다른 문서를 단순 연결해 장문 문맥인 것처럼 만들지 않는다.
9. Presentation 실행 폴더의 현재 cache/train.meta.json은 Q&A source를 가리킨다. 반면 tokens_050m/trainer_state.pt는 Presentation source와 1335개 block을 기록한다. 최종 체크포인트 기록은 Presentation 학습과 부합하나 캐시는 이후 변경된 것으로 보인다. 변경 경위는 확정할 수 없다. 현재 캐시와 과거 실행 결과를 동일한 provenance로 취급하지 않는다.

## 5. 제안 디렉토리

아래는 구현 전 제안이다. 기존 파일을 실제 이동한 상태가 아니다.

```text
chrono_bert_test/
├── pyproject.toml
├── configs/
│   ├── models/annual.yaml             # year → 공식 model id, revision, cutoff
│   ├── data/{fineweb,earnings}.yaml    # 출처, 시점, 분할, 길이 정책
│   ├── stages/{context,domain}.yaml   # 공유 MLM 엔진의 단계별 설정
│   └── experiments/
│       ├── direct.yaml
│       ├── context_then_domain.yaml
│       └── short_context_control.yaml
├── src/chrono_pipeline/
│   ├── cli.py                        # prepare/train/evaluate/embed/backtest
│   ├── data/
│   │   ├── fineweb.py
│   │   ├── earnings.py
│   │   ├── temporal.py                # availability cutoff 검증
│   │   ├── splits.py                  # 공통 call-level split
│   │   └── blocks.py                  # 길이 정책, tokenization, memmap
│   ├── training/
│   │   ├── mlm.py                     # 단일 학습 루프
│   │   └── checkpoint.py              # 초기화와 정확한 재개 분리
│   ├── evaluation/
│   │   ├── fixtures.py                # 문서·mask·donor 고정
│   │   ├── mlm.py
│   │   └── context.py                 # paired/relevant prefix 지표
│   ├── embeddings/{extract,pooling}.py
│   ├── downstream/{ridge,walk_forward}.py
│   └── pipeline/{runner,manifest}.py   # 단계 의존성, lineage, artifact 등록
├── scripts/run_years.py               # 모델 연도별 독립 실행
├── data/
│   ├── raw/                          # 원본 DB/스냅샷; 대용량 저장소 참조 가능
│   ├── manifests/<dataset_id>/        # 문서 id, 시점, hash, split
│   └── processed/<dataset_id>/        # 고정 버전 parquet
├── cache/<cache_key>/                 # 내용·tokenizer·블록 정책 hash
├── evaluations/<eval_id>/             # eval manifest, masks, donor ids
├── runs/base_2013/<experiment>/<section>/seed_42/<run_id>/
│   ├── resolved_config.yaml
│   ├── manifest.json
│   ├── context/                      # 직접 DAPT에서는 생략
│   │   ├── checkpoints/tokens_010m/
│   │   └── metrics.jsonl
│   ├── domain/
│   │   ├── checkpoints/tokens_020m/
│   │   └── metrics.jsonl
│   ├── evaluation/
│   ├── embeddings/
│   └── downstream/
├── reports/                          # 실행 id를 인용한 비교표·그림
├── docs/                             # 프로토콜 및 설계 근거
└── tests/                            # PIT, split, mask pairing, resume 검증
```

핵심은 도메인마다 trainer를 복제하지 않고 설정으로 같은 MLM 루프를 구동하는 것이다. runner만 direct와 context→domain의 차이를 안다. 데이터 cache와 평가 fixture는 모델 가중치 경로와 분리해 비교군이 공유한다.

manifest 필수 필드: run_id, base model/revision, base cutoff, experiment, section, seed, init checkpoint 및 parent run id, data/split/tokenizer/eval id, source content hash, 공개시각 정책, resolved config 및 코드 버전/hash, 라이브러리 버전, stage-local processed tokens와 전체 lineage 토큰 수, 고유 tokens/epochs, GPU 시간, 최종 산출물 경로. 현재 디렉토리는 Git 저장소가 아니므로 도입 전까지는 코드 파일 hash로 버전을 남길 수 있다.

cache key는 source 내용 또는 안정적인 dataset manifest + tokenizer vocab/설정 hash + max/min length + 블록 정책 + 전처리 버전으로 만든다. 동일 tokenizer를 사용하는 가중치별로 캐시를 복제하지 않는다. 실행 중에는 cache를 불변으로 참조하고 완료된 run 디렉토리를 재사용하지 않는다.

평가 fixture는 long 학습의 min_len 필터와 별개로 raw held-out manifest에서 만든다. 그래야 전체 문서 분포 평가와 장문 진단을 동시에 재현할 수 있다. RNG seed만 기록하지 않고 실제 선택 문서, target mask, donor mapping도 저장한다.

## 6. 현재 파일을 옮기는 순서

1. 기존 데이터·체크포인트는 제자리에 두고 manifest에 경로를 등록한다. 우선 연도, 단계, section, source의 일치 여부를 감사한다.
2. `collect_fineweb_2013_test.py`와 `collect_earning_call_test.py`를 data 모듈과 prepare CLI로 정리한다. split을 notebook 밖의 단일 call-level 규칙으로 고정한다.
3. `compare_chronobert_2013.py`와 `compare_chronobert_2013_resume_with_earning_call.py`의 중복을 단일 training/mlm.py로 통합한다. 입력 파일 목록, max/min length, 초기 모델, 토큰 예산을 설정으로 받는다.
4. `compare_chrono_bert_2013_by_checkpoint.py`의 fixed validation과 random-prefix를 독립 evaluation 모듈로 옮긴다. `*/*.pt` glob과 폴더 이름으로 checkpoint를 찾는 대신 manifest에 등록된 모델 목록을 읽는다.
5. 직접 DAPT 30M과 FineWeb 10M→DAPT 20M을 동일 fixture로 비교하고 이후 embeddings/EAR 파이프라인을 연결한다.
6. cutoff 위반, 공통 call split 위반, paired target 불일치, 단계 전환/재개 혼동을 잡는 핵심 검증을 둔다. 수치 비교는 실제 소규모 smoke run 후 본 실험으로 확장한다.

## 참고한 1차 자료

- 첨부 `2026-10-29_ChronoBERT_연구일지.pdf`, 1~9쪽.
- ChronoBERT 공식 모델 카드: https://huggingface.co/manelalab/chrono-bert-v1-20131231
- Gururangan et al. (2020), Don't Stop Pretraining: https://aclanthology.org/2020.acl-main.740/ — DAPT/TAPT의 가능성을 뒷받침하지만 FineWeb→earnings call의 우월성을 직접 검증하지는 않는다.
- FineWeb-Edu 공식 데이터 카드: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu — crawl 범위와 후대 분류기에 의한 선별 과정.
