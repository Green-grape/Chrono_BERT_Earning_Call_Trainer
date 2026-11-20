# 연도별 ChronoBERT Earnings Call DAPT

2013-12-31부터 2024-12-31까지 12개 원본 ChronoBERT에서 Presentation/Q&A를 **독립적으로 10M 처리 토큰씩** 학습한다. 총 24개 학습 작업이다. FineWeb 단계와 EAR/Ridge 학습은 포함하지 않는다.

```text
ChronoBERT-20131231 ─┬─ Presentation MLM 10M → Presentation 모델
                    └─ Q&A MLM 10M          → Q&A 모델
...
ChronoBERT-20241231 ─┬─ Presentation MLM 10M → Presentation 모델
                    └─ Q&A MLM 10M          → Q&A 모델
```

연도 사이에도 가중치를 이어 학습하지 않는다. 매번 해당 연도의 공식 원본에서 시작한다.

## 실행 환경

학습 실행은 Linux/WSL + CUDA 환경을 기준으로 한다. 작업별 프로세스 잠금에 `fcntl`을 사용한다. 모델 산출물은 다른 환경으로 복사할 수 있다.

```bash
conda activate pit_llm
cd /home/yhkim137/chrono_bert_test
python -m pip install -r main/requirements.txt
```

현재 `pit_llm`에 Transformers 5.0.0이 설치돼 있다. 의존성을 이미 만족하면 재설치할 필요가 없다. CPU는 작은 테스트 모델용이며 실제 8K 학습에는 CUDA를 사용한다.

## 실행 순서

```bash
# 파일을 생성하지 않고 24개 작업과 설정 확인
python main/run.py plan

# DB 연도별 문서 수와 원본 모델의 실제 로컬 가중치 확인; 다운로드하지 않음
python main/run.py check

# 실제 GPU에서 2013년 두 모델을 각각 2 optimizer steps만 실행
python main/run.py run --years 2013 --smoke-steps 2

# 해당 smoke checkpoint에서 이어서 10M까지 학습
python main/run.py run --years 2013 --resume

# 2013~2024년 전체 실행. 완료된 작업은 재학습하지 않고 export만 확인
python main/run.py run --resume
```

완료했거나 중단한 학습의 누적 토큰 목표를 늘리려면 `--target-tokens`를 지정한다.

```bash
# 기존 2013 Q&A checkpoint에서 이어서 누적 20M까지 학습
python main/run.py train --years 2013 --sections qna \
  --resume --target-tokens 20000000
```

20M은 추가량이 아닌 **누적 목표**다. 이미 약 10M을 처리했다면 약 10M을 더 학습한다.
가중치, optimizer, RNG, 데이터 순서와 누적 토큰을 복원하고 기존 run 안에 checkpoint를 추가한다.
`train`과 `run` 모두 지원하며, 기존 설정 파일의 `target_tokens`와 `save_tokens`는 바꾸지 않는다.
확장 도중 중단돼도 다음 `--resume`은 늘린 목표를 유지한다.
목표는 설정값 및 이전 실행 목표보다 작을 수 없다. 호환되는 checkpoint가 없거나 여러 개면 오류로 중단한다.
기존 데이터, 학습 설정, 라이브러리 버전 검증은 유지하며, 이번 변경 직전 코드의 checkpoint도 지원한다.

기본값은 원본 모델도 **로컬 캐시에서만** 읽는다. 가중치가 없는 연도는 다음처럼 다운로드를 허용한다.

```bash
python main/run.py run --resume --allow-download
```

기존 Hugging Face 캐시 폴더에 `refs/`나 `.no_exist/`만 있는 경우 실제 모델이 준비된 것이 아니다. `check`가 `model.safetensors` 또는 모든 shard와 tokenizer를 검사한다.

작업을 선택하거나 데이터 준비만 실행할 수도 있다.

```bash
python main/run.py prepare --years 2013
python main/run.py train --years 2013 --sections qna --resume
python main/run.py run --years 2014 2015 --sections presentation --resume
```

`train`과 `run`은 모두 검증된 준비 데이터를 재사용하고, 없으면 준비한다. 예외가 발생하면 이후 작업으로 넘어가지 않고 중단한다. 문제를 해결한 뒤 `--resume`으로 다시 실행한다. `--smoke-steps`는 이번 호출에서 실행할 최대 step 수이며, 목표 토큰 수와 실험 설정을 바꾸지 않는다. 목표를 달성하지 못한 smoke 모델은 최종 exports에 등록하지 않는다.

## 기본 설정

`configs/default.json`의 상대 경로는 **설정 파일의 위치**를 기준으로 해석한다.

| 설정 | 기본값 |
|---|---|
| DB | 프로젝트 루트 `call_sections_rebuilt_new.db` |
| 산출물 | `main/artifacts/` |
| 데이터 범위 | checkpoint와 같은 연도의 문서 |
| 섹션 | Presentation, Q&A 별도 모델 |
| 학습 예산 | 각 섹션 10,000,000 non-padding 입력 토큰 |
| 저장 지점 | 약 5M, 10M |
| 최대 길이 | 8,192, CLS/SEP 포함 |
| 최소 본문 블록 | 64 tokens |
| MLM | 30% 동적 masking |
| LR | 3e-5, 최초 500K 입력 토큰 동안 선형 warmup 후 고정 |
| Optimizer | AdamW, weight decay 0.01; bias/norm은 decay 제외 |
| 배치 | 문서 블록 1개 × accumulation 2 |
| 정밀도 | BF16 autocast, FP32 모델 가중치 |
| 메모리 | gradient checkpointing, ModernBERT sparse MLM |
| 검증 | 최근 약 10% calls, 고정 mask, 최대 64개 문서 |

`lookback_years: 1`은 당해 연도, `3`은 최근 3년, `0`은 DB의 모든 과거 자료를 cutoff까지 사용한다. 항상 cutoff는 해당 연도 12월 31일이다.

변경할 때는 설정 파일을 `configs/custom.json` 등으로 복사한 뒤 사용한다.

```bash
python main/run.py --config main/configs/custom.json plan
python main/run.py --config main/configs/custom.json run --years 2013
```

토큰 예산은 반복을 포함하고 CLS/SEP를 포함한다. 한 optimizer step을 마친 뒤 저장하므로 목표를 최대 한 step 정도 초과한다. 실제 처리 토큰·masked targets·고유 위치 토큰·반복 횟수를 기록한다. 긴 문서는 원문별로 분할하며 짧은 문서도 유지한다. 최소 길이보다 짧은 마지막 조각은 버린다. 다른 call을 연결해 장문으로 만들지 않는다.

## 데이터 분할과 시점

- Presentation/Q&A 공통 `call_id` 목록으로 시간 분할한다. 같은 날짜의 calls는 모두 같은 split에 둔다. 섹션별로 분할하지 않는다.
- exact duplicate text는 두 섹션 전체에서 중복 제거한다. 검증 자료를 먼저 처리해, 검증 텍스트와 같은 과거 학습 텍스트는 제외한다.
- 검증용 masking은 학습 RNG와 분리하고 실제 평가 tensor를 저장한다. validation CE는 masked token 수로 가중 평균한다. 학습 gradient accumulation도 실제 masked target 수로 가중한다.
- 입력 DB는 read-only, 일관된 SQLite transaction에서 읽는다. 원본 DB는 변경하지 않는다.
- 현재 DB에 transcript 공개시각이 없어 **`rdq`를 이용 가능일의 대리변수로 사용**한다. manifest에 이 한계를 기록한다. 실제 공개일 컬럼을 마련하면 `availability_column`을 변경한다. 미래 수정본의 수집 이력까지 보장하는 기능은 아니다.
- 2013-12-31 모델은 2014년 이후 예측용이다. 2013년 과거 예측에 소급 적용하지 않는다. 실제 운영 재현에서는 모델 준비 지연도 외부 프로젝트에서 고려한다.
- 학습에 사용한 문서를 EAR의 미관측 평가 데이터로 취급하지 않는다. 외부 EAR 프로젝트의 시간 분할은 별도로 관리한다.

## 저장 구조

```text
main/
├── run.py
├── load_model.py
├── configs/default.json
├── chrono_dapt/
│   ├── cli.py
│   ├── check.py
│   ├── config.py
│   ├── data.py
│   ├── training.py
│   └── export.py
├── tests/test_pipeline.py
└── artifacts/
    ├── data/<year>/<signature>/       # 공통 split + 섹션별 JSONL
    ├── cache/<signature>/            # memory-mapped token blocks
    ├── runs/<year>/<section>/seed_42/<run_id>/
    │   ├── identity.json
    │   ├── resolved_config.json
    │   ├── baseline.json
    │   ├── evaluation_fixture.json
    │   ├── latest.json
    │   └── checkpoints/
    │       └── step_..._tokens_.../
    │           ├── config.json
    │           ├── model.safetensors
    │           ├── tokenizer.json
    │           ├── training_manifest.json
    │           └── trainer_state.pt
    └── exports/
        ├── manifests/<year>_<section>.json
        ├── presentation/
        │   ├── models--manelalab--chrono-bert-v1-20131231/
        │   │   ├── .no_exist/<local_revision>/adapter_config.json
        │   │   ├── blobs/<file_hash>
        │   │   ├── refs/main
        │   │   └── snapshots/<local_revision>/
        │   │       ├── config.json
        │   │       ├── model.safetensors
        │   │       ├── tokenizer.json
        │   │       ├── tokenizer_config.json
        │   │       ├── special_tokens_map.json
        │   │       └── training_manifest.json
        │   └── ...20241231/
        └── qna/
            └── models--manelalab--chrono-bert-v1-.../
```

최종 `exports/`에는 optimizer를 포함하지 않는다. 같은 원본 repo 이름을 section별 cache root 안에서 사용하므로 기존 원본 캐시와 충돌하지 않는다. snapshot revision은 **로컬 학습 결과의 content hash**이며 원본 Hub commit과 다르다. 원본 commit은 training manifest에 별도로 기록한다.

snapshot의 파일은 `../../blobs/`를 가리키는 상대 symlink다. **다른 프로젝트로 옮길 때 `exports/` 전체를 복사**한다. symlink를 지원하지 않는 복사 도구라면 링크를 실제 파일로 복사하는 옵션을 사용한다. `snapshots/`만 따로 복사하면 링크가 끊어진다.

5M checkpoint는 runs 안에 남는다. 필요하면 다음처럼 따로 export할 수 있다. 이 명령은 해당 year/section의 `refs/main`을 지정한 checkpoint로 전환한다.

```bash
python main/run.py export --checkpoint /absolute/path/to/checkpoint --year 2013 --section qna
```

## 다른 프로젝트에서 로드

가장 확실한 방법은 검증된 **snapshot 디렉토리를 직접** 읽는 것이다. `load_model.py`와 `chrono_dapt/`도 함께 복사하면 다음 함수를 사용할 수 있다.

```python
from load_model import load_adapted_model

tokenizer, encoder = load_adapted_model(
    "/absolute/path/to/exports", 2013, "qna", device="cuda"
)
inputs = tokenizer("The company reported earnings.", return_tensors="pt").to("cuda")
import torch
with torch.inference_mode():
    hidden = encoder(**inputs).last_hidden_state
```

기존 코드에서 Hugging Face cache 방식으로도 읽을 수 있다.

```python
from transformers import AutoModel, AutoTokenizer

repo = "manelalab/chrono-bert-v1-20131231"
cache = "/absolute/path/to/exports/qna"
tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache, local_files_only=True)
encoder = AutoModel.from_pretrained(
    repo, cache_dir=cache, local_files_only=True,
    adapter_kwargs={"cache_dir": cache, "local_files_only": True},
    attn_implementation="sdpa",
).eval()
```

`adapter_kwargs`는 PEFT가 설치된 Transformers 5.0 환경에서 adapter 탐색에도 같은 cache/offline 설정을 전달하기 위한 것이다. 로컬 snapshot 경로를 직접 사용하는 helper는 이 원격 탐색 경로를 피한다.

모델 경로만 확인하려면:

```bash
python main/run.py resolve --year 2013 --section qna
```

## 재개와 검증

재개 checkpoint는 optimizer, Python/Torch RNG, 문서 순열, cursor, shuffle generator, 처리 토큰 및 step 수를 저장한다. 데이터 내용, tokenizer, 원본 revision, 주요 학습 설정, 코드 hash, 라이브러리 버전을 실험 식별자에 포함한다. 설정/코드가 달라지면 별도 실행 디렉토리를 사용한다. 다른 실험의 optimizer를 이어받지 않는다. GPU의 비결정적 연산까지 동일한 수치를 보장하는 것은 아니며 CPU 테스트에서는 연속 실행과 재개 실행의 최종 가중치가 정확히 일치하는지 검사한다.

```bash
PYTHONPATH=main python -m unittest discover -s main/tests -v
```

Transformers가 없는 환경에서는 데이터/파일 테스트만 실행하고 학습 통합 테스트는 건너뛴다. 통합 테스트는 작은 BERT로 재개·이동·오프라인 로드를 검증하고, 작은 ModernBERT로 sparse MLM + gradient checkpointing 학습 및 encoder 로드를 검증한다. 실제 8K/GPU 학습은 위 smoke 명령으로 확인한다.

형식 참고: [Hugging Face cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache), [save_pretrained/from_pretrained](https://huggingface.co/docs/transformers/main_classes/model).
