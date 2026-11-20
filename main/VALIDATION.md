# 구현 검증 기록

2026-10-30, 이 작업 세션에서 확인한 결과다. 실제 24개 모델 학습을 실행한 기록은 아니다.

- `python main/run.py plan`: 2013~2024 × Presentation/Q&A, 총 24개 독립 작업 확인.
- `python -m compileall -q main`: 통과.
- `PYTHONPATH=main /home/yhkim137/miniconda3/envs/pit_llm/bin/python -m unittest discover -s main/tests -v`: 8개 테스트 통과.
- 테스트 환경: Python 3.12, Transformers 5.0.0, PyTorch 2.14.1+cu132. 이 세션의 `torch.cuda.is_available()`은 False.
- 작은 BERT: MLM 학습, 중단 후 재개, 연속 실행과 최종 가중치의 정확한 일치, export 폴더 이동 후 tokenizer/encoder 로드, HF repo id 기반 cache 로드 검증. Cache 로드 테스트에서는 네트워크 socket 연결을 금지했다.
- 작은 ModernBERT: sparse MLM + gradient checkpointing 학습, safetensors 저장, AutoModel encoder 로드 검증.
- 실제 DB에서 2013년 최초 80 calls를 임시 폴더로 준비하고 실제 ChronoBERT-2013 tokenizer로 전처리 확인. 빈 텍스트/중복 처리 후 섹션별 train 63 / val 15 문서. Presentation train 226,882 tokens, Q&A train 365,838 tokens. 임시 데이터는 검증 종료 후 삭제.
- 로컬 원본 모델 점검: 2013·2014는 config/tokenizer/가중치와 8192 position capacity 확인. 2015~2024는 해당 cache 위치에 refs/.no_exist만 있고 실제 snapshot 가중치가 없어 다운로드 또는 별도 원본 cache 경로가 필요함.
- 실제 DB에는 요청한 모든 연도에 calls가 존재함: 2013년 5,471개부터 2024년 10,969개. 기본 window는 각 연도만 사용하므로 2025~2026 자료는 학습에 포함되지 않음.

CUDA의 실제 8192-token forward/backward, GPU 메모리 사용량, 10M-token 실행 및 EAR 성능은 이 세션에서 검증하지 않았다. GPU가 연결된 실행 환경에서 README의 2-step smoke 실행 후 본 학습을 시작한다.
