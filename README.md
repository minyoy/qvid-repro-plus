# Q-ViD Repro Plus

Q-ViD 스타일의 비디오 질의응답 실험을 재현하고, Qwen3-VL 같은 비전-언어 모델 백본으로 같은 평가 파이프라인을 실행하기 위한 프로젝트입니다.

## 개요

기본 파이프라인은 NExT-QA 형식의 CSV와 비디오를 입력으로 받아 다음 순서로 평가합니다.

1. 비디오에서 균등 간격 프레임 추출
2. 질문을 반영한 프레임별 캡션 생성
3. 캡션과 선택지를 기반으로 정답 선택
4. accuracy, 처리 시간, 메모리 사용량을 JSON으로 저장

## 설치

```bash
pip install -r requirements.txt
```

모델 가중치는 실행 시 Hugging Face에서 내려받습니다. 큰 모델을 사용하는 경우 GPU 또는 충분한 메모리가 필요합니다.

## 데이터 구조

```text
data/nextqa/val.csv        # NExT-QA 형식 검증 CSV
videos/nextqa/*.mp4        # CSV의 video ID와 같은 이름의 비디오 파일
outputs/                   # 평가 결과 JSON과 로그
```

`val.csv`는 `video`, `question`, `a0`~`a4`, `answer` 컬럼을 사용합니다. `answer`는 0~4 인덱스이며 코드에서 A~E로 변환됩니다.

## 실행 예시

기본 Q-ViD 파이프라인:

```bash
python evaluate.py \
  --csv_path data/nextqa/val.csv \
  --video_dir videos/nextqa \
  --output outputs/predictions.json \
  --n_frames 8 \
  --limit 10
```

Qwen3-VL 백본 파이프라인:

```bash
python evaluate_qwen3vl.py \
  --csv_path data/nextqa/val.csv \
  --video_dir videos/nextqa \
  --output outputs/qwen3vl_predictions.json \
  --n_frames 8 \
  --limit 10 \
  --model Qwen/Qwen3-VL-4B-Instruct
```

## 추가 실험: 프레임 수별 비교

프레임 샘플링 개수가 성능과 처리 시간에 미치는 영향을 보기 위해 `n_frames` 값을 바꿔 비교할 수 있습니다. 현재 저장된 비교 결과는 `outputs/frames_8.json`, `outputs/frames_32.json`, `outputs/frames_64.json`에 있습니다.

| 프레임 수 | 평가 샘플 수 | 정답 수 | Accuracy | 평균 처리 시간/샘플 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 20 | 13 | 65.0% | 3.37s |
| 32 | 20 | 13 | 65.0% | 12.98s |
| 64 | 20 | 13 | 65.0% | 25.42s |

같은 형식의 실험은 다음처럼 실행합니다.

```bash
python evaluate.py --n_frames 8 --limit 20 --output outputs/frames_8.json
python evaluate.py --n_frames 32 --limit 20 --output outputs/frames_32.json
python evaluate.py --n_frames 64 --limit 20 --output outputs/frames_64.json
```

이 실험에서는 프레임 수를 늘려도 20개 샘플 기준 정확도는 동일했고, 평균 처리 시간은 프레임 수에 비례해 증가했습니다.

## 추가 실험: Qwen3-VL 비교

Qwen3-VL 백본도 같은 NExT-QA 20개 샘플, 8프레임 설정으로 비교했습니다. 저장된 결과는 `outputs/qwen3vl_4b_predictions.json`, `outputs/qwen3vl_4b_predictions_adapted.json`, `outputs/qwen3vl_8b_predictions.json`, `outputs/qwen3vl_8b_predictions_adapted.json`에 있습니다.

| 모델 | 프롬프트 | 프레임 수 | 평가 샘플 수 | 정답 수 | Accuracy | 평균 처리 시간/샘플 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL 4B | 기본 | 8 | 20 | 10 | 50.0% | 5.34s |
| Qwen3-VL 4B | adapted | 8 | 20 | 15 | 75.0% | 5.76s |
| Qwen3-VL 8B | 기본 | 8 | 20 | 6 | 30.0% | 7.90s |
| Qwen3-VL 8B | adapted | 8 | 20 | 15 | 75.0% | 8.56s |

현재 `evaluate_qwen3vl.py`는 프레임 캡션 단계에서 질문을 직접 답하지 않고 시각적 근거만 설명하도록 하고, QA 단계에서는 반드시 A~E 중 하나만 고르도록 하는 adapted 프롬프트를 사용합니다.

```bash
python evaluate_qwen3vl.py \
  --output outputs/qwen3vl_4b_predictions_adapted.json \
  --n_frames 8 \
  --limit 20 \
  --model Qwen/Qwen3-VL-4B-Instruct

python evaluate_qwen3vl.py \
  --output outputs/qwen3vl_8b_predictions_adapted.json \
  --n_frames 8 \
  --limit 20 \
  --model Qwen/Qwen3-VL-8B-Instruct
```

이 비교에서는 adapted 프롬프트가 4B와 8B 모두에서 75.0%로 가장 높은 정확도를 보였습니다.

## 주요 파일

```text
qvid-repro-plus/
├── evaluate.py                  # 기본 Q-ViD 평가 엔트리포인트
├── evaluate_qwen3vl.py          # Qwen3-VL 평가 엔트리포인트
├── dataset_parser.py            # NExT-QA CSV를 평가 샘플로 변환
├── frame_extractor.py           # 비디오에서 균등 간격 프레임 추출
├── caption_generator.py         # InstructBLIP 기반 질문 조건부 캡션 생성
├── reasoning_module.py          # 캡션, 질문, 선택지 기반 정답 문자 추출
├── evaluation_metrics.py        # 프로세스/GPU/MPS 메모리 사용량 수집
├── requirements.txt             # 프로젝트 의존성
├── data/
│   └── nextqa/
│       └── val.csv              # NExT-QA 검증 CSV
├── videos/
│   └── nextqa/
│       └── *.mp4                # 평가 대상 비디오 파일
├── outputs/
│   ├── frames_*.json            # 프레임 수별 비교 결과
│   ├── qwen3vl_*_predictions*.json
│   │                           # Qwen3-VL 비교 결과
│   └── logs/                    # 실행 로그
└── models/
    ├── qvid_pipeline.py         # 여러 VL 백본이 공유하는 평가 루프
    └── qwen3vl/
        ├── adapter.py           # Qwen3-VL 모델 로딩 및 생성 어댑터
        └── pipeline.py          # Qwen3-VL 전용 프롬프트와 CLI 옵션
```

## 출력 형식

평가 결과 JSON은 다음 구조를 가집니다.

```text
summary.total
summary.correct
summary.accuracy
summary.accuracy_percent
summary.n_frames
summary.avg_elapsed_sec
summary.final_memory
results[]
```

각 `results` 항목에는 `video_id`, `question`, `options`, `gold`, `pred`, `raw_output`, `correct`, `captions`, `elapsed_sec`, `memory` 등이 저장됩니다.
