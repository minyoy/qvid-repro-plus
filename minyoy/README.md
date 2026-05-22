# Q-ViD Backbone Replacement Experiments

## 담당 파트

기존 InstructBLIP 기반 Q-ViD 파이프라인을 기준으로 NExT-QA validation 평가를
유지하면서, backbone만 교체한 추가 실험 구현을 작성했다.

| Experiment | Caption stage | QA reasoning stage |
| --- | --- | --- |
| InstructBLIP | `Salesforce/instructblip-flan-t5-xl` | `google/flan-t5-xl` |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | `Qwen/Qwen3-VL-4B-Instruct` |
| InternVL3.5 | `OpenGVLab/InternVL3_5-4B` | `OpenGVLab/InternVL3_5-4B` |

Qwen3-VL과 InternVL3.5는 caption 생성과 최종 QA reasoning을 모두 수행한다.

## Pipeline

```text
video
  -> uniformly sampled frames
  -> frame별 question-guided caption 생성
  -> captions + question + options
  -> text-only QA reasoning
  -> answer letter A/B/C/D/E 저장
```

QA reasoning 단계에는 이미지를 다시 넣지 않고, frame caption text만 사용한다.

## 유지한 조건

backbone 비교를 위해 기존 InstructBLIP 구현의 다음 요소를 유지했다.

- frame sampling과 기본 `n_frames=8`
- caption prompt와 QA prompt
- option 구성과 answer letter parsing
- prediction JSON schema와 accuracy 계산

caption prompt:

```text
Provide a detailed description of the image related to the {question}
```

generation 설정:

| Stage | Settings |
| --- | --- |
| Caption | `max_new_tokens=30`, `num_beams=5`, `repetition_penalty=1.5` |
| QA answer | `max_new_tokens=8`, `do_sample=False` |

## 구현 파일

| File | Role |
| --- | --- |
| `evaluate.py` | InstructBLIP baseline 실행 |
| `evaluate_qwen3vl.py` | Qwen3-VL 실행 |
| `evaluate_internvl35.py` | InternVL3.5 실행 |
| `models/qvid_pipeline.py` | 새 backbone 공통 evaluation runner |
| `models/qwen3vl/` | Qwen3-VL adapter |
| `models/internvl35/` | InternVL3.5 adapter와 image preprocessing |

공통 runner는 기존 `dataset_parser.py`, `frame_extractor.py`,
`reasoning_module.py`의 loader, frame sampling, QA prompt, answer parser를
재사용한다.

## Data

```text
qvid-repro-plus/
  data/
    nextqa/
      val.csv
  videos/
    nextqa/
      <video_id>.mp4
```

`val.csv`의 `video` 값과 `.mp4` 파일명이 같아야 한다.

별도 frame embedding이나 precomputed video feature는 사용하지 않는다. 실행 시
비디오에서 frame을 추출하고 각 backbone이 frame을 직접 처리한다.

## Install

```bash
pip install -r requirements.txt
pip install -r models/vl_backbones_requirements.txt
```

## Run

InstructBLIP:

```bash
python evaluate.py \
  --csv_path data/nextqa/val.csv \
  --video_dir videos/nextqa/ \
  --output outputs/instructblip_flan_t5_xl_predictions.json \
  --n_frames 8
```

Qwen3-VL:

```bash
python evaluate_qwen3vl.py \
  --csv_path data/nextqa/val.csv \
  --video_dir videos/nextqa/ \
  --output outputs/qwen3vl_4b_predictions.json \
  --n_frames 8
```

InternVL3.5:

```bash
python evaluate_internvl35.py \
  --csv_path data/nextqa/val.csv \
  --video_dir videos/nextqa/ \
  --output outputs/internvl35_4b_predictions.json \
  --n_frames 8
```

빠른 확인은 각 명령어에 `--limit 1` 또는 `--limit 5`를 추가한다.

## Output

prediction JSON은 `summary`와 `results`를 저장한다.

각 result에는 다음 값이 포함된다.

```text
video_id, question, options, gold, pred, raw_output,
correct, n_frames, elapsed_sec, captions
```

`pred`는 `A`부터 `E` 중 하나로 parsing된다. parsing에 실패하면 raw output은
남기고 `pred`는 `null`이 된다.
