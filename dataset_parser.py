import os
import json
import pandas as pd
from typing import List, Dict, Any

# NExT-QA val.csv 컬럼 구조:
#   video, qid, type, question, a0, a1, a2, a3, a4, answer (0~4 인덱스)

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

def load_nextqa(csv_path: str, video_dir: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        video_id = str(row["video"])
        samples.append({
            "video_id": video_id,
            "video_path": os.path.join(video_dir, f"{video_id}.mp4"),
            "question": row["question"],
            "options": {
                "A": row["a0"],
                "B": row["a1"],
                "C": row["a2"],
                "D": row["a3"],
                "E": row["a4"],
            },
            "answer": _IDX_TO_LETTER[int(row["answer"])],
        })
    return samples

def save_json(samples: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    samples = load_nextqa("data/nextqa/val.csv", "videos/nextqa/")
    save_json(samples, "data/nextqa_val.json")
    print(f"총 {len(samples)}개 샘플 저장 완료")
    print(samples[0])
