import os
import cv2
import numpy as np
from PIL import Image
from typing import List

def extract_frames(video_path: str, n_frames: int = 64) -> List[Image.Image]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"비디오 파일 없음: {video_path}")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > total:
        cap.release()
        raise ValueError(f"요청 프레임({n_frames}) > 실제 프레임({total})")

    indices = set(np.linspace(0, total - 1, n_frames, dtype=int).tolist())
    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    return frames

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
    frames = extract_frames(path, n_frames=64)
    print(f"추출된 프레임 수: {len(frames)}")
    print(f"프레임 크기: {frames[0].size}")
    frames[0].save("check_frame.jpg")
    print("check_frame.jpg 저장 완료")
