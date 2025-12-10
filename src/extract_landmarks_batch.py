import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from config_local import RAW_DIR, SAVE_DIR

os.makedirs(SAVE_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

#MediaPipe lip landmark indices
LIPS = list(range(61, 91)) + list(range(146, 166))

def extract_mouth_roi(frame, landmarks, size=88):
    h, w, _ = frame.shape
    lip_points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    x1, y1 = np.min(lip_points, axis=0)
    x2, y2 = np.max(lip_points, axis=0)

    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    mouth = frame[y1:y2, x1:x2]
    mouth = cv2.resize(mouth, (size, size))
    mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)

    return mouth


def process_video(video_path):
    filename = os.path.basename(video_path).replace(".mp4", "")
    save_path = os.path.join(SAVE_DIR, filename + ".npz")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = [results.multi_face_landmarks[0].landmark[i] for i in LIPS]
            mouth = extract_mouth_roi(frame, landmarks)
            frames.append(mouth)

    cap.release()

    if len(frames) > 0:
        frames = np.stack(frames, axis=0)
        np.savez(save_path, data=frames)
        print(f"Saved {save_path} with {frames.shape[0]} frames")
    else:
        print(f"WARNING: No face detected in {video_path}")


if __name__ == "__main__":
    videos = glob.glob(os.path.join(RAW_DIR, "**/*.mp4"), recursive=True)
    print(f"Found {len(videos)} videos")
    for v in videos:
        print("Processing:", v)
        process_video(v)
