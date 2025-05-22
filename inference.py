import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch_i3d import InceptionI3d

# === CONFIG ===
VIDEO_FOLDER = "data/WLASL2000"
LABELS_TXT = "preprocess/wlasl_class_list.txt"
MODEL_PATH = "models/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
NUM_CLASSES = 2000
OUTPUT_FILE = "inference_results.txt"

# === Load label list from TXT ===
def load_labels(label_path):
    with open(label_path, "r") as f:
        labels = []
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels

# === Preprocess video: (1, 3, T, 224, 224) ===
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    frames = np.array(frames).astype(np.float32)
    frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
    frames = np.expand_dims(frames, axis=0)     # (1, C, T, H, W)
    return torch.tensor(frames)

# === Run inference on single video ===
def run_inference(video_tensor, model):
    video_tensor = video_tensor.cuda()
    with torch.no_grad():
        logits = model(video_tensor)          # (1, num_classes, T)
        logits = torch.mean(logits, dim=2)    # (1, num_classes)
        probs = F.softmax(logits, dim=1)      # (1, num_classes)
        topk = torch.topk(probs, k=10, dim=1)
    return topk.indices[0].cpu().numpy(), topk.values[0].cpu().numpy()

# === Main ===
if __name__ == "__main__":
    print("🚀 Loading model...")
    model = InceptionI3d(num_classes=400, in_channels=3)
    model.replace_logits(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda().eval()

    labels = load_labels(LABELS_TXT)

    results = []

    print(f"🔍 Scanning folder: {VIDEO_FOLDER}")
    for fname in os.listdir(VIDEO_FOLDER):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        path = os.path.join(VIDEO_FOLDER, fname)
        video_tensor = preprocess_video(path)
        if video_tensor is None:
            print(f"⚠️ Skipped: {fname} (no frames)")
            continue
        print(f"\n📹 Video: {fname}")
        indices, confidences = run_inference(video_tensor, model)
        result_lines = [f"Video: {fname}"]
        for rank, (i, conf) in enumerate(zip(indices, confidences), 1):
            label = labels[i] if i < len(labels) else f"[{i}]"
            percentage = f"{conf * 100:.2f}%"
            result_lines.append(f"{rank}. {label:<20} ({percentage})")
            print(result_lines[-1])
        results.append("\n".join(result_lines))

    # Save results to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n\n".join(results))

    print(f"\n✅ Inference completed. Results saved to: {OUTPUT_FILE}")
