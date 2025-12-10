import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
import glob
from config_local import ROI_DIR, SAVE_MOBILE_DIR

# =======================================
# Raspberry Pi Performance Optimizations
# =======================================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ======== Directories ========
os.makedirs(SAVE_MOBILE_DIR, exist_ok=True)

# ======== LRW Normalization ========
MEAN = 0.421
STD = 0.165

# =======================================
# Load MobileNetV2 (modified for grayscale)
# =======================================
def load_mobilenet():
    model = mobilenet_v2(weights="IMAGENET1K_V1")

    # Change input layer from 3-channel to 1-channel
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )

    # Remove classifier — use feature descriptor only
    model.classifier = nn.Identity()

    model.eval()
    return model


model = load_mobilenet()


# =======================================
# Process a single *.npz mouth ROI file
# =======================================
def process_video(npz_path):
    filename = os.path.basename(npz_path).replace(".npz", "")
    save_path = os.path.join(SAVE_MOBILE_DIR, filename + ".pt")

    # Load (T, 88, 88) mouth ROI frames
    data = np.load(npz_path)["data"]
    T = data.shape[0]

    all_features = []

    # Process frames one-by-one (safe for Raspberry Pi)
    for i in range(T):
        frame = data[i].astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD

        # Shape: (1, 1, 88, 88)
        tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            feat = model(tensor)   # Output shape: (1, 1280)

        all_features.append(feat.cpu())

    # Concatenate into (T, 1280)
    features = torch.cat(all_features, dim=0)

    # Save tensor
    torch.save(features, save_path)
    print(f"Saved {save_path} — shape={tuple(features.shape)}")


# =======================================
# Main Loop — Process all npz files
# =======================================
if __name__ == "__main__":
    npz_files = glob.glob(os.path.join(ROI_DIR, "*.npz"))

    print(f"Found {len(npz_files)} ROI files.")

    for i, npz_path in enumerate(npz_files):
        print(f"[{i+1}/{len(npz_files)}] Processing: {npz_path}")
        process_video(npz_path)

    print("\nAll feature extraction complete!")
