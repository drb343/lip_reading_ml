import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
import glob
from config_local import ROI_DIR, SAVE_MOBILE_DIR
from tqdm import tqdm

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(SAVE_MOBILE_DIR, exist_ok=True)

#LRW must be normalized to these values
MEAN = 0.421
STD = 0.165

#Load mobilenet v2
def load_mobilenet():
    model = mobilenet_v2(weights="IMAGENET1K_V1")

    #Modify it to grayscale to match preprocessed input
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False #Accept one channel only
    )

    model.classifier = nn.Identity()

    model.eval()
    return model


model = load_mobilenet()
model = model.to(device)

# mixed precision for faster inference
if device.type == 'cuda':
    model = model.half()  # Use FP16 for 2x speedup
    print("Mixed precision (FP16) enabled")


# process all frames of the vid at once
def process_video(npz_path):
    filename = os.path.basename(npz_path).replace(".npz", "")
    save_path = os.path.join(SAVE_MOBILE_DIR, filename + ".pt")

    #Load (29, 88, 88) mouth ROI frames
    data = np.load(npz_path)["data"] 

    #Normalize all frames at once
    frames = data.astype(np.float32) / 255.0
    frames = (frames - MEAN) / STD
    
    #Convert to tensor: (T, 1, 88, 88)
    frames_tensor = torch.from_numpy(frames).unsqueeze(1).to(device)
    
    # use FP16 if model is in half precision
    if device.type == 'cuda':
        frames_tensor = frames_tensor.half()
    
    # process all frames at once
    with torch.no_grad():
        features = model(frames_tensor)  # (29, 1280)
    
    # save to disk
    torch.save(features.cpu(), save_path)
    return tuple(features.shape)


if __name__ == "__main__":
    npz_files = glob.glob(os.path.join(ROI_DIR, "*.npz"))

    print(f"Found {len(npz_files)} ROI files.")
    print(f"Device: {device}")
    
    # process with progress bar
    for npz_path in tqdm(npz_files, desc="Extracting features"):
        shape = process_video(npz_path)
    
    print(f"\nAll feature extraction complete! Processed {len(npz_files)} videos.")
    print(f"Output shape per video: {shape}")
    print(f"Saved to: {SAVE_MOBILE_DIR}")
    
    print(f"\nAll feature extraction complete! Processed {len(npz_files)} videos.")
    print(f"Output shape per video: {shape}")
    print(f"Saved to: {SAVE_MOBILE_DIR}")
