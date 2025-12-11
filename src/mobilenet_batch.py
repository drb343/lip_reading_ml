import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import numpy as np
import glob
from config_local import ROI_DIR, SAVE_MOBILE_DIR

#Only use a single thread for proof of concept with lightweight computation
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

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


#Process a single .npz mouth_roi file
def process_video(npz_path):
    filename = os.path.basename(npz_path).replace(".npz", "")
    save_path = os.path.join(SAVE_MOBILE_DIR, filename + ".pt")

    #Load (29, 88, 88) mouth ROI frames
    data = np.load(npz_path)["data"]
    T = data.shape[0]

    all_features = []

    #Process the frames one by one ONLY, otherwise CPU will crash if all go to RAM
    for i in range(T):
        frame = data[i].astype(np.float32) / 255.0
        frame = (frame - MEAN) / STD

        #Shape should be  (1, 1, 88, 88)
        tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            feat = model(tensor)   #Output shape: (1, 1280)

        all_features.append(feat.cpu())

    #Concatenate into (29, 1280)
    features = torch.cat(all_features, dim=0)
    torch.save(features, save_path)
    print(f"Saved {save_path} — shape={tuple(features.shape)}")


if __name__ == "__main__":
    npz_files = glob.glob(os.path.join(ROI_DIR, "*.npz"))

    print(f"Found {len(npz_files)} ROI files.")

    for i, npz_path in enumerate(npz_files):
        print(f"[{i+1}/{len(npz_files)}] Processing: {npz_path}")
        process_video(npz_path)

    print("\nAll feature extraction complete!")
