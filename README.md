#Efficient Lip Reading using MobileNetV2 + TCN

## Data
Model was trained on Oxford BBC's Lip Reading in the Wild Dataset (LRW). https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

## extract_landmarks.py
This uses the MediaPipe Face Mesh to track lip landmarks in each video frame and limit the region of interest to 88x88. Resulting frame sequence used for mobilenet input

## mobilenet_batch.py
Utilize modified MobileNetV2 to extract per-frame visual features from preprocessed mouth ROIs. Save temporal feature sequence for TCN input.

## tcn_batch.py
Implement a Temporal Convolutional Network (TCN) to accumulate per-frame visual features over time and predict the spoken word from full video sequence.

## train_tcn.py
Train and validate the TCN on the extracted MobileNet features, handle variable-length clips, gradient clipping, and checkpointing for learning word classifications.

## analyze_model_performance.py
Evaluated the trained TCN model on extracted features (using LRW test set). Compute Top-K accuracy and confusion matrix and visualize. 
