

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model architecture
from tcn_batch import TCN
from config_local import SAVE_MOBILE_DIR


class LRWDataset(Dataset):
    """Dataset for loading preprocessed lip reading features"""
    def __init__(self, feature_dir, word_to_idx):
        all_files = glob.glob(os.path.join(feature_dir, "*.pt"))
        
        self.feature_files = []
        skipped = 0
        for file_path in all_files:
            filename = os.path.basename(file_path)
            word = filename.split('_')[0]
            if word in word_to_idx:
                self.feature_files.append(file_path)
            else:
                skipped += 1
        
        self.word_to_idx = word_to_idx
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        feature_path = self.feature_files[idx]
        features = torch.load(feature_path, weights_only=True)
        
        if features.dtype == torch.float16:
            features = features.float()
        
        # Extract word from filename
        filename = os.path.basename(feature_path)
        word = filename.split('_')[0]
        label = self.word_to_idx[word]
        
        return features, label, word


def collate_fn(batch):
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    words = [item[2] for item in batch]
    
    max_len = max(f.shape[0] for f in features_list)
    
    padded_features = []
    for feat in features_list:
        if feat.shape[0] < max_len:
            padding = torch.zeros(max_len - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
        padded_features.append(feat)
    
    features_batch = torch.stack(padded_features, dim=0)
    
    return features_batch, labels, words


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict):
        num_classes = checkpoint.get('num_classes', 500)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        # Infer num_classes from final layer
        num_classes = state_dict['fc.3.weight'].shape[0]
    
    model = TCN(
        input_dim=1280,
        num_classes=num_classes,
        num_channels=[256, 256, 256, 256],
        kernel_size=3,
        dropout=0.2
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {num_classes} classes")
    return model, num_classes


def compute_predictions(model, dataloader, device):

    all_preds = []
    all_labels = []
    all_words = []
    all_logits = []
    
    with torch.no_grad():
        for features, labels, words in tqdm(dataloader, desc="Inference"):
            features = features.to(device)
            
            logits = model(features)
            all_logits.append(logits.cpu())
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_words.extend(words)
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    
    return all_preds, all_labels, all_words, all_logits


def compute_topk_accuracy(logits, labels, k_values=[1, 3, 5, 10]):
    logits_tensor = torch.from_numpy(logits)
    labels_tensor = torch.from_numpy(labels)
    
    results = {}
    for k in k_values:
        # Get top k predictions
        _, topk_preds = logits_tensor.topk(k, dim=1, largest=True, sorted=True)
        # Check if true label is in top k
        correct = topk_preds.eq(labels_tensor.view(-1, 1).expand_as(topk_preds))
        topk_acc = correct.any(dim=1).float().mean().item() * 100
        results[f'top{k}'] = topk_acc
        print(f"Top-{k} Accuracy: {topk_acc:.2f}%")
    
    return results


def find_most_mispredicted_words(preds, labels, words, idx_to_word, top_n=20):
    error_counts = Counter()
    
    for pred, true, word in zip(preds, labels, words):
        if pred != true:
            # Count errors for the true word
            error_counts[word] += 1
    
    most_mispredicted = error_counts.most_common(top_n)
    return most_mispredicted


def find_most_confused_pairs(preds, labels, words, idx_to_word, top_n=20):
    confusion_counts = Counter()
    
    for pred, true, word in zip(preds, labels, words):
        if pred != true:
            pred_word = idx_to_word[pred]
            true_word = idx_to_word[true]
            
            pair = tuple(sorted([true_word, pred_word]))
            confusion_counts[pair] += 1
    
    most_confused = confusion_counts.most_common(top_n)
    return most_confused


def plot_confusion_matrix_subset(preds, labels, words, idx_to_word, mispredicted_words, save_path):
    # Extract the top 20 most mis-predicted words
    target_words = [word for word, count in mispredicted_words]
    target_word_set = set(target_words)
    
    all_pred_words = set()
    for pred, label, word in zip(preds, labels, words):
        if word in target_word_set:
            pred_word = idx_to_word[pred]
            all_pred_words.add(pred_word)
    
    vocab_words = target_words.copy()
    for pred_word in sorted(all_pred_words):
        if pred_word not in target_word_set:
            vocab_words.append(pred_word)
    
    word_to_new_idx = {word: i for i, word in enumerate(vocab_words)}
    
    filtered_preds = []
    filtered_labels = []
    
    for pred, label, word in zip(preds, labels, words):
        if word in target_word_set:
            pred_word = idx_to_word[pred]
            if pred_word in word_to_new_idx:
                filtered_preds.append(word_to_new_idx[pred_word])
                filtered_labels.append(word_to_new_idx[word])
    
    if len(filtered_preds) == 0:
        print("Warning: No samples found for confused words")
        return
    
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=range(len(vocab_words)))
    
    # Keep only the rows for the target words and most confused columns
    cm_subset = cm[:len(target_words), :]
    
    top_cols = cm_subset.sum(axis=0).argsort()[-20:][::-1]
    cm_subset = cm_subset[:, top_cols]
    
    # Get corresponding word labels
    row_labels = target_words
    col_labels = [vocab_words[i] for i in top_cols]
    
    cm_normalized = cm_subset.astype('float') / (cm_subset.sum(axis=1, keepdims=True) + 1e-10)
    
    # Plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Normalized Frequency'})
    plt.title('Confusion Matrix: Top 20 Most Mis-Predicted Words', fontsize=16, pad=20)
    plt.xlabel('Predicted Word', fontsize=12)
    plt.ylabel('True Word', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_accuracy_over_epochs(history_path, save_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    epochs = list(range(1, len(train_acc) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Over Training Epochs', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to {save_path}")
    plt.close()


def plot_loss_over_epochs(history_path, save_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = list(range(1, len(train_loss) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Model Loss Over Training Epochs', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to {save_path}")
    plt.close()


def save_analysis_report(output_path, topk_results, confused_pairs, history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Training summary
        f.write("TRAINING SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Epochs: {len(history['train_acc'])}\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}\n\n")
        
        # Top-K accuracy
        f.write("TOP-K ACCURACY RESULTS\n")
        f.write("-" * 80 + "\n")
        for key, value in topk_results.items():
            f.write(f"{key.upper()}: {value:.2f}%\n")
        f.write("\n")
        
        # Most confused pairs
        f.write("TOP 20 MOST FREQUENTLY CONFUSED WORD PAIRS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Word 1':<20} {'Word 2':<20} {'Count':<10}\n")
        f.write("-" * 80 + "\n")
        for i, (pair, count) in enumerate(confused_pairs, 1):
            f.write(f"{i:<6} {pair[0]:<20} {pair[1]:<20} {count:<10}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Analysis report saved to {output_path}")


def main():
    # Configuration - use paths relative to parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    MODEL_PATH = os.path.join(parent_dir, "checkpoint_epoch_30.pth")
    HISTORY_PATH = os.path.join(parent_dir, "history.json")
    VOCAB_PATH = os.path.join(script_dir, "words.txt")
    OUTPUT_DIR = os.path.join(parent_dir, "analysis_results")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load vocabulary
    print("Loading vocabulary...")
    with open(VOCAB_PATH, 'r') as f:
        words = [line.strip() for line in f.readlines()]
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for idx, word in enumerate(words)}
    print(f"Vocabulary loaded: {len(word_to_idx)} words\n")
    
    # Load model
    model, num_classes = load_model(MODEL_PATH, device)
    
    # Create dataset and dataloader
    print("\nLoading dataset...")
    dataset = LRWDataset(SAVE_MOBILE_DIR, word_to_idx)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Use subset for faster analysis (optional)
    USE_SUBSET = True
    SUBSET_SIZE = 20000
    if USE_SUBSET and len(dataset) > SUBSET_SIZE:
        indices = torch.randperm(len(dataset))[:SUBSET_SIZE].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using subset of {len(dataset)} samples for analysis")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Get predictions
    preds, labels, words, logits = compute_predictions(model, dataloader, device)
    
    
    overall_acc = (preds == labels).mean() * 100
    print(f"Overall Accuracy: {overall_acc:.2f}%\n")
    
    print("Top-K Accuracy:")
    topk_results = compute_topk_accuracy(logits, labels, k_values=[1, 3, 5, 10])
    
    # Find most mis-predicted words
    print("\nFinding most mis-predicted words...")
    mispredicted_words = find_most_mispredicted_words(preds, labels, words, idx_to_word, top_n=20)
    print(f"\nTop 10 most mis-predicted words:")
    for i, (word, count) in enumerate(mispredicted_words[:10], 1):
        print(f"{i}. {word}: {count} errors")
    
    # Find confused pairs
    print("\nFinding most confused word pairs...")
    confused_pairs = find_most_confused_pairs(preds, labels, words, idx_to_word, top_n=20)
    print(f"\nTop 10 most confused pairs:")
    for i, (pair, count) in enumerate(confused_pairs[:10], 1):
        print(f"{i}. {pair[0]} <-> {pair[1]}: {count} times")

    
    # Confusion matrix for most mis-predicted words
    plot_confusion_matrix_subset(
        preds, labels, words, idx_to_word, mispredicted_words,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_top20.png")
    )
    
    # Accuracy over epochs
    if os.path.exists(HISTORY_PATH):
        plot_accuracy_over_epochs(
            HISTORY_PATH,
            save_path=os.path.join(OUTPUT_DIR, "accuracy_over_epochs.png")
        )
        
        plot_loss_over_epochs(
            HISTORY_PATH,
            save_path=os.path.join(OUTPUT_DIR, "loss_over_epochs.png")
        )
        
        # Save report
        save_analysis_report(
            os.path.join(OUTPUT_DIR, "analysis_report.txt"),
            topk_results,
            confused_pairs,
            HISTORY_PATH
        )
    else:
        print(f"History file not found at {HISTORY_PATH}")


if __name__ == "__main__":
    main()
