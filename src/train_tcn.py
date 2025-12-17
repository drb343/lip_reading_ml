import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
from config_local import SAVE_MOBILE_DIR
from tcn_batch import TCN
import json
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

def load_vocabulary(vocab_path="words.txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, vocab_path)
    
    with open(full_path, 'r') as f:
        words = [line.strip() for line in f.readlines()]
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    return words, word_to_idx

# pad sequences - frustratingly some clips are 28 frames seemingly
def collate_fn(batch):
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    max_len = max(f.shape[0] for f in features_list)
    
    padded_features = []
    for feat in features_list:
        if feat.shape[0] < max_len:
            padding = torch.zeros(max_len - feat.shape[0], feat.shape[1])
            feat = torch.cat([feat, padding], dim=0)
        padded_features.append(feat)
    
    features_batch = torch.stack(padded_features, dim=0)
    
    return features_batch, labels

class LRWDataset(Dataset):
    def __init__(self, feature_dir, word_to_idx, cache_in_ram=False):
        self.feature_files = glob.glob(os.path.join(feature_dir, "*.pt"))
        self.word_to_idx = word_to_idx
        self.cache_in_ram = cache_in_ram
        self.cache = {}
        
        # Preload all data into RAM if caching enabled - unfortunately I don't have enough ram to actually do this locally
        if self.cache_in_ram:
            for idx in tqdm(range(len(self.feature_files)), desc="Caching data"):
                feature_path = self.feature_files[idx]
                features = torch.load(feature_path, weights_only=True)
                if features.dtype == torch.float16:
                    features = features.float()
                if not features.is_contiguous():
                    features = features.contiguous()
                self.cache[idx] = features
            print(f"Cached {len(self.cache)} samples in RAM")
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        # use cached data if available
        if self.cache_in_ram:
            features = self.cache[idx]
        else:
            feature_path = self.feature_files[idx]
            features = torch.load(feature_path, weights_only=True)
            
            # convert to float32 if necessary
            if features.dtype == torch.float16:
                features = features.float()
            
            if not features.is_contiguous():
                features = features.contiguous()
        
        # fetch word from filename
        filename = os.path.basename(self.feature_files[idx])
        word = filename.split('_')[0]
        label = self.word_to_idx[word]
        
        return features, label

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, accumulation_steps=1, debug=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (features, labels) in enumerate(tqdm(dataloader, desc="Training")):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # mixed precision forward pass
        if scaler is not None:
            with autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels) / accumulation_steps
            
            # output a bunch of model info for debugging
            if debug and batch_idx == 0:
                print(f"Features shape: {features.shape}, mean: {features.mean():.4f}, std: {features.std():.4f}")
                print(f"Labels shape: {labels.shape}, unique labels in batch: {labels.unique().numel()}")
                print(f"Outputs shape: {outputs.shape}")
                print(f"Output logits - mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")
                print(f"Loss: {loss.item() * accumulation_steps:.4f}")
                with torch.no_grad():
                    probs = torch.softmax(outputs, dim=1)
                    print(f"Max prob: {probs.max():.4f}, min prob: {probs.min():.6f}")
                    _, pred = outputs.max(1)
                    print(f"Predicted labels sample: {pred[:10].cpu().tolist()}")
                    print(f"True labels sample: {labels[:10].cpu().tolist()}")
            
            scaler.scale(loss).backward()
            
            # using grad accumulation, so we only update weights every so often
            if (batch_idx + 1) % accumulation_steps == 0:
                
                # clipping gradients to avoid grad explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # make sure gradient looks good
                if debug and batch_idx == 0:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"Gradient norm after clipping: {total_norm:.4f}\n")
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(features)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# val function
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast('cuda') if device.type == 'cuda' else torch.no_grad():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# main training loop
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ## use tf32 if possible
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster training")
    
    # for small tests
    USE_SUBSET = False  
    SUBSET_SIZE = 50000 
    
    CACHE_IN_RAM = False  # had to set to false since I don't have enough ram
    
    # load in vocabulary
    words, word_to_idx = load_vocabulary()
    num_classes = len(words)
    print(f"Loaded {num_classes} words")
    
    # make dataset and dataloader
    dataset = LRWDataset(SAVE_MOBILE_DIR, word_to_idx, cache_in_ram=CACHE_IN_RAM)
    
    # use datasubset if enabled
    if USE_SUBSET and len(dataset) > SUBSET_SIZE:
        dataset = torch.utils.data.Subset(dataset, range(SUBSET_SIZE))
        print(f"Using subset of {len(dataset)} samples for faster training")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
      
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True, prefetch_factor=4)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # initialize model
    model = TCN(input_dim=1280, num_classes=num_classes, 
                num_channels=[256, 256, 256, 256], kernel_size=3, dropout=0.2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # if possible, resume training from last checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if os.path.exists('best_tcn_model.pth'):
        print("Loading checkpoint...")
        try:
            checkpoint = torch.load('best_tcn_model.pth', map_location=device, weights_only=False)
            
            # Handle both full checkpoint and state_dict only formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                if 'best_val_acc' in checkpoint:
                    best_val_acc = checkpoint['best_val_acc']
                print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
            else:
                # Just state dict
                model.load_state_dict(checkpoint)
                print("Loaded model weights (no optimizer state)")
        except Exception as e:
            print(f"Could not load checkpoint (architecture mismatch?): {e}")
            print("Starting training from scratch with new architecture...")
            start_epoch = 0
            best_val_acc = 0.0
    
    # mixed precision scaler
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Mixed precision training enabled")
    

                    #### ACTUAL TRAINING LOOP HERE #####

    # training params
    num_epochs = 30  
    accumulation_steps = 1 
    validation_frequency = 1 
    
    # keeping track of all the data I could think of
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': [],
        'best_val_acc': best_val_acc,
        'best_epoch': start_epoch
    }
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        debug_mode = (epoch == start_epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, accumulation_steps, debug=debug_mode)
        
        # only validate every n epochs
        if epoch == 0 or (epoch + 1) % validation_frequency == 0 or epoch == num_epochs - 1:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = history['val_loss'][-1] if history['val_loss'] else 0, history['val_acc'][-1] if history['val_acc'] else 0
            print("Skipping validation this epoch for faster training")
        
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()  
        
        # save our best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), 'best_tcn_model.pth')
            print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
        
        # checkpoint model every few epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # store training history
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=4)
    
    # final summary
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}% at epoch {history['best_epoch']}")