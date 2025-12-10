import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import ast
import os
import sys
import time
import argparse
import glob
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# I got help from GenAI in coding

# Pick the best device available - GPU if we have one, otherwise CPU
def get_device():
    """Get the appropriate device for training"""
    if torch.cuda.is_available():
        # Make sure we're using the first GPU
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print(f"CUDA device selected: {torch.cuda.get_device_name(0)}")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        print("WARNING: No GPU detected! Using CPU (training will be very slow)")
        return torch.device('cpu')

# Global device variable - gets set in main()
device = None

# Configuration
CONFIG = {
    'image_dirs': ['cleaned_posters_local', 'cleaned_posters'],  # Check both directories
    'csv_file': 'meta_processed.csv',
    'image_size': (224, 224),
    'batch_size': 64,  # Bumped up from 32 to speed things up (lower this if you run out of memory)
    'num_epochs': 30,  # Maximum epochs - early stopping might finish sooner if things aren't improving
    'learning_rate': 0.0015,  # Turned up a bit to help the model learn faster and avoid getting stuck
    'early_stopping_patience': 7,  # Stop training if no improvement for 7 epochs
    'use_mixed_precision': True,  # Use FP16 for 2x speedup (requires CUDA)
    'train_split': 0.8,
    'val_split': 0.1,
    'num_workers': 4,  # Reduced for Windows compatibility (use 0-4 on Windows, 4-16 on Linux)
    'model_save_path': 'genre_predictor_model_BCE.pth',  # BCE Loss version
    'checkpoint_dir': 'checkpoints',  # Directory to save epoch checkpoints
    'save_every_epoch': True,  # Save checkpoint every epoch
    'use_pretrained': True,  # Use ImageNet pre-trained weights (much better performance)
    'freeze_backbone': False,  # Set to True if you want to freeze the ResNet layers (trains faster but less flexible)
    'freeze_early_layers': False,  # Freeze the early conv layers to speed things up
    'use_class_weights': False,  # Use class weights to balance out imbalanced classes
    'gradient_clip': 1.0,  # Clip gradients to stop them from exploding
    'weight_decay': 5e-5,  # L2 regularization - lowered this a bit (used to be 1e-4)
    'scheduler_patience': 3,  # Patience for ReduceLROnPlateau
    'scheduler_factor': 0.5,  # Factor for ReduceLROnPlateau
    'learn_thresholds': False,  # Fixed threshold at 0.5 (not learnable)
    'fixed_threshold': 0.5  # Fixed threshold value for predictions
}


class PosterDataset(Dataset):
    """Dataset class for movie posters and genres"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # If image can't be loaded, create a black image
            image = Image.new('RGB', CONFIG['image_size'], color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.FloatTensor(self.labels[idx])
        
        return image, label


class GenreCNN(nn.Module):
    """Standard ResNet50-based model for genre prediction using transfer learning"""
    
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False, 
                 freeze_early_layers=False, learn_thresholds=False):
        super(GenreCNN, self).__init__()
        
        # Load ResNet50 with ImageNet weights if we want pretrained
        # Try the newer API first, fall back to old one if needed
        try:
            resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        except TypeError:
            # Older PyTorch versions use this instead
            resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze all the backbone layers if we're doing fine-tuning
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        
        # Swap out the final layer - ResNet was trained for 1000 ImageNet classes,
        # but we need it to output our number of genre classes instead
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, num_classes)
        
        self.model = resnet
        self.learn_thresholds = False  # Not actually used anymore, just keeping this for compatibility
    
    def forward(self, x):
        return self.model(x)


def parse_genres(genre_str):
    """Parse genre string to list"""
    try:
        if pd.isna(genre_str) or genre_str == '':
            return []
        # Try parsing it as a Python list/string
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list):
            return genres
        elif isinstance(genres, str):
            return [genres]
        return []
    except:
        return []


def load_data():
    """Load and preprocess data"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading CSV file: {CONFIG['csv_file']}")
    
    # Load CSV
    start_time = time.time()
    df = pd.read_csv(CONFIG['csv_file'], header=None, low_memory=False)
    load_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CSV loaded in {load_time:.2f}s - {len(df)} rows")
    
    # Extract poster IDs and genres
    poster_ids = df[0].astype(str).tolist()
    genre_strings = df[3].astype(str).tolist()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing images and genres...")
    print(f"  Checking directories: {', '.join(CONFIG['image_dirs'])}")
    
    # Parse genres
    all_genres = []
    valid_indices = []
    image_paths = []
    missing_images = 0
    missing_genres = 0
    images_from_dir1 = 0
    images_from_dir2 = 0
    
    def find_image_path(poster_id):
        """Find image in any of the configured directories"""
        for image_dir in CONFIG['image_dirs']:
            image_path = os.path.join(image_dir, f"{poster_id}.jpg")
            if os.path.exists(image_path):
                return image_path, image_dir
        return None, None
    
    for idx, (poster_id, genre_str) in enumerate(tqdm(zip(poster_ids, genre_strings), 
                                                       total=len(poster_ids), 
                                                       desc="Processing data",
                                                       unit="rows",
                                                       file=sys.stdout,
                                                       dynamic_ncols=True,
                                                       leave=True)):
        genres = parse_genres(genre_str)
        if len(genres) > 0:
            image_path, found_dir = find_image_path(poster_id)
            if image_path:
                all_genres.append(genres)
                valid_indices.append(idx)
                image_paths.append(image_path)
                # Keep track of where we found each image
                if found_dir == CONFIG['image_dirs'][0]:
                    images_from_dir1 += 1
                elif found_dir == CONFIG['image_dirs'][1]:
                    images_from_dir2 += 1
            else:
                missing_images += 1
        else:
            missing_genres += 1
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data processing complete:")
    print(f"  ✓ Found {len(image_paths)} valid images with genres")
    if len(CONFIG['image_dirs']) >= 2:
        print(f"    - From {CONFIG['image_dirs'][0]}: {images_from_dir1} images")
        print(f"    - From {CONFIG['image_dirs'][1]}: {images_from_dir2} images")
    print(f"  ✗ {missing_images} images missing")
    print(f"  ✗ {missing_genres} rows with missing/invalid genres")
    
    # Convert genre lists into binary vectors using MultiLabelBinarizer
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Encoding genre labels...")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(all_genres)
    num_classes = len(mlb.classes_)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Label encoding complete:")
    print(f"  Number of genre classes: {num_classes}")
    print(f"  Genre classes: {', '.join(mlb.classes_[:10])}{'...' if num_classes > 10 else ''}")
    
    # Show genre distribution
    genre_counts = labels.sum(axis=0)
    top_genres = sorted(zip(mlb.classes_, genre_counts), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top 10 genres by frequency:")
    for genre, count in top_genres:
        print(f"    - {genre}: {count} movies ({count/len(labels)*100:.1f}%)")
    
    return image_paths, labels, mlb, num_classes


def create_data_loaders(image_paths, labels, mlb):
    """Create train, validation, and test data loaders"""
    
    print("=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Splitting data...")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=1-CONFIG['train_split'], random_state=42
    )
    
    val_size = CONFIG['val_split'] / (1 - CONFIG['train_split'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_size, random_state=42
    )
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data split complete:")
    print(f"  Train samples: {len(X_train)} ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"  Validation samples: {len(X_val)} ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"  Test samples: {len(X_test)} ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    # Set up data augmentation - helps the model generalize better
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating data transforms...")
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to a bit bigger than final size
        transforms.RandomCrop(CONFIG['image_size']),  # Random crop to add variety
        transforms.RandomHorizontalFlip(0.5),  # Flip horizontally sometimes
        transforms.RandomVerticalFlip(0.2),  # Flip vertically occasionally (posters can be upside down)
        transforms.RandomRotation(15),  # Rotate up to 15 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),  # Shift, scale, and shear a bit
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Vary the colors
        transforms.ToTensor(),  # Convert to tensor (needed before RandomErasing)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Randomly erase parts of the image
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating datasets...")
    train_dataset = PosterDataset(X_train, y_train, transform=train_transform)
    val_dataset = PosterDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = PosterDataset(X_test, y_test, transform=val_test_transform)
    
    # Create the data loaders
    # Windows has issues with multiprocessing, so we set workers to 0 there
    is_windows = sys.platform == 'win32'
    effective_num_workers = 0 if is_windows else CONFIG['num_workers']
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating data loaders (batch_size={CONFIG['batch_size']}, workers={effective_num_workers})...")
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=True, num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=False, num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'], 
        shuffle=False, num_workers=effective_num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Data loaders created successfully")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, mlb, y_train


def calculate_class_weights(labels):
    """Calculate class weights for imbalanced multi-label classification"""
    # Count how many times each class appears
    pos_counts = labels.sum(axis=0).astype(float)
    total_samples = len(labels)
    neg_counts = total_samples - pos_counts
    
    # Make sure we don't divide by zero
    pos_counts = np.maximum(pos_counts, 1.0)
    neg_counts = np.maximum(neg_counts, 1.0)
    
    # Give more weight to classes that appear less often
    weights = total_samples / (len(labels[0]) * pos_counts)
    
    # Normalize so the average weight is 1.0
    weights = weights / weights.mean()
    
    return torch.FloatTensor(weights)


def train_model(model, train_loader, val_loader, mlb, train_labels=None, 
                start_epoch=0, optimizer=None, scheduler=None, scaler=None,
                train_losses=None, val_losses=None, best_val_loss=None, 
                epochs_without_improvement=0):
    """Train the model
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        mlb: MultiLabelBinarizer
        train_labels: Pre-computed training labels (optional)
        start_epoch: Starting epoch (for resuming training)
        optimizer: Optimizer (if resuming, pass existing optimizer)
        scheduler: Scheduler (if resuming, pass existing scheduler)
        scaler: GradScaler (if resuming, pass existing scaler)
        train_losses: Previous training losses (if resuming)
        val_losses: Previous validation losses (if resuming)
        best_val_loss: Previous best validation loss (if resuming)
        epochs_without_improvement: Previous epochs without improvement (if resuming)
    """
    
    print("=" * 60)
    print("TRAINING MODEL")
    if start_epoch > 0:
        print(f"RESUMING FROM EPOCH {start_epoch}")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing training...")
    
    # Use mixed precision training if we can - makes things about 2x faster on modern GPUs
    use_amp = CONFIG['use_mixed_precision'] and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print(f"  ✓ Mixed precision training (FP16) enabled - expect ~2x speedup")
    else:
        if CONFIG['use_mixed_precision'] and device.type != 'cuda':
            print(f"  ⚠ Mixed precision requested but CUDA not available - using FP32")
        else:
            print(f"  Using FP32 precision")
    
    # Calculate class weights if enabled
    if CONFIG['use_class_weights']:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating class weights...")
        # Use the labels we already have if they're available (way faster)
        if train_labels is not None:
            print(f"  Using pre-computed training labels (fast method)...")
            class_weights = calculate_class_weights(train_labels).to(device)
        else:
            # Otherwise we have to go through the loader to collect them (slower)
            print(f"  Collecting labels from data loader (this may take a moment)...")
            all_train_labels = []
            for batch_idx, (_, labels) in enumerate(tqdm(train_loader, desc="Collecting labels", 
                                                         unit="batch", leave=False, ncols=100)):
                if isinstance(labels, torch.Tensor):
                    all_train_labels.append(labels.cpu().numpy())
                else:
                    all_train_labels.append(labels)
            all_train_labels = np.vstack(all_train_labels)
            class_weights = calculate_class_weights(all_train_labels).to(device)
        
        print(f"  Class weights calculated (min: {class_weights.min():.3f}, max: {class_weights.max():.3f}, mean: {class_weights.mean():.3f})")
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        print(f"  Using weighted BCEWithLogitsLoss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"  Using standard BCEWithLogitsLoss")
    
    # Set up optimizer, scheduler, etc. if we're not resuming from a checkpoint
    if optimizer is None:
        # Use different learning rates for the backbone vs the classifier
        if CONFIG['freeze_backbone']:
            # If backbone is frozen, only train the final layer
            params_to_optimize = list(model.model.fc.parameters())
            optimizer = optim.Adam(params_to_optimize, 
                                  lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training configuration:")
            print(f"  Optimizer: Adam (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']}) - Training classifier only")
        else:
            # Use a lower learning rate for the pretrained backbone, higher for the new classifier
            # Collect all the backbone parameters (everything except the final fc layer)
            backbone_params = []
            for name, param in model.model.named_parameters():
                if 'fc' not in name:
                    if param.requires_grad:
                        backbone_params.append(param)
            
            classifier_params = list(model.model.fc.parameters())
            
            optimizer_groups = []
            
            # Only add backbone params if there are any trainable ones
            if len(backbone_params) > 0:
                optimizer_groups.append({
                    'params': backbone_params, 
                    'lr': CONFIG['learning_rate'] * 0.1, 
                    'weight_decay': CONFIG['weight_decay']
                })
            
            optimizer_groups.append({
                'params': classifier_params, 
                'lr': CONFIG['learning_rate'], 
                'weight_decay': CONFIG['weight_decay']
            })
            
            optimizer = optim.Adam(optimizer_groups)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training configuration:")
            if len(backbone_params) > 0:
                print(f"  Optimizer: Adam - Backbone LR: {CONFIG['learning_rate'] * 0.1} (trainable layers only), Classifier LR: {CONFIG['learning_rate']}")
            else:
                print(f"  Optimizer: Adam - Classifier LR: {CONFIG['learning_rate']} (backbone fully frozen)")
            print(f"  Weight decay: {CONFIG['weight_decay']}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using optimizer from checkpoint")
    
    # Use ReduceLROnPlateau to automatically lower the learning rate when we're not improving
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=CONFIG['scheduler_factor'], 
            patience=CONFIG['scheduler_patience'], min_lr=1e-7
        )
        print(f"  Learning rate scheduler: ReduceLROnPlateau (factor={CONFIG['scheduler_factor']}, patience={CONFIG['scheduler_patience']})")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using scheduler from checkpoint")
    
    if scaler is None and use_amp:
        scaler = torch.cuda.amp.GradScaler()
    elif scaler is not None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using scaler from checkpoint")
    
    print(f"  Gradient clipping: {CONFIG['gradient_clip']}")
    print(f"  Early stopping patience: {CONFIG['early_stopping_patience']} epochs")
    print(f"  Max epochs: {CONFIG['num_epochs']}")
    if start_epoch > 0:
        print(f"  Resuming from epoch: {start_epoch}")
    
    if best_val_loss is None:
        best_val_loss = float('inf')
    if train_losses is None:
        train_losses = []
    if val_losses is None:
        val_losses = []
    if epochs_without_improvement is None:
        epochs_without_improvement = 0
    
    training_start_time = time.time()
    
    # Make sure the checkpoint directory exists if we're saving every epoch
    if CONFIG['save_every_epoch']:
        import os
        checkpoint_dir = CONFIG.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint directory: {checkpoint_dir} (will save every epoch)")
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        epoch_start_time = time.time()
        print("\n" + "-" * 60)
        print(f"EPOCH [{epoch+1}/{CONFIG['num_epochs']}]")
        print("-" * 60)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting epoch {epoch+1}...")
        print(f"  Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                         unit="batch", 
                         file=sys.stdout,
                         dynamic_ncols=True,
                         leave=False,
                         ncols=100)
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Double-check everything is on the right device (only do this once)
            if batch_idx == 0 and epoch == 0:
                model_device = next(model.parameters()).device
                print(f"\n[DEBUG] First batch device check:")
                print(f"  Images device: {images.device}")
                print(f"  Labels device: {labels.device}")
                print(f"  Model device: {model_device}")
                if device.type == 'cuda' and images.device.type != 'cuda':
                    print(f"  ⚠ WARNING: Images not on GPU! This will cause CPU usage.")
                if device.type == 'cuda' and model_device.type != 'cuda':
                    print(f"  ⚠ ERROR: Model not on GPU! Training will be very slow.")
                    print(f"  Attempting to move model to GPU...")
                    model = model.cuda()
                    print(f"  Model moved to: {next(model.parameters()).device}")
            
            optimizer.zero_grad()
            
            # Do the forward pass and backward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                
                # Clip gradients to prevent them from getting too large
                if CONFIG['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Clip gradients to keep them from exploding
                if CONFIG['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
                
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update the progress bar with current average loss
            current_avg_loss = train_loss / train_batches
            train_pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'batch': f'{batch_idx+1}/{len(train_loader)}'
            })
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        train_time = time.time() - epoch_start_time
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training complete:")
        print(f"  Average train loss: {avg_train_loss:.4f}")
        print(f"  Training time: {train_time:.2f}s ({train_time/60:.2f} min)")
        
        # Now validate on the validation set
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running validation...")
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", 
                       unit="batch",
                       file=sys.stdout,
                       dynamic_ncols=True,
                       leave=False,
                       ncols=100)
        
        with torch.no_grad():
            # Use a fixed threshold of 0.5 to convert probabilities to predictions
            threshold = 0.5
            
            for batch_idx, (images, labels) in enumerate(val_pbar):
                images = images.to(device)
                labels = labels.to(device)
                
                # Use mixed precision here too to speed things up
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Convert logits to probabilities and then to binary predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                
                # Make sure we predict at least one genre per movie
                # If nothing is above threshold, just pick the highest probability one
                for i in range(preds.shape[0]):
                    if preds[i].sum() == 0:
                        max_prob_idx = probs[i].argmax().item()
                        preds[i, max_prob_idx] = 1.0
                
                val_correct += (preds == labels).all(dim=1).sum().item()
                val_total += labels.size(0)
                
                current_avg_loss = val_loss / val_batches
                current_acc = val_correct / val_total if val_total > 0 else 0
                val_pbar.set_postfix({
                    'loss': f'{current_avg_loss:.4f}',
                    'acc': f'{current_acc:.4f}',
                    'batch': f'{batch_idx+1}/{len(val_loader)}'
                })
        
        avg_val_loss = val_loss / val_batches
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_time = time.time() - epoch_start_time - train_time
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation complete:")
        print(f"  Average val loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"  Validation time: {val_time:.2f}s")
        
        # Update the learning rate based on how validation loss is doing
        scheduler.step(avg_val_loss)
        
        # Print a summary of this epoch
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        remaining_epochs = CONFIG['num_epochs'] - (epoch + 1)
        estimated_remaining = (elapsed_time / (epoch + 1)) * remaining_epochs
        
        print(f"\n[Epoch {epoch+1} Summary]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Epoch time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        print(f"  Total elapsed: {elapsed_time/60:.2f} min | Est. remaining: {estimated_remaining/60:.2f} min")
        
        # Save a checkpoint every epoch so we can resume if needed
        if CONFIG['save_every_epoch']:
            checkpoint_dir = CONFIG.get('checkpoint_dir', 'checkpoints')
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'mlb': mlb,
                'num_classes': len(mlb.classes_),
                'config': CONFIG,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'epochs_without_improvement': epochs_without_improvement,
                'learned_thresholds': None,
                'learn_thresholds': False
            }, epoch_checkpoint_path)
            print(f"  ✓ Saved epoch checkpoint: {epoch_checkpoint_path}")
        
        # Save the best model so far and check if we should stop early
        if avg_val_loss < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = avg_val_loss
            improvement = ((prev_best - avg_val_loss) / prev_best * 100) if prev_best != float('inf') else 0
            epochs_without_improvement = 0  # Reset the counter since we improved
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'mlb': mlb,
                'num_classes': len(mlb.classes_),
                'config': CONFIG,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learned_thresholds': None,
                'learn_thresholds': False
            }, CONFIG['model_save_path'])
            if prev_best != float('inf'):
                print(f"  ✓ Saved best model! (val_loss improved from {prev_best:.4f} to {avg_val_loss:.4f}, {improvement:.2f}% improvement)")
            else:
                print(f"  ✓ Saved best model! (val_loss: {avg_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  (No improvement - best val_loss: {best_val_loss:.4f}, patience: {epochs_without_improvement}/{CONFIG['early_stopping_patience']})")
        
        # Stop early if we haven't improved in a while
        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING TRIGGERED")
            print(f"{'='*60}")
            print(f"No improvement for {CONFIG['early_stopping_patience']} epochs.")
            print(f"Stopping training at epoch {epoch+1} (best model was at epoch {epoch+1-epochs_without_improvement})")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total training time: {total_training_time/60:.2f} min ({total_training_time/3600:.2f} hours)")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def optimize_thresholds(model, val_loader, mlb, device, n_trials=50):
    """
    Find the best thresholds by trying different values and seeing what gives the best F1 score.
    We do this separately since thresholds don't get gradients during training.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZING THRESHOLDS ON VALIDATION SET")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting threshold optimization...")
    
    if not model.learn_thresholds:
        print("  Model doesn't use learnable thresholds, skipping optimization.")
        return None
    
    model.eval()
    all_probs = []
    all_labels = []
    
    # First, get all the predictions on the validation set
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Collecting validation predictions...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Collecting", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Optimizing thresholds...")
    
    # Try different threshold values and see which one works best
    best_f1 = 0.0
    best_thresholds = None
    
    # First try using the same threshold for all classes
    print("  Trying uniform thresholds...")
    for threshold in np.linspace(0.1, 0.9, n_trials):
        thresholds = torch.full((all_probs.shape[1],), threshold)
        preds = (all_probs > thresholds.unsqueeze(0)).float()
        
        # Make sure we predict at least one genre per movie
        for i in range(preds.shape[0]):
            if preds[i].sum() == 0:
                max_prob_idx = all_probs[i].argmax().item()
                preds[i, max_prob_idx] = 1.0
        
        # See how good this threshold is
        f1 = f1_score(all_labels.numpy(), preds.numpy(), average='micro', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds = thresholds.clone()
            print(f"    Threshold: {threshold:.3f} -> F1: {f1:.4f}")
    
    # Now try optimizing a different threshold for each class
    print("  Trying per-class thresholds...")
    per_class_thresholds = torch.zeros(all_probs.shape[1])
    
    for class_idx in range(all_probs.shape[1]):
        class_probs = all_probs[:, class_idx]
        class_labels = all_labels[:, class_idx]
        
        best_class_f1 = 0.0
        best_class_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, n_trials):
            preds = (class_probs > threshold).float()
            f1 = f1_score(class_labels.numpy().reshape(-1, 1), 
                        preds.numpy().reshape(-1, 1), 
                        average='micro', zero_division=0)
            
            if f1 > best_class_f1:
                best_class_f1 = f1
                best_class_threshold = threshold
        
        per_class_thresholds[class_idx] = best_class_threshold
    
    # See how well the per-class thresholds work
    preds = (all_probs > per_class_thresholds.unsqueeze(0)).float()
    for i in range(preds.shape[0]):
        if preds[i].sum() == 0:
            max_prob_idx = all_probs[i].argmax().item()
            preds[i, max_prob_idx] = 1.0
    
    per_class_f1 = f1_score(all_labels.numpy(), preds.numpy(), average='micro', zero_division=0)
    
    print(f"\n  Results:")
    print(f"    Uniform threshold F1: {best_f1:.4f} (threshold: {best_thresholds[0]:.3f})")
    print(f"    Per-class threshold F1: {per_class_f1:.4f}")
    
    # Pick whichever approach worked better
    if per_class_f1 > best_f1:
        print(f"  ✓ Using per-class thresholds (F1: {per_class_f1:.4f})")
        optimized_thresholds = per_class_thresholds
    else:
        print(f"  ✓ Using uniform threshold (F1: {best_f1:.4f})")
        optimized_thresholds = best_thresholds
    
    # Update the model's thresholds (need to convert to logits space)
    with torch.no_grad():
        # Convert back to logits: logit = log(threshold / (1 - threshold))
        # Clamp to avoid issues with 0 and 1
        epsilon = 1e-7
        optimized_thresholds = torch.clamp(optimized_thresholds, epsilon, 1 - epsilon)
        threshold_logits = torch.log(optimized_thresholds / (1 - optimized_thresholds))
        model.threshold_logits.data = threshold_logits.to(device)
    
    print(f"  Updated model thresholds:")
    print(f"    Min: {optimized_thresholds.min():.3f}, Max: {optimized_thresholds.max():.3f}, Mean: {optimized_thresholds.mean():.3f}")
    
    return optimized_thresholds.numpy()


def evaluate_model(model, test_loader, mlb):
    """Evaluate the model on test set"""
    
    print("=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting evaluation...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    eval_pbar = tqdm(test_loader, desc="Evaluating", 
                    unit="batch",
                    file=sys.stdout,
                    dynamic_ncols=True,
                    leave=True,
                    ncols=100)
    
    with torch.no_grad():
        # Use a fixed threshold of 0.5
        thresholds = torch.tensor(0.5)
        
        for batch_idx, (images, labels) in enumerate(eval_pbar):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            # Convert probabilities to binary predictions
            preds = (probs > thresholds.unsqueeze(0)).cpu().numpy()
            
            # Make sure we predict at least one genre per movie
            # If nothing is above threshold, just pick the highest probability one
            for i in range(preds.shape[0]):
                if preds[i].sum() == 0:
                    max_prob_idx = probs[i].argmax().cpu().item()
                    preds[i, max_prob_idx] = 1
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            eval_pbar.set_postfix({
                'batch': f'{batch_idx+1}/{len(test_loader)}'
            })
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Calculating metrics...")
    
    # Compute all the evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    print("=" * 60)
    
    return accuracy, f1_macro, f1_micro, hamming


def predict_genre(image_path, model, mlb, threshold=0.5):
    """Predict genre for a single image"""
    
    model.eval()
    
    # Set up the same transforms we used during training
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            probs = torch.sigmoid(output)
            
            # Convert probabilities to binary predictions using the threshold
            preds = (probs > threshold).cpu().numpy()[0]
            
            # Make sure we predict at least one genre
            # If nothing is above threshold, just pick the highest probability one
            if preds.sum() == 0:
                max_prob_idx = probs[0].argmax().cpu().item()
                preds[max_prob_idx] = 1
        
        # Convert the binary predictions back to genre names
        predicted_genres = mlb.inverse_transform([preds])[0]
        probabilities = {mlb.classes_[i]: float(probs[0][i].cpu()) 
                        for i in range(len(mlb.classes_)) if preds[i]}
        
        return predicted_genres, probabilities
    except Exception as e:
        print(f"Error predicting: {e}")
        return [], {}


def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint in the checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # Figure out which checkpoint is the latest by looking at the epoch number
    def get_epoch_num(path):
        try:
            filename = os.path.basename(path)
            epoch_str = filename.replace('checkpoint_epoch_', '').replace('.pth', '')
            return int(epoch_str)
        except:
            return -1
    
    latest_checkpoint = max(checkpoint_files, key=get_epoch_num)
    return latest_checkpoint


def load_checkpoint(checkpoint_path, model, device):
    """Load checkpoint and return resume parameters"""
    print(f"\n[RESUME] Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Model state loaded")
    
    # Get all the info we need to resume training
    saved_epoch = checkpoint.get('epoch', 0)
    start_epoch = saved_epoch  # Start from this epoch (it's 1-indexed in the checkpoint)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
    
    print(f"  Resume info:")
    print(f"    Last completed epoch: {saved_epoch}")
    print(f"    Resuming from epoch: {start_epoch}")
    print(f"    Previous best val loss: {best_val_loss:.4f}")
    print(f"    Epochs without improvement: {epochs_without_improvement}")
    print(f"    Training history: {len(train_losses)} epochs")
    
    # Set up optimizer and scheduler (we'll load their states in a moment)
    if CONFIG['freeze_backbone']:
        params_to_optimize = list(model.model.fc.parameters())
        optimizer = optim.Adam(params_to_optimize, 
                              lr=CONFIG['learning_rate'],
                              weight_decay=CONFIG['weight_decay'])
    else:
        backbone_params = []
        for name, param in model.model.named_parameters():
            if 'fc' not in name and param.requires_grad:
                backbone_params.append(param)
        classifier_params = list(model.model.fc.parameters())
        optimizer_groups = [
            {'params': backbone_params, 'lr': CONFIG['learning_rate'] * 0.1, 'weight_decay': CONFIG['weight_decay']},
            {'params': classifier_params, 'lr': CONFIG['learning_rate'], 'weight_decay': CONFIG['weight_decay']}
        ]
        optimizer = optim.Adam(optimizer_groups)
    
    # Load the optimizer state if we have it
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Optimizer state loaded")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['scheduler_factor'], 
        patience=CONFIG['scheduler_patience'], min_lr=1e-7
    )
    
    # Load the scheduler state if we have it
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  ✓ Scheduler state loaded")
    
    # Load the scaler if we're using mixed precision
    scaler = None
    if CONFIG['use_mixed_precision'] and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"  ✓ Scaler state loaded")
    
    return {
        'start_epoch': start_epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_without_improvement': epochs_without_improvement
    }

def main():
    """Main training function"""
    global device
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a genre prediction model using BCE loss')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from, or "latest" to resume from latest checkpoint')
    args = parser.parse_args()
    
    # Check for CUDA before we do anything else
    print("\n" + "=" * 60)
    print("DEVICE DETECTION")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  CUDA Device Count: {torch.cuda.device_count()}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Current CUDA Device: {torch.cuda.current_device()}")
        print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test that CUDA actually works
        try:
            test_tensor = torch.randn(1, 1).cuda()
            print(f"  ✓ CUDA test successful - GPU is working!")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠ CUDA test failed: {e}")
            print(f"  This may indicate a driver or installation issue.")
    else:
        print("⚠ WARNING: CUDA is not available! Training will use CPU (much slower).")
        print("  Possible reasons:")
        print("  1. PyTorch was installed without CUDA support")
        print("  2. NVIDIA drivers are not installed or outdated")
        print("  3. CUDA toolkit is not properly installed")
        print("  To use GPU, install PyTorch with CUDA support:")
        print("  Visit: https://pytorch.org/get-started/locally/")
        print("  Select your CUDA version (you have CUDA 11.8)")
    
    device = get_device()
    print(f"\nSelected device: {device}")
    
    print("\n" + "=" * 60)
    print("MOVIE POSTER GENRE PREDICTION - CNN TRAINING (BCE Loss)")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Selected Device: {device}")
    if device.type == 'mps':
        print(f"GPU: Apple Silicon (MPS - Metal Performance Shaders)")
        print(f"PyTorch Version: {torch.__version__}")
    print("=" * 60 + "\n")
    
    total_start_time = time.time()
    
    # Load the data
    image_paths, labels, mlb, num_classes = load_data()
    
    # Set up the data loaders
    train_loader, val_loader, test_loader, mlb, y_train = create_data_loaders(image_paths, labels, mlb)
    
    # Create the model
    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing ResNet50-based model...")
    print(f"  Using pre-trained weights: {CONFIG['use_pretrained']}")
    print(f"  Freeze backbone: {CONFIG['freeze_backbone']}")
    if CONFIG.get('freeze_early_layers', False) and not CONFIG['freeze_backbone']:
        print(f"  Freeze early layers: True (Level 1, 2, 3: conv1 + layer1 + layer2 + layer3 - 40 CONV layers frozen)")
    
    # See if we're resuming from a checkpoint
    resume_params = None
    if args.resume:
        if args.resume.lower() == 'latest':
            checkpoint_path = find_latest_checkpoint(CONFIG.get('checkpoint_dir', 'checkpoints'))
            if checkpoint_path is None:
                print("⚠️  No checkpoint found. Starting fresh training.")
                args.resume = None
            else:
                print(f"Found latest checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.resume
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint not found: {checkpoint_path}")
                print("Starting fresh training.")
                args.resume = None
        
        if args.resume:
            # Load the checkpoint to see what config it was using
            temp_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_config = temp_checkpoint.get('config', CONFIG)
            num_classes = temp_checkpoint.get('num_classes', num_classes)
            print(f"\n[RESUME] Checkpoint info:")
            print(f"  Epoch: {temp_checkpoint.get('epoch', 'unknown')}")
            print(f"  Config from checkpoint will be used")
    
    # Create the model
    model = GenreCNN(num_classes, 
                     pretrained=CONFIG['use_pretrained'],
                     freeze_backbone=CONFIG['freeze_backbone'],
                     freeze_early_layers=CONFIG.get('freeze_early_layers', False),
                     learn_thresholds=CONFIG.get('learn_thresholds', False))
    
    # Move the model to GPU if we have one
    if device.type == 'cuda':
        print(f"  Moving model to GPU: {device}")
        model = model.to(device)
        # Make sure everything actually got moved to GPU
        all_on_gpu = all(p.device.type == 'cuda' for p in model.parameters())
        if not all_on_gpu:
            print(f"  ⚠ WARNING: Some parameters not on GPU! Forcing move...")
            model = model.cuda()
        print(f"  ✓ Model successfully moved to GPU")
    else:
        model = model.to(device)
        print(f"  Model on device: {device}")
    
    # Double-check the model is on the right device
    next_param_device = next(model.parameters()).device
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model device verification:")
    print(f"  Expected device: {device}")
    print(f"  Model parameters device: {next_param_device}")
    if device.type == 'cuda':
        if next_param_device.type == 'cuda':
            print(f"  ✓ Model is correctly on GPU!")
            print(f"  GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            print(f"  ⚠ ERROR: Model should be on GPU but is on {next_param_device}")
            print(f"  Attempting to force move to GPU...")
            model = model.cuda()
            print(f"  After force move: {next(model.parameters()).device}")
    
    # Count up the parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model created successfully:")
    print(f"  Output classes: {num_classes}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    if frozen_params > 0:
        print(f"  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Load the checkpoint if we're resuming
    if args.resume:
        resume_params = load_checkpoint(checkpoint_path, model, device)
    
    # Start training
    if resume_params:
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, mlb, y_train,
            start_epoch=resume_params['start_epoch'],
            optimizer=resume_params['optimizer'],
            scheduler=resume_params['scheduler'],
            scaler=resume_params['scaler'],
            train_losses=resume_params['train_losses'],
            val_losses=resume_params['val_losses'],
            best_val_loss=resume_params['best_val_loss'],
            epochs_without_improvement=resume_params['epochs_without_improvement']
        )
    else:
        train_losses, val_losses = train_model(model, train_loader, val_loader, mlb, y_train)
    
    # Load the best model we saved during training
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading best model from {CONFIG['model_save_path']}...")
    # Need weights_only=False because we saved sklearn objects (MultiLabelBinarizer) in there
    # It's safe since this is our own model file
    checkpoint = torch.load(CONFIG['model_save_path'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best model loaded (from epoch {checkpoint.get('epoch', 'unknown')})")
    
    # We're using a fixed threshold of 0.5 (not learning it)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using fixed threshold: {CONFIG.get('fixed_threshold', 0.5)}")
    print(f"  Threshold optimization skipped (fixed threshold mode)")
    
    # Run evaluation on the test set
    evaluate_model(model, test_loader, mlb)
    
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time/60:.2f} min ({total_time/3600:.2f} hours)")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    # Fix for Windows multiprocessing issues
    import multiprocessing
    if sys.platform == 'win32':
        multiprocessing.freeze_support()
    main()

