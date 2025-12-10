import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import ast
import os
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
from tqdm import tqdm

# I got help from GenAI in coding

# Import the model architecture from training script
# Handle both relative import (when used as module) and absolute import (when run directly)
try:
    # Try relative import first (when imported as Training.evaluate_model)
    from .train import GenreCNN, CONFIG
except ImportError:
    # Fallback for when running as script or when relative import fails
    # Add current directory to path so we can import train.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from train import GenreCNN, CONFIG


def parse_genres(genre_str):
    """Parse genre string to list"""
    try:
        if pd.isna(genre_str) or genre_str == '':
            return []
        # Try to evaluate as Python literal
        genres = ast.literal_eval(genre_str)
        if isinstance(genres, list):
            return genres
        elif isinstance(genres, str):
            return [genres]
        return []
    except:
        return []


def f2_score(y_true, y_pred):
    """Calculate F0.5 score (F-beta with beta=0.5, emphasizes precision over recall)"""
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    if precision + recall == 0:
        return 0.0
    # F0.5: beta=0.5 gives more weight to precision than recall
    # Formula: (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    # With beta=0.5: (1 + 0.25) * (precision * recall) / (0.25 * precision + recall)
    beta = 0.5
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f2


def load_model(model_path='genre_predictor_model_BCE.pth'):
    """Load trained model and MultiLabelBinarizer"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    # Load checkpoint with weights_only=False to allow sklearn objects (MultiLabelBinarizer)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check if model uses learnable thresholds
    learn_thresholds = checkpoint.get('learn_thresholds', False)
    
    # Create model (pretrained=False since we're loading from checkpoint)
    model = GenreCNN(checkpoint['num_classes'], pretrained=False, learn_thresholds=learn_thresholds)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    mlb = checkpoint['mlb']
    config = checkpoint.get('config', CONFIG)
    
    # Get learned thresholds if available
    learned_thresholds = checkpoint.get('learned_thresholds', None)
    if learned_thresholds is not None:
        learned_thresholds = torch.tensor(learned_thresholds)
        print(f"✓ Loaded learned thresholds: min={learned_thresholds.min():.3f}, max={learned_thresholds.max():.3f}, mean={learned_thresholds.mean():.3f}")
    
    return model, mlb, config, learned_thresholds


def predict_image(image_path, model, mlb, config, device, threshold=0.5, learned_thresholds=None):
    """Predict genre for a single image"""
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        if not os.path.exists(image_path):
            return [], None, None
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            
            # Use learned thresholds if available, otherwise use provided threshold
            if learned_thresholds is not None:
                thresholds_tensor = learned_thresholds.to(device)
                preds = (probs > thresholds_tensor.unsqueeze(0)).cpu().numpy()
            else:
                # Fallback: use model's learned thresholds if available
                if model.learn_thresholds:
                    thresholds = model.get_thresholds()
                    preds = (probs > thresholds.unsqueeze(0)).cpu().numpy()
                else:
                    preds = (probs > threshold).cpu().numpy()
            
            # Ensure at least one genre is predicted
            # If no predictions above threshold, select the highest probability genre
            if preds.sum() == 0:
                # Find the index with the highest probability
                max_prob_idx = probs[0].argmax().cpu().item()
                preds[0, max_prob_idx] = 1
        
        # Get predicted genres - ensure preds is 2D array for inverse_transform
        predicted_genres = mlb.inverse_transform(preds)[0]
        prob_values = probs[0].cpu().numpy()
        
        return predicted_genres, prob_values, preds[0]
    except Exception as e:
        # Return None to skip this image, but don't print error for every failed image
        return [], None, None


def load_ground_truth(csv_file, image_dirs):
    """Load ground truth genres from CSV, checking multiple directories"""
    
    print("Loading ground truth data...")
    # Handle both single directory (string) and multiple directories (list)
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    
    print(f"Checking directories: {', '.join(image_dirs)}")
    df = pd.read_csv(csv_file, header=None, low_memory=False)
    
    poster_ids = df[0].astype(str).tolist()
    genre_strings = df[3].astype(str).tolist()
    
    ground_truth = {}
    valid_images = []
    
    def find_image_path(poster_id):
        """Find image in any of the configured directories"""
        for image_dir in image_dirs:
            image_path = os.path.join(image_dir, f"{poster_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        return None
    
    for poster_id, genre_str in zip(poster_ids, genre_strings):
        genres = parse_genres(genre_str)
        if len(genres) > 0:
            image_path = find_image_path(poster_id)
            if image_path:
                ground_truth[poster_id] = genres
                valid_images.append(poster_id)
    
    print(f"Found {len(valid_images)} images with ground truth genres")
    return ground_truth, valid_images


def evaluate_model(model, mlb, config, ground_truth, image_ids, image_dirs, device, threshold=0.5, learned_thresholds=None):
    """Evaluate model on dataset and calculate metrics"""
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    print(f"\nEvaluating model on {len(image_ids)} images...")
    if learned_thresholds is not None:
        print(f"Using learned thresholds: min={learned_thresholds.min():.3f}, max={learned_thresholds.max():.3f}, mean={learned_thresholds.mean():.3f}")
    elif model.learn_thresholds:
        thresholds = model.get_thresholds()
        print(f"Using model's learned thresholds: min={thresholds.min():.3f}, max={thresholds.max():.3f}, mean={thresholds.mean():.3f}")
    else:
        print(f"Using fixed threshold: {threshold}")
    print("=" * 80)
    
    all_predictions = []
    all_ground_truth = []
    individual_results = []
    failed_images = 0
    
    # Handle both single directory (string) and multiple directories (list)
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    
    def find_image_path(image_id):
        """Find image in any of the configured directories"""
        for image_dir in image_dirs:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        return None
    
    # Process each image
    for image_id in tqdm(image_ids, desc="Evaluating images", file=sys.stdout, dynamic_ncols=True):
        image_path = find_image_path(image_id)
        if image_path is None:
            failed_images += 1
            continue
        
        # Predict
        predicted_genres, prob_values, pred_binary = predict_image(
            image_path, model, mlb, config, device, threshold, learned_thresholds
        )
        
        if pred_binary is None:
            failed_images += 1
            continue
        
        # Get ground truth
        true_genres = ground_truth[image_id]
        
        # Encode ground truth
        true_binary = mlb.transform([true_genres])[0]
        
        # Store for overall metrics
        all_predictions.append(pred_binary)
        all_ground_truth.append(true_binary)
        
        # Calculate per-image metrics
        img_accuracy = accuracy_score([true_binary], [pred_binary])
        img_precision = precision_score([true_binary], [pred_binary], average='micro', zero_division=0)
        img_recall = recall_score([true_binary], [pred_binary], average='micro', zero_division=0)
        img_f1 = f1_score([true_binary], [pred_binary], average='micro', zero_division=0)
        img_f2 = f2_score([true_binary], [pred_binary])
        
        # Store individual result
        individual_results.append({
            'image_id': image_id,
            'image_path': image_path,
            'true_genres': true_genres,
            'predicted_genres': list(predicted_genres),
            'accuracy': img_accuracy,
            'precision': img_precision,
            'recall': img_recall,
            'f1': img_f1,
            'f2': img_f2
        })
    
    if len(all_predictions) == 0:
        print(f"\nWARNING: No successful predictions! {failed_images} images failed.")
        print("This might indicate an issue with the model or image loading.")
        return [], {
            'accuracy': 0.0,
            'precision_micro': 0.0,
            'recall_micro': 0.0,
            'f1_micro': 0.0,
            'f2': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'num_images': 0
        }
    
    if failed_images > 0:
        print(f"\nNote: {failed_images} images failed to process and were skipped.")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
    overall_precision = precision_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_recall = recall_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_f1 = f1_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_f2 = f2_score(all_ground_truth, all_predictions)
    
    # Also calculate macro averages
    overall_precision_macro = precision_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    overall_recall_macro = recall_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    overall_f1_macro = f1_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    
    return individual_results, {
        'accuracy': overall_accuracy,
        'precision_micro': overall_precision,
        'recall_micro': overall_recall,
        'f1_micro': overall_f1,
        'f2': overall_f2,
        'precision_macro': overall_precision_macro,
        'recall_macro': overall_recall_macro,
        'f1_macro': overall_f1_macro,
        'num_images': len(individual_results)
    }


def print_individual_results(individual_results, max_display=20):
    """Print individual image predictions and performance"""
    
    print("\n" + "=" * 80)
    print("INDIVIDUAL IMAGE PREDICTIONS")
    print("=" * 80)
    
    # Sort by F1 score (descending)
    sorted_results = sorted(individual_results, key=lambda x: x['f1'], reverse=True)
    
    # Display top results
    print(f"\nTop {min(max_display, len(sorted_results))} predictions (sorted by F1 score):")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:max_display], 1):
        print(f"\n[{i}] Image ID: {result['image_id']}")
        print(f"    True Genres:     {', '.join(result['true_genres']) if result['true_genres'] else 'None'}")
        print(f"    Predicted Genres: {', '.join(result['predicted_genres']) if result['predicted_genres'] else 'None'}")
        print(f"    Metrics:")
        print(f"      Accuracy:  {result['accuracy']:.4f}")
        print(f"      Precision: {result['precision']:.4f}")
        print(f"      Recall:    {result['recall']:.4f}")
        print(f"      F1 Score:  {result['f1']:.4f}")
        print(f"      F0.5 Score (precision-weighted):  {result['f2']:.4f}")
    
    if len(sorted_results) > max_display:
        print(f"\n... and {len(sorted_results) - max_display} more images")
    
    # Show worst predictions
    print(f"\n\nWorst {min(5, len(sorted_results))} predictions:")
    print("-" * 80)
    for i, result in enumerate(sorted_results[-5:], 1):
        print(f"\n[{i}] Image ID: {result['image_id']}")
        print(f"    True Genres:     {', '.join(result['true_genres']) if result['true_genres'] else 'None'}")
        print(f"    Predicted Genres: {', '.join(result['predicted_genres']) if result['predicted_genres'] else 'None'}")
        print(f"    F1 Score: {result['f1']:.4f}")


def print_overall_metrics(metrics):
    """Print overall performance metrics"""
    
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\nNumber of images evaluated: {metrics['num_images']}")
    print("\nMicro-averaged metrics (treats each label independently):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision_micro']:.4f} ({metrics['precision_micro']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall_micro']:.4f} ({metrics['recall_micro']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_micro']:.4f} ({metrics['f1_micro']*100:.2f}%)")
    print(f"  F0.5 Score (precision-weighted):  {metrics['f2']:.4f} ({metrics['f2']*100:.2f}%)")
    
    print("\nMacro-averaged metrics (averages across all labels):")
    print(f"  Precision: {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    print("=" * 80)


def evaluate_with_threshold(model, mlb, config, ground_truth, image_ids, image_dirs, device, threshold, learned_thresholds=None):
    """Evaluate model with a specific threshold and return only metrics (faster for multiple thresholds)"""
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    failed_images = 0
    
    # Handle both single directory (string) and multiple directories (list)
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    
    def find_image_path(image_id):
        """Find image in any of the configured directories"""
        for image_dir in image_dirs:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        return None
    
    # Process each image
    for image_id in image_ids:
        image_path = find_image_path(image_id)
        if image_path is None:
            failed_images += 1
            continue
        
        # Predict
        predicted_genres, prob_values, pred_binary = predict_image(
            image_path, model, mlb, config, device, threshold, learned_thresholds
        )
        
        if pred_binary is None:
            failed_images += 1
            continue
        
        # Get ground truth
        true_genres = ground_truth[image_id]
        
        # Encode ground truth
        true_binary = mlb.transform([true_genres])[0]
        
        # Store for overall metrics
        all_predictions.append(pred_binary)
        all_ground_truth.append(true_binary)
    
    if len(all_predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision_micro': 0.0,
            'recall_micro': 0.0,
            'f1_micro': 0.0,
            'f2': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'num_images': 0
        }
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
    overall_precision = precision_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_recall = recall_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_f1 = f1_score(all_ground_truth, all_predictions, average='micro', zero_division=0)
    overall_f2 = f2_score(all_ground_truth, all_predictions)
    
    # Also calculate macro averages
    overall_precision_macro = precision_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    overall_recall_macro = recall_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    overall_f1_macro = f1_score(all_ground_truth, all_predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': overall_accuracy,
        'precision_micro': overall_precision,
        'recall_micro': overall_recall,
        'f1_micro': overall_f1,
        'f2': overall_f2,
        'precision_macro': overall_precision_macro,
        'recall_macro': overall_recall_macro,
        'f1_macro': overall_f1_macro,
        'num_images': len(all_predictions)
    }


def evaluate_multiple_thresholds(model, mlb, config, ground_truth, image_ids, image_dirs, device, thresholds, learned_thresholds=None):
    """Evaluate model with multiple threshold values and compare results"""
    
    print("\n" + "=" * 80)
    print("EVALUATING WITH MULTIPLE THRESHOLDS")
    print("=" * 80)
    print(f"Testing {len(thresholds)} threshold values: {thresholds}")
    print("=" * 80)
    
    results = []
    
    for threshold in tqdm(thresholds, desc="Evaluating thresholds", file=sys.stdout, dynamic_ncols=True):
        metrics = evaluate_with_threshold(
            model, mlb, config, ground_truth, image_ids, 
            image_dirs, device, threshold, learned_thresholds
        )
        results.append({
            'threshold': threshold,
            **metrics
        })
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 (Micro)':<12} {'F1 (Macro)':<12} {'F0.5':<12}")
    print("-" * 80)
    
    best_f1_micro = -1
    best_f1_macro = -1
    best_threshold_micro = None
    best_threshold_macro = None
    
    for result in results:
        threshold = result['threshold']
        acc = result['accuracy']
        prec = result['precision_micro']
        rec = result['recall_micro']
        f1_micro = result['f1_micro']
        f1_macro = result['f1_macro']
        f2 = result['f2']
        
        print(f"{threshold:<12.3f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1_micro:<12.4f} {f1_macro:<12.4f} {f2:<12.4f}")
        
        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro
            best_threshold_micro = threshold
        
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold_macro = threshold
    
    print("=" * 80)
    print(f"\nBest F1 (Micro): {best_f1_micro:.4f} at threshold {best_threshold_micro:.3f}")
    print(f"Best F1 (Macro): {best_f1_macro:.4f} at threshold {best_threshold_macro:.3f}")
    print("=" * 80)
    
    return results, best_threshold_micro, best_threshold_macro


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on dataset and calculate metrics')
    parser.add_argument('--model', type=str, default='genre_predictor_model_BCE.pth',
                       help='Path to trained model file')
    parser.add_argument('--csv', type=str, default='Training/meta_processed.csv',
                       help='Path to CSV file with ground truth')
    parser.add_argument('--image_dir', type=str, nargs='+', default=['cleaned_posters_local', 'Data/cleaned_posters'],
                       help='Path(s) to directory(ies) containing images (can specify multiple)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Test single threshold value (default: None, will test multiple thresholds). '
                            'Use this to override default multi-threshold testing.')
    parser.add_argument('--test_thresholds', type=float, nargs='+', default=None,
                       help='Test multiple threshold values (e.g., --test_thresholds 0.3 0.4 0.5 0.6 0.7). '
                            'If specified, will evaluate with all thresholds and show comparison table.')
    parser.add_argument('--threshold_range', type=str, default=None,
                       help='Test a range of thresholds (e.g., "0.2:0.8:0.1" for start:end:step). '
                            'Alternative to --test_thresholds for testing many values.')
    parser.add_argument('--max_images', type=int, default=1000,
                       help='Maximum number of images to evaluate (default: 1000, use 0 or None for all)')
    parser.add_argument('--random_sample', action='store_true',
                       help='Randomly sample images instead of taking first N (more representative)')
    parser.add_argument('--max_display', type=int, default=20,
                       help='Maximum number of individual results to display (default: 20)')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("=" * 80)
    print("MODEL EVALUATION ON DATASET")
    print("=" * 80)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    elif device.type == 'mps':
        print(f"GPU: Apple Silicon (MPS)")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model, mlb, config, learned_thresholds = load_model(args.model)
    print(f"Model loaded successfully!")
    print(f"Number of genre classes: {len(mlb.classes_)}")
    
    # Load ground truth
    ground_truth, image_ids = load_ground_truth(args.csv, args.image_dir)
    
    # Limit number of images if specified
    total_images = len(image_ids)
    if args.max_images and args.max_images > 0:
        if args.random_sample:
            # Randomly sample images for more representative evaluation
            import random
            random.seed(42)  # For reproducibility
            if args.max_images < total_images:
                image_ids = random.sample(image_ids, args.max_images)
                print(f"\nRandomly sampling {len(image_ids)} images from {total_images} total images")
            else:
                print(f"\nMax images ({args.max_images}) >= total images ({total_images}), using all images")
        else:
            # Take first N images
            image_ids = image_ids[:args.max_images]
            print(f"\nLimiting evaluation to first {len(image_ids)} images (out of {total_images} total)")
    else:
        print(f"\nEvaluating on all {total_images} images")
    
    # Determine if we should test multiple thresholds
    # Default: test multiple thresholds if not explicitly disabled
    test_multiple = False
    thresholds_to_test = []
    
    if args.test_thresholds:
        test_multiple = True
        thresholds_to_test = args.test_thresholds
        print(f"\nWill test {len(thresholds_to_test)} threshold values: {thresholds_to_test}")
    elif args.threshold_range:
        test_multiple = True
        # Parse range: "start:end:step"
        try:
            parts = args.threshold_range.split(':')
            start = float(parts[0])
            end = float(parts[1])
            step = float(parts[2]) if len(parts) > 2 else 0.1
            thresholds_to_test = np.arange(start, end + step/2, step).tolist()
            print(f"\nWill test threshold range: {start} to {end} with step {step} ({len(thresholds_to_test)} values)")
        except Exception as e:
            print(f"Error parsing threshold range: {e}")
            print("Expected format: start:end:step (e.g., '0.2:0.8:0.1')")
            sys.exit(1)
    elif args.threshold is not None:
        # Single threshold specified - use single evaluation
        test_multiple = False
        threshold_to_use = args.threshold
        print(f"\nSingle threshold specified: {threshold_to_use}")
    else:
        # Default: test multiple thresholds automatically
        test_multiple = True
        thresholds_to_test = [0.3, 0.5, 0.7]
        print(f"\nNo threshold specified - will test multiple thresholds by default: {thresholds_to_test}")
        print(f"  (Use --threshold <value> to test single threshold, or --test_thresholds to specify custom values)")
    
    # If testing multiple thresholds, use the comparison function
    if test_multiple:
        # Note: When testing multiple thresholds, we ignore learned thresholds for comparison
        # (learned thresholds are per-class, not uniform)
        print("\nNote: When testing multiple uniform thresholds, learned per-class thresholds are ignored.")
        print("This allows fair comparison of uniform threshold values.\n")
        
        results, best_threshold_micro, best_threshold_macro = evaluate_multiple_thresholds(
            model, mlb, config, ground_truth, image_ids, 
            args.image_dir, device, thresholds_to_test, learned_thresholds=None
        )
        
        # Test multiple uniform thresholds
        print("\nTesting multiple uniform threshold values for comparison.\n")
        
        results, best_threshold_micro, best_threshold_macro = evaluate_multiple_thresholds(
            model, mlb, config, ground_truth, image_ids, 
            args.image_dir, device, thresholds_to_test, learned_thresholds=None
        )
    else:
        # Single threshold evaluation (original behavior)
        # Note: Model uses fixed threshold (not learnable)
        print(f"\nUsing fixed threshold: {threshold_to_use}")
        
        # Evaluate model
        individual_results, overall_metrics = evaluate_model(
            model, mlb, config, ground_truth, image_ids, 
            args.image_dir, device, threshold_to_use, learned_thresholds=None
        )
        
        # Print results
        print_individual_results(individual_results, args.max_display)
        print_overall_metrics(overall_metrics)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

