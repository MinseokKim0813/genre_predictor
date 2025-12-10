#!/usr/bin/env python3
"""
Predict movie genres from poster images using the trained model.

This script allows you to predict genres for a single image or a batch of images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to ensure Training imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the model architecture and utilities from training script
from Training.train import GenreCNN, CONFIG, get_device
from Training.evaluate_model import load_model, predict_image


def format_predictions(image_path, predicted_genres, prob_values, mlb, threshold=0.7):
    """Format prediction results for display"""
    
    # Create a dictionary of genre probabilities (only for predicted genres)
    genre_probs = {}
    for genre in predicted_genres:
        genre_idx = list(mlb.classes_).index(genre)
        genre_probs[genre] = float(prob_values[genre_idx])
    
    # Sort by probability (descending)
    genre_probs = dict(sorted(genre_probs.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'image': image_path,
        'predicted_genres': list(predicted_genres),
        'probabilities': genre_probs,
        'threshold': threshold
    }


def predict_single_image(image_path, model, mlb, config, device, threshold=0.7, learned_thresholds=None):
    """Predict genre for a single image and return formatted results"""
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    predicted_genres, prob_values, pred_binary = predict_image(
        image_path, model, mlb, config, device, threshold, learned_thresholds
    )
    
    if pred_binary is None:
        print(f"Error: Failed to process image: {image_path}")
        return None
    
    return format_predictions(image_path, predicted_genres, prob_values, mlb, threshold)


def predict_directory(image_dir, model, mlb, config, device, threshold=0.7, learned_thresholds=None, 
                     extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
    """Predict genres for all images in a directory"""
    
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return []
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
    
    if len(image_files) == 0:
        print(f"No image files found in directory: {image_dir}")
        return []
    
    print(f"Found {len(image_files)} image(s) in directory")
    
    results = []
    failed = 0
    
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        result = predict_single_image(
            str(image_path), model, mlb, config, device, threshold, learned_thresholds
        )
        if result:
            results.append(result)
        else:
            failed += 1
    
    if failed > 0:
        print(f"\nWarning: Failed to process {failed} image(s)")
    
    return results


def print_results(results, verbose=False):
    """Print prediction results in a readable format"""
    
    if not results:
        print("No results to display.")
        return
    
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {os.path.basename(result['image'])}")
        print(f"    Path: {result['image']}")
        print(f"    Predicted Genres: {', '.join(result['predicted_genres'])}")
        
        if verbose:
            print(f"    Probabilities:")
            for genre, prob in result['probabilities'].items():
                print(f"      - {genre}: {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"    Threshold: {result['threshold']}")
    
    print("\n" + "=" * 80)


def save_results_json(results, output_path):
    """Save results to a JSON file"""
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def save_results_csv(results, output_path):
    """Save results to a CSV file"""
    
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Path', 'Predicted Genres', 'Genre Probabilities'])
        
        for result in results:
            genres_str = ', '.join(result['predicted_genres'])
            probs_str = ', '.join([f"{g}:{p:.4f}" for g, p in result['probabilities'].items()])
            writer.writerow([result['image'], genres_str, probs_str])
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Predict movie genres from poster images using the trained model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict a single image
  python predict.py --image path/to/poster.jpg
  
  # Predict all images in a directory
  python predict.py --directory path/to/posters/
  
  # Use a custom threshold
  python predict.py --image poster.jpg --threshold 0.3
  
  # Save results to JSON
  python predict.py --directory posters/ --output results.json
  
  # Save results to CSV
  python predict.py --directory posters/ --output results.csv --format csv
  
  # Verbose output with probabilities
  python predict.py --image poster.jpg --verbose
        """
    )
    
    parser.add_argument('--image', '-i', type=str, help='Path to a single image file')
    parser.add_argument('--directory', '-d', type=str, help='Path to directory containing images')
    parser.add_argument('--model', '-m', type=str, default='genre_predictor_model_BCE.pth',
                       help='Path to trained model file (default: genre_predictor_model_BCE.pth)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='Threshold for genre prediction (default: 0.7)')
    parser.add_argument('--output', '-o', type=str, help='Output file path (JSON or CSV)')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'csv'], default='json',
                       help='Output format when saving to file (default: json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed probabilities for each genre')
    
    args = parser.parse_args()
    
    # Validate arguments - check if input files/directories are provided
    if not args.image and not args.directory:
        print("Error: No input specified. You must provide either an image file or a directory.")
        print("\nPlease specify one of the following:")
        print("  --image, -i    : Path to a single image file")
        print("  --directory, -d: Path to directory containing images")
        print("\nExamples:")
        print("  python predict.py --image path/to/poster.jpg")
        print("  python predict.py --directory path/to/posters/")
        print("\nUse --help for more information.")
        sys.exit(1)
    
    if args.image and args.directory:
        print("Error: Cannot specify both --image and --directory. Please choose one.")
        print("\nUse --help for more information.")
        sys.exit(1)
    
    # Validate that the provided input file/directory exists
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            print("Please check that the file path is correct.")
            sys.exit(1)
        if not os.path.isfile(args.image):
            print(f"Error: Path is not a file: {args.image}")
            print("Please provide a valid image file path.")
            sys.exit(1)
    
    if args.directory:
        if not os.path.exists(args.directory):
            print(f"Error: Directory not found: {args.directory}")
            print("Please check that the directory path is correct.")
            sys.exit(1)
        if not os.path.isdir(args.directory):
            print(f"Error: Path is not a directory: {args.directory}")
            print("Please provide a valid directory path.")
            sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please make sure the model file exists or specify a different path with --model")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    try:
        model, mlb, config, learned_thresholds = load_model(args.model)
        print(f"✓ Model loaded successfully")
        print(f"  Number of genre classes: {len(mlb.classes_)}")
        if learned_thresholds is not None:
            print(f"  Using learned thresholds (will override --threshold)")
        else:
            print(f"  Using threshold: {args.threshold}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get device
    device = get_device()
    model = model.to(device)
    print(f"  Using device: {device}")
    
    # Make predictions
    results = []
    
    if args.image:
        print(f"\nPredicting genres for: {args.image}")
        result = predict_single_image(
            args.image, model, mlb, config, device, args.threshold, learned_thresholds
        )
        if result:
            results.append(result)
        else:
            sys.exit(1)
    
    elif args.directory:
        print(f"\nPredicting genres for images in: {args.directory}")
        results = predict_directory(
            args.directory, model, mlb, config, device, args.threshold, learned_thresholds
        )
        if not results:
            sys.exit(1)
    
    # Display results
    print_results(results, verbose=args.verbose)
    
    # Save results if output path specified
    if args.output:
        output_format = args.format
        # Auto-detect format from extension if not specified
        if not output_format:
            ext = os.path.splitext(args.output)[1].lower()
            if ext == '.csv':
                output_format = 'csv'
            else:
                output_format = 'json'
        
        if output_format == 'json':
            save_results_json(results, args.output)
        else:
            save_results_csv(results, args.output)


if __name__ == '__main__':
    main()

