# Movie Poster Genre Prediction

A deep learning project that predicts movie genres from poster images using a ResNet50-based CNN model with Binary Cross-Entropy (BCE) loss for multi-label classification.

## Overview

This project trains a convolutional neural network to predict multiple movie genres from poster images. The model uses transfer learning with a pre-trained ResNet50 backbone and is trained using Binary Cross-Entropy loss for multi-label classification.

Unlike other movie genre predictor models, this model was trained exclusively on posters **without titles**. We removed all text from the movie posters using OCR and in-painting techniques, ensuring the model learns to identify genres based purely on visual features rather than relying on textual cues. This approach eliminates the bias that comes from reading movie titles and forces the model to extract meaningful visual patterns from the poster artwork itself.

> **Model Performance:** The model achieves **59.98% precision**, correctly predicting genres **59.98% of the time** across **28 different genre classes**. This performance is **15.8 times better** than a random baseline guess (which would achieve approximately 3.57% precision), demonstrating substantial learning capability and successful extraction of meaningful visual features from movie posters.

## Performance of the Model

### Threshold Optimization

We tested multiple thresholds to balance Precision and Recall. The model performance varies significantly with different threshold values:

| Threshold | Accuracy   | Precision  | Recall     | F1 (Micro) | F1 (Macro) | F0.5       |
| :-------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- |
| 0.3       | 12.53%     | 51.06%     | 43.94%     | 0.4723     | 0.1010     | 0.4945     |
| 0.5       | 17.69%     | 59.05%     | 33.43%     | 0.4269     | 0.0670     | 0.5120     |
| **0.7**   | **17.91%** | **59.98%** | **30.12%** | **0.4010** | **0.0541** | **0.5005** |

**Note:** The model uses a **default threshold of 0.7** for predictions, which provides the highest precision (59.98%) and accuracy (17.91%) while maintaining good performance. This threshold can be adjusted when using the prediction script.

### Model Performance vs. Random Baseline

The model predicts across **28 different genre classes**. For comparison, a random baseline that guesses genres uniformly would achieve approximately **3.57% precision** (1/28 ≈ 0.0357).

Our model significantly outperforms this baseline:

- **Threshold 0.3:** 51.06% precision → **13.3x better** than random (1,330% improvement)
- **Threshold 0.5:** 59.05% precision → **15.5x better** than random (1,554% improvement)
- **Threshold 0.7:** 59.98% precision → **15.8x better** than random (1,580% improvement)

The model demonstrates substantial learning capability, achieving precision scores that are over **16 times better** than random guessing, indicating successful extraction of meaningful visual features from movie posters for genre classification.

### Key Metrics

- **Validation Loss:** 0.163
- **Micro F1:** 0.4376
- **Precision weighted (F0.5):** 55.40%

## Project Structure

```
.
├── Training/
│   ├── train.py                     # Training script
│   ├── evaluate_model.py           # Evaluation script
│   ├── meta_processed.csv          # Movie metadata CSV file
│   └── requirements.txt            # Full dependencies for training/evaluation
├── predict.py                       # Prediction script for new images
├── genre_predictor_model_BCE.pth   # Trained model checkpoint
├── requirements.txt                # Minimal dependencies for inference only
└── README.md                       # This file
```

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended for training) or CPU
- Sufficient disk space for the dataset

## Installation

1. **Download this working directory**

2. **Install Python dependencies:**

   **For training/evaluation (full dependencies):**

   ```bash
   pip install -r Training/requirements.txt
   ```

   **For inference only (minimal dependencies):**

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The root `requirements.txt` contains minimal dependencies needed to run `predict.py` for inference. If you plan to train or evaluate the model, use `Training/requirements.txt` instead, which includes additional dependencies like `pandas` for data processing.

## Using the Trained Model

The pre-trained model (`genre_predictor_model_BCE.pth`) is ready to use for predicting genres from movie poster images.

### Predicting Genres for New Images

Use the `predict.py` script to predict genres for a single image or a batch of images:

**Predict a single image:**

```bash
python predict.py --image path/to/poster.jpg
```

**Example execution:**

```bash
$ python predict.py --image ./Data/cleaned_posters/661.jpg
Loading model...
✓ Model loaded successfully
  Number of genre classes: 28
  Using threshold: 0.7
  Using device: mps

Predicting genres for: ./Data/cleaned_posters/661.jpg

================================================================================
PREDICTION RESULTS
================================================================================

[1] 661.jpg
    Path: ./Data/cleaned_posters/661.jpg
    Predicted Genres: Drama
    Threshold: 0.7

================================================================================
```

**Predict all images in a directory:**

```bash
python predict.py --directory path/to/posters/
```

**Use a custom threshold:**

```bash
python predict.py --image poster.jpg --threshold 0.3
```

**Save results to a file:**

```bash
# Save as JSON
python predict.py --directory posters/ --output results.json

# Save as CSV
python predict.py --directory posters/ --output results.csv --format csv
```

**Show detailed probabilities:**

```bash
python predict.py --image poster.jpg --verbose
```

**Available options:**

- `--image` / `-i`: Path to a single image file
- `--directory` / `-d`: Path to directory containing images
- `--model` / `-m`: Path to trained model file (default: `genre_predictor_model_BCE.pth`)
- `--threshold` / `-t`: Threshold for genre prediction (default: **0.7**)
- `--output` / `-o`: Output file path (JSON or CSV)
- `--format` / `-f`: Output format when saving to file (`json` or `csv`, default: `json`)
- `--verbose` / `-v`: Show detailed probabilities for each genre

**Note:** By default, the prediction script uses a threshold of **0.7**, which provides the highest precision (60.00%) as shown in the performance metrics above. You can override this with the `--threshold` option if needed.

## Training the Model

Note that the best model is already trained and saved as `genre_predictor_model_BCE.pth`. However, if you want to train the model yourself:

### Quick Start

```bash
python Training/train.py
```

### Training Configuration

- Batch size: 64
- Learning rate: 0.0015
- Epochs: 30 (with early stopping)
- Image size: 224x224
- Train/Val/Test split: 80%/10%/10%

### Download the Dataset

To train the model, you need to download and organize the movie poster images dataset. The dataset consists of **90,000 high-quality cleaned movie poster images** that have been processed to remove text (titles and metadata) using OCR and in-painting techniques.

#### Dataset Requirements

1. **Movie Poster Images:**

   - Download the cleaned poster images dataset
   - The dataset should contain approximately **90,000 images**
   - Each image should be named as `{poster_id}.jpg` (e.g., `1.jpg`, `2.jpg`, `661.jpg`)
   - Supported image formats: JPG, JPEG, PNG

2. **Directory Structure:**

   - Place the images in one of the following directories:
     - `cleaned_posters_local/` (preferred location)
     - `Data/cleaned_posters/` (alternative location)
   - The training script will automatically check both directories and use whichever contains the images

3. **Metadata File:**
   - The `meta_processed.csv` file is already included in the `Training/` directory
   - This CSV file contains the genre labels for each poster ID
   - Format:
     - Column 0: Poster ID (used as image filename)
     - Column 3: Genre labels (as a Python list string, e.g., `"['Action', 'Drama']"`)

#### Dataset Organization Example

```
genre_predictor/
├── cleaned_posters_local/     # Preferred location
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── ...
├── Data/
│   └── cleaned_posters/       # Alternative location
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
└── Training/
    └── meta_processed.csv     # Already included
```

#### Verification

After downloading and organizing the dataset, verify the setup:

```bash
# Check if images are in the correct location
ls cleaned_posters_local/ | head -10    # Should show image files
# or
ls Data/cleaned_posters/ | head -10     # Alternative location

# Verify CSV file exists
ls Training/meta_processed.csv          # Should exist
```

**Note:** The training script will automatically detect which directory contains your images and report how many valid images it found during the data loading phase.

### Training Process

The training process follows these steps:

1. **Data Loading:** The script loads movie poster images from local image folders (`cleaned_posters_local/` or `Data/cleaned_posters/`) and their corresponding genre labels from `meta_processed.csv`. Images are resized to 224x224 pixels and normalized.

2. **Data Splitting:** The dataset is split into training (80%), validation (10%), and test (10%) sets.

3. **Model Initialization:** A ResNet50 backbone pre-trained on ImageNet is loaded. The first 10 layers (Conv1) are frozen to retain low-level feature extraction capabilities, while the rest of the network is fine-tuned.

4. **Training Loop:** The model is trained using Binary Cross-Entropy loss for multi-label classification. Training includes:

   - Forward pass through the network
   - Loss calculation
   - Backpropagation
   - Parameter updates using Adam optimizer
   - Validation after each epoch

5. **Early Stopping:** Training stops early if validation loss doesn't improve for a specified number of epochs to prevent overfitting.

6. **Model Checkpointing:** The best model (based on validation loss) is saved as `genre_predictor_model_BCE.pth`, which includes model weights, MultiLabelBinarizer for label encoding, and training configuration.

7. **Evaluation:** After training, the model can be evaluated using `Training/evaluate_model.py`, which tests multiple thresholds and calculates comprehensive metrics including Accuracy, Precision, Recall, and F1 Scores.

## Data Processing Pipeline

Based on our analysis of **300,000 movie posters**, we implemented a strict cleaning pipeline to ensure the model learns visual features rather than simply "reading" the movie title.

1.  **Dataset Filtration:** Removed invalid image links and corrupted files.

2.  **Text Removal (OCR):** We utilized Optical Character Recognition (OCR) to identify title text and other metadata on the posters.

3.  **In-painting:** Identified bounding boxes of text were filled with the surrounding background color to remove the text visual cues entirely.

4.  **Final Dataset:** After processing, the dataset was consolidated to **90,000 high-quality images**.

## Model Architecture

### Model: ResNet50 (Transfer Learning)

- **Backbone:** ResNet-50 (50 layers) pre-trained on ImageNet.

- **Configuration:** The first 10 layers (Conv1) were **frozen** to retain low-level feature extraction capabilities.

- **Classifier Head:**

  - FC1: 2048 $\rightarrow$ 1024

  - FC2: 1024 $\rightarrow$ 512

  - FC3: 512 $\rightarrow$ 27 (Output Classes)

## Future Improvements

To further improve the model's performance, specifically Recall, we have identified the following areas for development:

- **Advanced Data Augmentation:** Implementing random crops, rotation, and color jitter (brightness/contrast/saturation) to force the model to learn more complex features.

- **Regularization:** Tuning Dropout rates to mitigate overfitting.

- **Handling Class Imbalance:** Utilizing weighted loss functions or oversampling for underrepresented genres.

## Dataset Setup

The project expects:

1. **CSV File (`meta_processed.csv`):**

   - Column 0: Poster ID (used as image filename)
   - Column 3: Genre labels (as a Python list string, e.g., `"['Action', 'Drama']"`)

2. **Image Directories:**

   - Place poster images in either `cleaned_posters_local/` or `Data/cleaned_posters/` directory.
   - Images must be named as `{poster_id}.jpg` (e.g., `1.jpg`, `2.jpg`, etc.)
   - Supported formats: JPG, JPEG, PNG

**Note:** The script will automatically check both directories and use whichever contains the images.

### Evaluating the Model

**Evaluation (tests multiple thresholds):**

```bash
python Training/evaluate_model.py
```

This script will run the evaluation metrics discussed in the "Performance of the Model" section, calculating Accuracy, Precision, Recall, and F1 Scores across the test set.

## Output Files

- `genre_predictor_model_BCE.pth`: Best model checkpoint (includes model weights, MultiLabelBinarizer, and config).
- `checkpoints/checkpoint_epoch_*.pth`: Epoch checkpoints.
