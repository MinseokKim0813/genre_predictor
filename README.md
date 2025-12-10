# Movie Poster Genre Prediction

A deep learning project that predicts movie genres from poster images using a ResNet50-based CNN model with Binary Cross-Entropy (BCE) loss for multi-label classification.

## Overview

This project trains a convolutional neural network to predict multiple movie genres from poster images. The model uses transfer learning with a pre-trained ResNet50 backbone and is trained using Binary Cross-Entropy loss for multi-label classification. The project explores various architectures, including custom CNNs and Transfer Learning, to optimize Precision and Recall.

## Data Processing Pipeline

[cite_start]Based on our analysis of **300,000 movie posters**[cite: 6], we implemented a strict cleaning pipeline to ensure the model learns visual features rather than simply "reading" the movie title.

1.  **Dataset Filtration:** Removed invalid image links and corrupted files.

2.  [cite_start]**Text Removal (OCR):** We utilized Optical Character Recognition (OCR) to identify title text and other metadata on the posters[cite: 8].

3.  [cite_start]**In-painting:** Identified bounding boxes of text were filled with the surrounding background color to remove the text visual cues entirely[cite: 10].

4.  [cite_start]**Final Dataset:** After processing, the dataset was consolidated to **90,000 high-quality images**[cite: 12].

## Model Evolution & Experiments

We experimented with three distinct architectures to achieve optimal performance.

### Model 1: Custom CNN (Baseline)

- [cite_start]**Architecture:** 5 Convolutional Blocks (Conv + ReLU + MaxPool) followed by 2 Fully Connected layers[cite: 22].

- [cite_start]**Filters:** scaled from 64 to 512 channels[cite: 36].

- **Results:**

  - [cite_start]High Precision (70.24%) but extremely low Recall (7.12%)[cite: 46].

  - [cite_start]F1 Score: ~12.9%[cite: 46].

  - _Conclusion:_ The model struggled to generalize and identify genres correctly, resulting in acceptable precision but poor coverage.

### Model 2: ResNet50 (Transfer Learning)

- [cite_start]**Backbone:** ResNet-50 (50 layers) pre-trained on ImageNet[cite: 55].

- [cite_start]**Configuration:** The first 10 layers (Conv1) were **frozen** to retain low-level feature extraction capabilities[cite: 57].

- **Classifier Head:**

  - FC1: 2048 $\rightarrow$ 1024

  - FC2: 1024 $\rightarrow$ 512

  - [cite_start]FC3: 512 $\rightarrow$ 27 (Output Classes) [cite: 73]

- **Performance:**

  - Validation Loss: 0.163

  - Micro F1: 0.4376

  - [cite_start]Precision weighted (F0.5): 55.40% [cite: 107]

#### Threshold Optimization

We tested multiple thresholds to balance Precision and Recall. The model performance varies significantly with different threshold values:

| Threshold | Accuracy   | Precision  | Recall     | F1 (Micro) | F1 (Macro) | F0.5       |
| :-------- | :--------- | :--------- | :--------- | :--------- | :--------- | :--------- |
| 0.3       | 12.53%     | 51.06%     | 43.94%     | 0.4723     | 0.1010     | 0.4945     |
| 0.5       | 17.69%     | 59.05%     | 33.43%     | 0.4269     | 0.0670     | 0.5120     |
| **0.7**   | **17.91%** | **59.98%** | **30.12%** | **0.4010** | **0.0541** | **0.5005** |

**Note:** The model uses a **default threshold of 0.7** for predictions, which provides the highest precision (59.98%) and accuracy (17.91%) while maintaining good performance. This threshold can be adjusted when using the prediction script.

[cite_start]_[cite: 110]_

**Model Performance vs. Random Baseline:**

The model predicts across **28 different genre classes**. For comparison, a random baseline that guesses genres uniformly would achieve approximately **3.57% precision** (1/28 ≈ 0.0357).

Our model significantly outperforms this baseline:

- **Threshold 0.3:** 51.06% precision → **13.3x better** than random (1,330% improvement)
- **Threshold 0.5:** 59.05% precision → **15.5x better** than random (1,554% improvement)
- **Threshold 0.7:** 59.98% precision → **15.8x better** than random (1,580% improvement)

The model demonstrates substantial learning capability, achieving precision scores that are over **16 times better** than random guessing, indicating successful extraction of meaningful visual features from movie posters for genre classification.

### Model 3: Optimized CNN

We attempted a third architecture to break the precision ceiling of 0.65 observed in previous attempts.

- [cite_start]**Structure:** Deeper custom CNN with Dropout layers to prevent overfitting[cite: 120].

- [cite_start]**Observations:** The model showed potential to reach 0.8 precision with extended training time, though ResNet50 remained the most robust for general application[cite: 148].

## Future Improvements

[cite_start]To further improve the model's performance, specifically Recall, we have identified the following areas for development[cite: 167]:

- [cite_start]**Advanced Data Augmentation:** Implementing random crops, rotation, and color jitter (brightness/contrast/saturation) to force the model to learn more complex features[cite: 172].

- **Regularization:** Tuning Dropout rates to mitigate overfitting.

- **Handling Class Imbalance:** Utilizing weighted loss functions or oversampling for underrepresented genres.

---

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

3. **Download the dataset:**

   - You need to obtain the movie poster images dataset separately.
   - The images should be organized in `cleaned_posters_local/` (preferred) or `Data/cleaned_posters/`.
   - Each image should be named as `{poster_id}.jpg`.

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

## Usage

### Training the Model

Note that the best model is already trained and saved as `genre_predictor_model_BCE.pth`.

To train from scratch:

```bash
python Training/train.py
```

**Training Configuration:**

- Batch size: 64
- Learning rate: 0.0015
- Epochs: 30 (with early stopping)
- Image size: 224x224
- Train/Val/Test split: 80%/10%/10%

### Evaluating the Model

**Evaluation (tests multiple thresholds):**

```bash
python Training/evaluate_model.py
```

This script will run the evaluation metrics discussed in the "Model Evolution" section, calculating Accuracy, Precision, Recall, and F1 Scores across the test set.

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

## Output Files

- `genre_predictor_model_BCE.pth`: Best model checkpoint (includes model weights, MultiLabelBinarizer, and config).
- `checkpoints/checkpoint_epoch_*.pth`: Epoch checkpoints.
