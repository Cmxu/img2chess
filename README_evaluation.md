# Model Evaluation Script

This script provides comprehensive evaluation of the trained chess piece classifier model using the same dataloader that was used during training.

## Features

- **Comprehensive Metrics**: Overall accuracy, per-class accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visual representation of prediction errors
- **Confidence Analysis**: Distribution of confidence scores for correct vs incorrect predictions
- **Wrong Predictions Analysis**: Save and visualize images that the model classified incorrectly
- **Detailed Reports**: Both console output and Excel file with all results
- **Visualizations**: Automatic generation of plots for analysis

## Usage

### Basic Usage
```bash
python evaluate_model.py
```

This will:
- Load the default model (`chess_piece_classifier.pth`)
- Evaluate on 5000 samples
- Use auto-detected device (MPS/CUDA/CPU)
- Save results to `evaluation_results/` directory

### Advanced Usage

```bash
python evaluate_model.py \
    --model chess_piece_classifier.pth \
    --samples 10000 \
    --batch_size 64 \
    --device cuda \
    --output_dir my_evaluation_results \
    --save_wrong_images \
    --max_wrong_images 100 \
    --no_plots
```

### Command Line Arguments

- `--model`: Path to the trained model file (default: `chess_piece_classifier.pth`)
- `--samples`: Number of evaluation samples to generate (default: 5000)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--device`: Device to use - 'auto', 'cpu', 'cuda', or 'mps' (default: 'auto')
- `--output_dir`: Directory to save results (default: 'evaluation_results')
- `--no_plots`: Skip generating plots (useful for faster evaluation)
- `--save_wrong_images`: Save individual images of wrong predictions to a separate folder
- `--max_wrong_images`: Maximum number of wrong images to show in grid visualization (default: 64)

## Outputs

The script generates several outputs in the specified directory:

### 1. Console Report
Detailed evaluation report printed to console including:
- Overall accuracy
- Per-class accuracy
- Best and worst performing classes
- Average confidence for correct/incorrect predictions
- Full classification report (precision, recall, F1-score)

### 2. Excel File (`evaluation_results.xlsx`)
Contains three sheets:
- **Detailed_Results**: Individual predictions with true labels, predicted labels, confidence, and correctness
- **Overall_Metrics**: Summary metrics (accuracy, confidence statistics, total samples)
- **Class_Accuracy**: Per-class accuracy breakdown

### 3. Wrong Predictions Analysis (if using `--save_wrong_images`)

#### Individual Images (`wrong_predictions/`)
- Each incorrectly classified image is saved as a separate PNG file
- Filename format: `sample_XXXXX_true_CLASS_pred_CLASS_conf_X.XXX.png`
- Example: `sample_00123_true_wp_pred_bp_conf_0.856.png`

#### Summary File (`wrong_predictions/wrong_predictions_summary.txt`)
- Text file listing all wrong predictions with sample index, true label, predicted label, and confidence
- Easy to scan for patterns in model mistakes

#### Grid Visualization (`wrong_predictions_grid.png`)
- Grid layout showing up to `--max_wrong_images` wrong predictions
- Each image shows true label, predicted label, and confidence
- Helps identify visual patterns in model errors

### 4. Visualizations (if not using `--no_plots`)

#### Confusion Matrix (`confusion_matrix.png`)
- Shows prediction vs actual class distribution
- Helps identify which classes are commonly confused
- Darker colors indicate higher counts

#### Per-Class Accuracy (`class_accuracy.png`)
- Bar chart showing accuracy for each chess piece class
- Helps identify problematic classes
- Values displayed on top of each bar

#### Confidence Distribution (`confidence_distribution.png`)
- Histogram showing confidence distribution
- Separate curves for correct vs incorrect predictions
- Helps understand model's confidence calibration

## Example Output

```
============================================================
MODEL EVALUATION REPORT
============================================================

Overall Accuracy: 0.9234 (92.34%)
Total Samples: 5,000

Average Confidence:
  Correct predictions: 0.9456
  Incorrect predictions: 0.7234

Per-Class Accuracy:
  empty: 0.9567 (95.67%)
    wp: 0.9234 (92.34%)
    wr: 0.9345 (93.45%)
    wn: 0.9123 (91.23%)
    wb: 0.9456 (94.56%)
    wq: 0.9678 (96.78%)
    wk: 0.9789 (97.89%)
    bp: 0.9012 (90.12%)
    br: 0.9234 (92.34%)
    bn: 0.8890 (88.90%)
    bb: 0.9345 (93.45%)
    bq: 0.9567 (95.67%)
    bk: 0.9789 (97.89%)

Best performing classes:
  1. wk: 0.9789 (97.89%)
  2. bk: 0.9789 (97.89%)
  3. wq: 0.9678 (96.78%)

Worst performing classes:
  1. bn: 0.8890 (88.90%)
  2. bp: 0.9012 (90.12%)
  3. wn: 0.9123 (91.23%)

Saving 384 wrong prediction images to evaluation_results/wrong_predictions...
Wrong prediction images saved to evaluation_results/wrong_predictions
Summary file: evaluation_results/wrong_predictions/wrong_predictions_summary.txt
Wrong predictions grid saved to evaluation_results/wrong_predictions_grid.png
```

## Requirements

Make sure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

The evaluation script requires:
- torch, torchvision
- transformers
- numpy, pandas
- matplotlib, seaborn
- scikit-learn
- tqdm
- openpyxl
- pillow

## Tips

1. **Sample Size**: Use more samples (e.g., 10000) for more reliable metrics
2. **Device**: Use GPU (CUDA/MPS) for faster evaluation
3. **Batch Size**: Increase batch size on GPU for faster processing
4. **No Plots**: Use `--no_plots` for faster evaluation when you only need metrics
5. **Custom Output**: Use `--output_dir` to organize multiple evaluation runs
6. **Wrong Images**: Use `--save_wrong_images` to analyze model mistakes visually
7. **Grid Size**: Adjust `--max_wrong_images` based on how many wrong predictions you want to see in the grid

## Analyzing Wrong Predictions

When you use `--save_wrong_images`, you can:

1. **Browse Individual Images**: Look at each wrong prediction in the `wrong_predictions/` folder
2. **Check Summary File**: See patterns in the `wrong_predictions_summary.txt` file
3. **View Grid**: Get an overview of all wrong predictions in `wrong_predictions_grid.png`
4. **Identify Patterns**: Look for common mistakes like:
   - Similar pieces being confused (e.g., bishops vs knights)
   - Color confusion (white vs black pieces)
   - Background interference
   - Piece orientation issues

## Troubleshooting

- **Memory Issues**: Reduce batch size or number of samples
- **Device Errors**: Try using `--device cpu` if GPU issues occur
- **Missing Dependencies**: Run `pip install -r requirements.txt`
- **Model Not Found**: Ensure the model file exists at the specified path
- **Too Many Wrong Images**: Reduce `--max_wrong_images` if grid becomes too large 