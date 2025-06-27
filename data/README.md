# Data Directory

This directory contains the dataset used for XSS detection training.

## Files

- `data.csv` - XSS detection dataset with payload and label columns

## Dataset Format

The dataset should contain at least two columns:
- `payload` - The web payload text to analyze
- `label` - Binary label (0 for benign, 1 for XSS)

## Usage

The dataset is automatically used by the training pipeline:

```bash
# The configuration points to this file
python src/train.py --data-file data/data.csv

# Or use the default path
python src/train.py
```

## Data Privacy

If this dataset contains sensitive information:
1. Add `data/` to your `.gitignore` file
2. Consider using a smaller sample for public repositories
3. Ensure compliance with data protection regulations

## Adding New Data

To use a different dataset:
1. Replace `data.csv` with your new file
2. Ensure it has the same column structure
3. Update the configuration if needed

## Backup

Consider backing up your original dataset before running preprocessing, as the pipeline will create processed versions. 