# Migration Guide

This guide helps users migrate from the old codebase structure to the new professional organization.

## Overview of Changes

The repository has been completely reorganized to follow professional software development practices:

### Old Structure
```
├── preprocess.py
├── model_lstm.py
├── model_lstm_attention.py
├── evaluate_lstm.py
├── evaluate_lstm_attention.py
├── Readme.md
└── data.csv
```

### New Structure
```
├── src/
│   ├── data/preprocessor.py
│   ├── models/lstm_model.py
│   ├── models/lstm_attention_model.py
│   ├── evaluation/evaluator.py
│   ├── utils/config.py
│   └── train.py
├── configs/config.yaml
├── requirements/requirements.txt
├── examples/quick_start.py
├── docs/
├── setup.py
├── LICENSE
├── .gitignore
└── README.md
```

## Migration Steps

### 1. Old Files Location

All old files have been moved to the `backup/` directory:
- `backup/preprocess.py`
- `backup/model_lstm.py`
- `backup/model_lstm_attention.py`
- `backup/evaluate_lstm.py`
- `backup/evaluate_lstm_attention.py`
- `backup/Readme.md`

### 2. New Usage Patterns

#### Old Way (Direct Script Execution)
```bash
# Preprocess data
python preprocess.py

# Train LSTM model
python model_lstm.py

# Train LSTM with Attention model
python model_lstm_attention.py

# Evaluate models
python evaluate_lstm.py
python evaluate_lstm_attention.py
```

#### New Way (Professional API)
```python
# Complete pipeline
from src.utils.config import Config
from src.train import XSSTrainingPipeline

config = Config()
pipeline = XSSTrainingPipeline(config)
results = pipeline.run_complete_pipeline('data.csv')
```

#### New Way (Command Line)
```bash
# Complete pipeline
python src/train.py --data-file data.csv

# Quick start example
python examples/quick_start.py
```

### 3. Configuration Changes

#### Old Way (Hardcoded Parameters)
```python
# Parameters scattered throughout code
MAX_VOCAB = 10000
MAX_LEN = 200
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 10
```

#### New Way (Configuration File)
```yaml
# configs/config.yaml
data:
  max_vocab_size: 10000
  max_sequence_length: 200
  test_size: 0.2

training:
  batch_size: 32
  epochs: 10
```

### 4. Model Training

#### Old Way
```python
# Direct model creation and training
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, ...)
```

#### New Way
```python
# Professional model class
from src.models.lstm_model import XSSLSTMModel

model = XSSLSTMModel(config)
model.build_model(vocab_size=10000, max_len=200)
history = model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
```

### 5. Data Preprocessing

#### Old Way
```python
# Direct preprocessing in script
df = pd.read_csv('data.csv')
df.drop_duplicates(subset='payload', inplace=True)
# ... more preprocessing code
```

#### New Way
```python
# Professional preprocessor class
from src.data.preprocessor import XSSDataPreprocessor

preprocessor = XSSDataPreprocessor(config)
processed_data = preprocessor.preprocess('data.csv')
```

### 6. Evaluation

#### Old Way
```python
# Separate evaluation scripts
# evaluate_lstm.py and evaluate_lstm_attention.py
```

#### New Way
```python
# Comprehensive evaluator
from src.evaluation.evaluator import XSSModelEvaluator

evaluator = XSSModelEvaluator(config)
results = evaluator.evaluate_all_models()
evaluator.plot_confusion_matrices()
evaluator.plot_roc_curves()
```

## Key Improvements

### 1. Modularity
- **Before**: Monolithic scripts
- **After**: Modular, reusable components

### 2. Configuration Management
- **Before**: Hardcoded parameters
- **After**: YAML configuration files

### 3. Error Handling
- **Before**: Basic error handling
- **After**: Comprehensive error handling and logging

### 4. Documentation
- **Before**: Basic README
- **After**: Comprehensive documentation with API docs

### 5. Code Quality
- **Before**: Mixed French/English, inconsistent style
- **After**: Professional English code with consistent style

### 6. Testing and Validation
- **Before**: No testing framework
- **After**: Setup for testing with pytest

### 7. Deployment
- **Before**: Manual setup
- **After**: Setup.py for easy installation

## Quick Migration Checklist

- [ ] Install new dependencies: `pip install -r requirements/requirements.txt`
- [ ] Review configuration in `configs/config.yaml`
- [ ] Try quick start example: `python examples/quick_start.py`
- [ ] Run complete pipeline: `python src/train.py --data-file data.csv`
- [ ] Check generated results in `results/` directory
- [ ] Review new documentation in `docs/`

## Backward Compatibility

The old files are preserved in the `backup/` directory for reference. If you need to use the old code:

```bash
# Copy old files back (not recommended)
cp backup/preprocess.py .
cp backup/model_lstm.py .
# ... etc.
```

## Support

If you encounter issues during migration:

1. Check the new documentation in `docs/`
2. Review the examples in `examples/`
3. Check the configuration file `configs/config.yaml`
4. Ensure all dependencies are installed correctly

## Benefits of Migration

1. **Maintainability**: Easier to maintain and extend
2. **Reusability**: Components can be reused in other projects
3. **Scalability**: Easy to add new models or features
4. **Professionalism**: Industry-standard code organization
5. **Collaboration**: Easier for teams to work together
6. **Deployment**: Ready for production deployment 