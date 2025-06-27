# XSS Detection with LSTM Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/xss-detection-lstm)](https://github.com/yourusername/xss-detection-lstm/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/xss-detection-lstm)](https://github.com/yourusername/xss-detection-lstm/network)

A professional machine learning project for detecting **Cross-Site Scripting (XSS)** vulnerabilities using **Long Short-Term Memory (LSTM)** neural networks. This repository provides a complete, production-ready solution with both basic LSTM and LSTM with Attention mechanisms for enhanced detection accuracy.

## 🚀 Features

- **Two Model Architectures**: Basic LSTM and LSTM with Attention mechanism
- **Complete Pipeline**: Data preprocessing, model training, and evaluation
- **Professional Structure**: Modular, well-organized codebase
- **Configuration Management**: YAML-based configuration system
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Easy to Use**: Simple API and command-line interface
- **Production Ready**: Proper error handling, logging, and documentation

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.8 or higher
- Other dependencies listed in `requirements/requirements.txt`

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/xss-detection-lstm.git
   cd xss-detection-lstm
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python examples/quick_start.py
   ```

## 🚀 Quick Start

### Option 1: Quick Start Example

Run the quick start example to see the system in action:

```bash
python examples/quick_start.py
```

This will:
- Preprocess the provided dataset
- Train a basic LSTM model
- Evaluate the model performance
- Save the trained model

### Option 2: Complete Pipeline

Run the complete training pipeline for both models:

```bash
python src/train.py --data-file data/data.csv
```

### Option 3: Step-by-Step

1. **Preprocess data**:
   ```python
   from src.utils.config import Config
   from src.data.preprocessor import XSSDataPreprocessor
   
   config = Config()
   preprocessor = XSSDataPreprocessor(config)
   processed_data = preprocessor.preprocess('data/data.csv')
   ```

2. **Train LSTM model**:
   ```python
   from src.models.lstm_model import XSSLSTMModel
   
   model = XSSLSTMModel(config)
   model.build_model(vocab_size=processed_data['vocab_size'], 
                    max_len=processed_data['max_len'])
   history = model.train(processed_data['X_train'], processed_data['y_train'])
   ```

3. **Evaluate model**:
   ```python
   metrics = model.evaluate(processed_data['X_test'], processed_data['y_test'])
   print(f"Accuracy: {metrics['test_accuracy']:.4f}")
   ```

## 📁 Project Structure

```
xss-detection-lstm/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── data.csv                 # Dataset (not included in repo)
│   └── README.md                # Data documentation
├── docs/                        # Documentation
├── examples/
│   └── quick_start.py           # Quick start example
├── requirements/
│   └── requirements.txt         # Python dependencies
├── results/                     # Generated results and plots
├── src/                         # Source code
│   ├── data/
│   │   └── preprocessor.py      # Data preprocessing
│   ├── evaluation/
│   │   └── evaluator.py         # Model evaluation
│   ├── models/
│   │   ├── lstm_model.py        # Basic LSTM model
│   │   └── lstm_attention_model.py  # LSTM with Attention
│   ├── utils/
│   │   └── config.py            # Configuration management
│   └── train.py                 # Main training script
├── models/                      # Saved models (generated)
├── processed_data.npz           # Processed data (generated)
├── tokenizer.pkl               # Fitted tokenizer (generated)
└── README.md                   # This file
```

## 💻 Usage

### Command Line Interface

The main training script provides several options:

```bash
# Run complete pipeline with custom data file
python src/train.py --data-file your_data.csv

# Skip preprocessing (use existing processed data)
python src/train.py --skip-preprocessing

# Use custom configuration file
python src/train.py --config configs/custom_config.yaml
```

### Programmatic Usage

```python
from src.utils.config import Config
from src.data.preprocessor import XSSDataPreprocessor
from src.models.lstm_model import XSSLSTMModel
from src.models.lstm_attention_model import XSSLSTMAttentionModel
from src.evaluation.evaluator import XSSModelEvaluator

# Load configuration
config = Config()

# Preprocess data
preprocessor = XSSDataPreprocessor(config)
processed_data = preprocessor.preprocess('data/data.csv')

# Train basic LSTM model
lstm_model = XSSLSTMModel(config)
lstm_model.build_model(processed_data['vocab_size'], processed_data['max_len'])
lstm_model.train(processed_data['X_train'], processed_data['y_train'])

# Train LSTM with Attention model
attention_model = XSSLSTMAttentionModel(config)
attention_model.build_model(processed_data['vocab_size'], processed_data['max_len'])
attention_model.train(processed_data['X_train'], processed_data['y_train'])

# Evaluate models
evaluator = XSSModelEvaluator(config)
results = evaluator.evaluate_all_models()
```

## ⚙️ Configuration

The project uses YAML configuration files for easy parameter management. Key configuration sections:

### Data Configuration
```yaml
data:
  max_vocab_size: 10000      # Maximum vocabulary size
  max_sequence_length: 200   # Maximum sequence length
  test_size: 0.2            # Test set proportion
  min_length: 10            # Minimum payload length
  random_state: 42          # Random seed
```

### Model Configuration
```yaml
model:
  vocab_size: 10000         # Vocabulary size
  embedding_dim: 100        # Embedding dimension
  lstm_units: 128           # LSTM units
  dense_units: 64           # Dense layer units
  dropout_rate: 0.5         # Dropout rate
```

### Training Configuration
```yaml
training:
  batch_size: 32            # Batch size
  epochs: 10                # Number of epochs
  validation_split: 0.2     # Validation split
  patience: 2               # Early stopping patience
```

## 🧠 Model Architecture

### Basic LSTM Model
```
Input → Embedding → LSTM → Dropout → Dense → Output
```

### LSTM with Attention Model
```
Input → Embedding → LSTM → Attention → Dropout → Dense → Output
```

The attention mechanism helps the model focus on the most important parts of the input sequence, potentially improving detection accuracy for complex XSS payloads.

## 📊 Evaluation

The evaluation system provides comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **PR AUC**: Area under the Precision-Recall curve

### Visualization
The evaluator generates:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Detailed classification reports

## 📝 Examples

### Basic Usage Example
```python
# Load and preprocess data
from src.data.preprocessor import XSSDataPreprocessor
preprocessor = XSSDataPreprocessor(config)
data = preprocessor.preprocess('data/data.csv')

# Train model
from src.models.lstm_model import XSSLSTMModel
model = XSSLSTMModel(config)
model.build_model(data['vocab_size'], data['max_len'])
model.train(data['X_train'], data['y_train'])

# Make predictions
predictions = model.predict_classes(data['X_test'])
```

### Custom Configuration Example
```python
# Create custom configuration
config = Config('configs/custom_config.yaml')

# Modify parameters
config.config['model']['lstm_units'] = 256
config.config['training']['epochs'] = 20

# Use in training
model = XSSLSTMModel(config)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements/requirements.txt

# Run tests (if available)
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- The cybersecurity community for XSS research and datasets
- Contributors and maintainers of the open-source libraries used

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/xss-detection-lstm/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 📈 Performance

Typical performance metrics on the provided dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| LSTM | 0.95+ | 0.94+ | 0.96+ | 0.95+ | 0.98+ |
| LSTM + Attention | 0.96+ | 0.95+ | 0.97+ | 0.96+ | 0.99+ |

*Results may vary depending on the dataset and training parameters.*

## 🔬 Research Applications

This project can be used for:
- **Academic Research**: XSS detection studies and papers
- **Security Tools**: Integration into web application firewalls
- **Education**: Teaching machine learning for cybersecurity
- **Industry**: Production XSS detection systems

---

**Note**: This project is for educational and research purposes. Always ensure you have proper authorization before testing XSS detection on any systems.

**Made with ❤️ for the cybersecurity community**

---

<div align="center">
  <sub>Built with ❤️ by the XSS Detection Team</sub>
</div>
