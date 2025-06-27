"""
Quick Start Example for XSS Detection

This script demonstrates how to quickly get started with the XSS detection
models using the provided dataset.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import Config
from data.preprocessor import XSSDataPreprocessor
from models.lstm_model import XSSLSTMModel
from evaluation.evaluator import XSSModelEvaluator


def quick_start_example():
    """Run a quick start example with the LSTM model."""
    print("🚀 XSS Detection - Quick Start Example")
    print("=" * 50)
    
    # Load configuration
    config = Config()
    
    # Step 1: Preprocess data
    print("\n📊 Step 1: Data Preprocessing")
    preprocessor = XSSDataPreprocessor(config)
    processed_data = preprocessor.preprocess()
    
    print(f"✅ Data preprocessing completed!")
    print(f"   Training samples: {processed_data['X_train'].shape[0]}")
    print(f"   Test samples: {processed_data['X_test'].shape[0]}")
    print(f"   Vocabulary size: {processed_data['vocab_size']}")
    
    # Step 2: Train LSTM model
    print("\n🤖 Step 2: Training LSTM Model")
    model = XSSLSTMModel(config)
    model.build_model(
        vocab_size=processed_data['vocab_size'],
        max_len=processed_data['max_len']
    )
    
    history = model.train(
        processed_data['X_train'],
        processed_data['y_train']
    )
    
    # Step 3: Evaluate model
    print("\n📈 Step 3: Model Evaluation")
    metrics = model.evaluate(
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    print(f"✅ Model evaluation completed!")
    print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   Precision: {metrics['test_precision']:.4f}")
    print(f"   Recall: {metrics['test_recall']:.4f}")
    print(f"   F1-Score: {metrics['test_f1']:.4f}")
    
    # Step 4: Save model
    model.save_model()
    print(f"\n💾 Model saved successfully!")
    
    print("\n🎉 Quick start example completed successfully!")


if __name__ == "__main__":
    quick_start_example() 