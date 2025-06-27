"""
Main training script for XSS Detection models.

This script orchestrates the complete training pipeline including data preprocessing,
model training, and evaluation for both LSTM and LSTM with Attention models.
"""

import os
import logging
import argparse
from typing import Dict, Any

from utils.config import Config
from data.preprocessor import XSSDataPreprocessor
from models.lstm_model import XSSLSTMModel
from models.lstm_attention_model import XSSLSTMAttentionModel
from evaluation.evaluator import XSSModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XSSTrainingPipeline:
    """
    Complete training pipeline for XSS detection models.
    
    Handles data preprocessing, model training, and evaluation in a
    coordinated manner.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the training pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessor = XSSDataPreprocessor(config)
        self.lstm_model = XSSLSTMModel(config)
        self.lstm_attention_model = XSSLSTMAttentionModel(config)
        self.evaluator = XSSModelEvaluator(config)
        
    def run_preprocessing(self, data_file: str = None) -> Dict[str, Any]:
        """
        Run the data preprocessing pipeline.
        
        Args:
            data_file: Path to the input data file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data preprocessing pipeline")
        
        try:
            processed_data = self.preprocessor.preprocess(data_file)
            logger.info("Data preprocessing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def train_lstm_model(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the basic LSTM model.
        
        Args:
            processed_data: Dictionary containing processed data
            
        Returns:
            Training history
        """
        logger.info("Training LSTM model")
        
        try:
            # Build model
            self.lstm_model.build_model(
                vocab_size=processed_data['vocab_size'],
                max_len=processed_data['max_len']
            )
            
            # Train model
            history = self.lstm_model.train(
                processed_data['X_train'],
                processed_data['y_train']
            )
            
            # Evaluate model
            metrics = self.lstm_model.evaluate(
                processed_data['X_test'],
                processed_data['y_test']
            )
            
            # Save model
            self.lstm_model.save_model()
            
            logger.info("LSTM model training completed successfully")
            return {
                'history': history,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"LSTM model training failed: {str(e)}")
            raise
    
    def train_lstm_attention_model(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the LSTM with Attention model.
        
        Args:
            processed_data: Dictionary containing processed data
            
        Returns:
            Training history
        """
        logger.info("Training LSTM with Attention model")
        
        try:
            # Build model
            self.lstm_attention_model.build_model(
                vocab_size=processed_data['vocab_size'],
                max_len=processed_data['max_len']
            )
            
            # Train model
            history = self.lstm_attention_model.train(
                processed_data['X_train'],
                processed_data['y_train']
            )
            
            # Evaluate model
            metrics = self.lstm_attention_model.evaluate(
                processed_data['X_test'],
                processed_data['y_test']
            )
            
            # Save model
            self.lstm_attention_model.save_model()
            
            logger.info("LSTM with Attention model training completed successfully")
            return {
                'history': history,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"LSTM with Attention model training failed: {str(e)}")
            raise
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Returns:
            Evaluation results
        """
        logger.info("Evaluating all models")
        
        try:
            results = self.evaluator.evaluate_all_models()
            logger.info("Model evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise
    
    def run_complete_pipeline(self, data_file: str = None, 
                            skip_preprocessing: bool = False) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_file: Path to the input data file
            skip_preprocessing: Whether to skip preprocessing (use existing data)
            
        Returns:
            Dictionary containing all results
        """
        logger.info("Starting complete XSS detection training pipeline")
        
        results = {}
        
        try:
            # Step 1: Data preprocessing
            if not skip_preprocessing:
                processed_data = self.run_preprocessing(data_file)
                results['preprocessing'] = processed_data
            else:
                logger.info("Skipping preprocessing, using existing data")
                # Load existing processed data
                import numpy as np
                data = np.load(self.config.get('paths.processed_data', 'processed_data.npz'))
                processed_data = {
                    'X_train': data['X_train'],
                    'X_test': data['X_test'],
                    'y_train': data['y_train'],
                    'y_test': data['y_test'],
                    'vocab_size': 10000,  # Default value
                    'max_len': data['X_train'].shape[1]
                }
            
            # Step 2: Train LSTM model
            lstm_results = self.train_lstm_model(processed_data)
            results['lstm_training'] = lstm_results
            
            # Step 3: Train LSTM with Attention model
            lstm_attention_results = self.train_lstm_attention_model(processed_data)
            results['lstm_attention_training'] = lstm_attention_results
            
            # Step 4: Evaluate all models
            evaluation_results = self.evaluate_models()
            results['evaluation'] = evaluation_results
            
            logger.info("Complete training pipeline finished successfully")
            
            # Print summary
            self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the training results.
        
        Args:
            results: Dictionary containing all results
        """
        print("\n" + "="*60)
        print("XSS DETECTION TRAINING PIPELINE SUMMARY")
        print("="*60)
        
        if 'lstm_training' in results:
            lstm_metrics = results['lstm_training']['metrics']
            print(f"\n📊 LSTM Model Results:")
            print(f"   Accuracy: {lstm_metrics['test_accuracy']:.4f}")
            print(f"   F1-Score: {lstm_metrics['test_f1']:.4f}")
            print(f"   ROC AUC: {lstm_metrics['test_roc_auc']:.4f}")
        
        if 'lstm_attention_training' in results:
            lstm_att_metrics = results['lstm_attention_training']['metrics']
            print(f"\n📊 LSTM with Attention Model Results:")
            print(f"   Accuracy: {lstm_att_metrics['test_accuracy']:.4f}")
            print(f"   F1-Score: {lstm_att_metrics['test_f1']:.4f}")
            print(f"   ROC AUC: {lstm_att_metrics['test_roc_auc']:.4f}")
        
        print("\n✅ Training pipeline completed successfully!")
        print("📁 Models saved in 'models/' directory")
        print("📁 Results saved in 'results/' directory")


def main():
    """Main function for the training pipeline."""
    parser = argparse.ArgumentParser(description='XSS Detection Training Pipeline')
    parser.add_argument(
        '--data-file', 
        type=str, 
        default=None,
        help='Path to the input data file (CSV)'
    )
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip preprocessing and use existing processed data'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Create training pipeline
        pipeline = XSSTrainingPipeline(config)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            data_file=args.data_file,
            skip_preprocessing=args.skip_preprocessing
        )
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 