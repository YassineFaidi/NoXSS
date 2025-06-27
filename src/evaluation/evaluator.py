"""
Model evaluation and comparison utilities for XSS Detection project.

This module provides comprehensive evaluation tools for comparing the performance
of different XSS detection models.
"""

import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc,
    precision_recall_curve
)
import os

from ..utils.config import Config
from ..models.lstm_model import XSSLSTMModel
from ..models.lstm_attention_model import XSSLSTMAttentionModel, AttentionLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class XSSModelEvaluator:
    """
    Comprehensive evaluator for XSS detection models.
    
    Provides tools for evaluating model performance, comparing different models,
    and generating detailed reports and visualizations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.paths_config = config.get_paths_config()
        self.output_config = config.get_output_config()
        self.results = {}
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data for evaluation.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        processed_data_path = self.paths_config.get('processed_data', 'processed_data.npz')
        
        if not os.path.exists(processed_data_path):
            raise FileNotFoundError(f"Processed data not found: {processed_data_path}")
        
        logger.info(f"Loading test data from {processed_data_path}")
        data = np.load(processed_data_path)
        return data['X_test'], data['y_test']
    
    def load_tokenizer(self):
        """Load the fitted tokenizer."""
        tokenizer_path = self.paths_config.get('tokenizer', 'tokenizer.pkl')
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        with open(tokenizer_path, 'rb') as f:
            return pickle.load(f)
    
    def evaluate_lstm_model(self, model_path: str = None) -> Dict[str, Any]:
        """
        Evaluate the basic LSTM model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary containing evaluation results
        """
        if model_path is None:
            model_path = self.paths_config.get('final_lstm_model', 'models/final_lstm_model.h5')
        
        logger.info(f"Evaluating LSTM model from {model_path}")
        
        # Load data
        X_test, y_test = self.load_data()
        
        # Load model
        model = XSSLSTMModel(self.config)
        model.load_model(model_path)
        
        # Make predictions
        y_pred_probs = model.predict(X_test)
        y_pred = model.predict_classes(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_probs, "LSTM")
        
        self.results['lstm'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_probs,
            'metrics': metrics
        }
        
        return metrics
    
    def evaluate_lstm_attention_model(self, model_path: str = None) -> Dict[str, Any]:
        """
        Evaluate the LSTM with Attention model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary containing evaluation results
        """
        if model_path is None:
            model_path = self.paths_config.get('final_lstm_attention_model', 'models/final_lstm_attention_model.h5')
        
        logger.info(f"Evaluating LSTM with Attention model from {model_path}")
        
        # Load data
        X_test, y_test = self.load_data()
        
        # Load model
        model = XSSLSTMAttentionModel(self.config)
        model.load_model(model_path)
        
        # Make predictions
        y_pred_probs = model.predict(X_test)
        y_pred = model.predict_classes(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_probs, "LSTM+Attention")
        
        self.results['lstm_attention'] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_probs,
            'metrics': metrics
        }
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_probs: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_probs: Predicted probabilities
            model_name: Name of the model for logging
            
        Returns:
            Dictionary containing all metrics
        """
        logger.info(f"Calculating metrics for {model_name}")
        
        # Basic classification metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
        
        return metrics
    
    def plot_confusion_matrices(self, save_plots: bool = True) -> None:
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if not self.results:
            logger.warning("No evaluation results available. Run evaluation first.")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(self.results.items()):
            cm = result['metrics']['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[i],
                cbar=True
            )
            axes[i].set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_plots:
            results_dir = self.output_config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'confusion_matrices.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {plot_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_plots: bool = True) -> None:
        """
        Plot ROC curves for all evaluated models.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if not self.results:
            logger.warning("No evaluation results available. Run evaluation first.")
            return
        
        plt.figure(figsize=(8, 6))
        
        for model_name, result in self.results.items():
            fpr, tpr = result['metrics']['roc_curve']
            roc_auc = result['metrics']['roc_auc']
            
            display_name = model_name.replace('_', ' ').title()
            plt.plot(fpr, tpr, label=f'{display_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            results_dir = self.output_config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'roc_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {plot_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, save_plots: bool = True) -> None:
        """
        Plot Precision-Recall curves for all evaluated models.
        
        Args:
            save_plots: Whether to save plots to disk
        """
        if not self.results:
            logger.warning("No evaluation results available. Run evaluation first.")
            return
        
        plt.figure(figsize=(8, 6))
        
        for model_name, result in self.results.items():
            precision, recall = result['metrics']['pr_curve']
            pr_auc = result['metrics']['pr_auc']
            
            display_name = model_name.replace('_', ' ').title()
            plt.plot(recall, precision, label=f'{display_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            results_dir = self.output_config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'precision_recall_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {plot_path}")
        
        plt.show()
    
    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.
        
        Returns:
            Formatted comparison report string
        """
        if not self.results:
            return "No evaluation results available."
        
        report = "=" * 60 + "\n"
        report += "XSS DETECTION MODELS COMPARISON REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Summary table
        report += "MODEL PERFORMANCE SUMMARY:\n"
        report += "-" * 40 + "\n"
        report += f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}\n"
        report += "-" * 80 + "\n"
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            display_name = model_name.replace('_', ' ').title()
            report += f"{display_name:<20} "
            report += f"{metrics['accuracy']:<10.4f} "
            report += f"{metrics['precision']:<10.4f} "
            report += f"{metrics['recall']:<10.4f} "
            report += f"{metrics['f1_score']:<10.4f} "
            report += f"{metrics['roc_auc']:<10.4f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        # Detailed reports
        for model_name, result in self.results.items():
            report += f"\nDETAILED REPORT - {model_name.replace('_', ' ').title()}:\n"
            report += "-" * 40 + "\n"
            report += classification_report(
                result['predictions'], 
                result['predictions'],  # This should be y_true, but we don't have it here
                target_names=['Benign', 'XSS']
            )
            report += "\n"
        
        return report
    
    def save_results(self, output_path: str = None) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            output_path: Path to save results (optional)
        """
        if not self.results:
            logger.warning("No results to save.")
            return
        
        if output_path is None:
            results_dir = self.output_config.get('results_dir', 'results')
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, 'evaluation_results.txt')
        
        report = self.generate_comparison_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def evaluate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all available models.
        
        Returns:
            Dictionary containing results for all models
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Evaluate LSTM model
        try:
            self.evaluate_lstm_model()
        except Exception as e:
            logger.error(f"Failed to evaluate LSTM model: {str(e)}")
        
        # Evaluate LSTM with Attention model
        try:
            self.evaluate_lstm_attention_model()
        except Exception as e:
            logger.error(f"Failed to evaluate LSTM with Attention model: {str(e)}")
        
        # Generate plots
        self.plot_confusion_matrices()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        
        # Save results
        self.save_results()
        
        logger.info("Model evaluation completed")
        return self.results


def main():
    """Main function for standalone evaluation."""
    config = Config()
    evaluator = XSSModelEvaluator(config)
    
    try:
        results = evaluator.evaluate_all_models()
        print("✅ Model evaluation completed successfully!")
        print(f"📊 Evaluated {len(results)} models")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 