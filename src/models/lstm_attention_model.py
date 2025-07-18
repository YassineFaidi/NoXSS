"""
LSTM with Attention-based XSS Detection Model.

This module provides an LSTM neural network with attention mechanism for
detecting Cross-Site Scripting (XSS) vulnerabilities in web payloads.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

from ..utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(Layer):
    """
    Custom attention layer for LSTM models.
    
    Implements a simple attention mechanism that learns to focus on
    important parts of the input sequence.
    """
    
    def __init__(self, **kwargs):
        """Initialize the attention layer."""
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        Build the attention layer weights.
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super().build(input_shape)
    
    def call(self, inputs):
        """
        Apply attention mechanism to the input.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Context vector of shape (batch_size, features)
        """
        # Calculate attention scores
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        
        # Apply softmax to get attention weights
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        
        # Calculate context vector
        context = inputs * alpha
        return K.sum(context, axis=1)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super(AttentionLayer, self).get_config()
        return config


class XSSLSTMAttentionModel:
    """
    LSTM with Attention model for XSS detection.
    
    A neural network using LSTM layers with attention mechanism to classify
    web payloads as malicious (XSS) or benign.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the LSTM with Attention model.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.model_config = config.get_model_config()
        self.training_config = config.get_training_config()
        self.paths_config = config.get_paths_config()
        self.model = None
        self.history = None
        
    def build_model(self, vocab_size: int, max_len: int) -> Model:
        """
        Build the LSTM with Attention model architecture.
        
        Args:
            vocab_size: Size of the vocabulary
            max_len: Maximum sequence length
            
        Returns:
            Compiled Keras Model
        """
        embedding_dim = self.model_config.get('embedding_dim', 100)
        lstm_units = self.model_config.get('lstm_units', 128)
        dense_units = self.model_config.get('dense_units', 64)
        dropout_rate = self.model_config.get('dropout_rate', 0.5)
        
        logger.info(f"Building LSTM with Attention model:")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        logger.info(f"  - Max sequence length: {max_len}")
        logger.info(f"  - Embedding dimension: {embedding_dim}")
        logger.info(f"  - LSTM units: {lstm_units}")
        logger.info(f"  - Dense units: {dense_units}")
        logger.info(f"  - Dropout rate: {dropout_rate}")
        
        # Input layer
        inputs = Input(shape=(max_len,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len
        )(inputs)
        
        # LSTM layer with return_sequences=True for attention
        lstm = LSTM(lstm_units, return_sequences=True)(embedding)
        
        # Attention layer
        attention = AttentionLayer()(lstm)
        
        # Dropout for regularization
        dropout = Dropout(dropout_rate)(attention)
        
        # Dense layers
        dense = Dense(dense_units, activation='relu')(dropout)
        output = Dense(1, activation='sigmoid')(dense)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=output)
        
        # Compile the model
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        logger.info("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks for the model.
        
        Returns:
            List of Keras callbacks
        """
        patience = self.training_config.get('patience', 2)
        model_path = self.paths_config.get('lstm_attention_model', 'models/lstm_attention_model.h5')
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the LSTM with Attention model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        batch_size = self.training_config.get('batch_size', 32)
        epochs = self.training_config.get('epochs', 10)
        validation_split = self.training_config.get('validation_split', 0.2)
        
        logger.info(f"Starting training with:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Validation split: {validation_split}")
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
        else:
            validation_data = None
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        logger.info("Training completed successfully")
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model on test set")
        
        # Evaluate the model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        logger.info("Test Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict classes (0 or 1) based on probability threshold.
        
        Args:
            X: Input features
            threshold: Classification threshold
            
        Returns:
            Predicted classes
        """
        probabilities = self.predict(X)
        return (probabilities > threshold).astype(int)
    
    def save_model(self, file_path: str = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if file_path is None:
            file_path = self.paths_config.get('final_lstm_attention_model', 'models/final_lstm_attention_model.h5')
        
        logger.info(f"Saving model to {file_path}")
        self.model.save(file_path, include_optimizer=True)
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
        """
        logger.info(f"Loading model from {file_path}")
        self.model = tf.keras.models.load_model(
            file_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info("Model loaded successfully")


def main():
    """Main function for standalone model training."""
    import os
    
    # Load configuration
    config = Config()
    
    # Check if processed data exists
    processed_data_path = config.get('paths.processed_data', 'processed_data.npz')
    if not os.path.exists(processed_data_path):
        logger.error("Processed data not found. Please run preprocessing first.")
        return
    
    # Load processed data
    data = np.load(processed_data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    # Create and train model
    model = XSSLSTMAttentionModel(config)
    model.build_model(vocab_size=10000, max_len=X_train.shape[1])
    
    # Train the model
    history = model.train(X_train, y_train)
    
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)
    
    # Save the model
    model.save_model()
    
    print("✅ LSTM with Attention model training completed successfully!")
    print(f"📊 Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"📊 Test F1 Score: {metrics['test_f1']:.4f}")


if __name__ == "__main__":
    main() 