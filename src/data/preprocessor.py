"""
Data preprocessing module for XSS Detection project.

This module handles data loading, cleaning, tokenization, and preparation
for LSTM-based XSS detection models.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XSSDataPreprocessor:
    """
    Data preprocessor for XSS detection dataset.
    
    Handles data loading, cleaning, tokenization, and train/test splitting
    for LSTM-based XSS detection models.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.data_config = config.get_data_config()
        self.paths_config = config.get_paths_config()
        self.tokenizer = None
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and perform basic cleaning of the dataset.
        
        Args:
            file_path: Path to the CSV data file
            
        Returns:
            Cleaned pandas DataFrame
        """
        if file_path is None:
            file_path = self.paths_config.get('data_file', 'data.csv')
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} samples")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Basic cleaning
        df = self._clean_data(df)
        logger.info(f"After cleaning: {len(df)} samples")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by removing duplicates, nulls, and short payloads.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        initial_count = len(df)
        df.drop_duplicates(subset='payload', inplace=True)
        logger.info(f"Removed {initial_count - len(df)} duplicate payloads")
        
        # Remove null payloads
        df.dropna(subset=['payload'], inplace=True)
        
        # Convert to string and strip whitespace
        df['payload'] = df['payload'].astype(str).str.strip()
        
        # Filter by minimum length
        min_length = self.data_config.get('min_length', 10)
        df = df[df['payload'].str.len() >= min_length]
        
        # Add length statistics
        df['length'] = df['payload'].apply(lambda x: len(x.split()))
        
        logger.info("Payload length statistics:")
        logger.info(df['length'].describe())
        
        return df
    
    def create_tokenizer(self, texts: pd.Series) -> Tokenizer:
        """
        Create and fit tokenizer on the text data.
        
        Args:
            texts: Series of text payloads
            
        Returns:
            Fitted Keras Tokenizer
        """
        max_vocab = self.data_config.get('max_vocab_size', 10000)
        
        logger.info(f"Creating tokenizer with max vocabulary size: {max_vocab}")
        
        self.tokenizer = Tokenizer(num_words=max_vocab, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer.word_index) + 1}")
        
        return self.tokenizer
    
    def tokenize_and_pad(self, texts: pd.Series) -> np.ndarray:
        """
        Tokenize and pad sequences to uniform length.
        
        Args:
            texts: Series of text payloads
            
        Returns:
            Padded sequences as numpy array
        """
        if self.tokenizer is None:
            self.create_tokenizer(texts)
        
        max_len = self.data_config.get('max_sequence_length', 200)
        
        logger.info(f"Tokenizing and padding sequences to length: {max_len}")
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=max_len, 
            padding='post', 
            truncating='post'
        )
        
        return padded_sequences
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.data_config.get('test_size', 0.2)
        random_state = self.data_config.get('random_state', 42)
        
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Save processed data to disk.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        output_path = self.paths_config.get('processed_data', 'processed_data.npz')
        
        logger.info(f"Saving processed data to {output_path}")
        
        np.savez_compressed(
            output_path,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
    
    def save_tokenizer(self) -> None:
        """Save the fitted tokenizer to disk."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted. Call create_tokenizer() first.")
        
        tokenizer_path = self.paths_config.get('tokenizer', 'tokenizer.pkl')
        
        logger.info(f"Saving tokenizer to {tokenizer_path}")
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def preprocess(self, file_path: str = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path: Path to the CSV data file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Load and clean data
        df = self.load_data(file_path)
        
        # Prepare features and labels
        X = self.tokenize_and_pad(df['payload'])
        y = np.array(df['label'])
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test)
        self.save_tokenizer()
        
        logger.info("Data preprocessing completed successfully")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'tokenizer': self.tokenizer,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'max_len': X_train.shape[1]
        }


def main():
    """Main function for standalone preprocessing execution."""
    config = Config()
    preprocessor = XSSDataPreprocessor(config)
    
    try:
        processed_data = preprocessor.preprocess()
        print("✅ Data preprocessing completed successfully!")
        print(f"📊 Training set: {processed_data['X_train'].shape}")
        print(f"📊 Test set: {processed_data['X_test'].shape}")
        print(f"📊 Vocabulary size: {processed_data['vocab_size']}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 