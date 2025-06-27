"""
Configuration management utilities for XSS Detection project.
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration manager for the XSS Detection project."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'data.max_vocab_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data preprocessing configuration."""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get file paths configuration."""
        return self.config.get('paths', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {}) 