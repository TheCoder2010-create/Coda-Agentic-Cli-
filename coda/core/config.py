"""
Configuration management for Coda
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class CodaConfig:
    """Configuration manager for Coda"""
    
    DEFAULT_CONFIG = {
        'llm': {
            'default_provider': 'openai',
            'default_models': {
                'openai': 'gpt-5',
                'anthropic': 'claude-sonnet-4-20250514',
                'ollama': 'llama3'
            },
            'max_tokens': 4000,
            'temperature': 0.7
        },
        'embeddings': {
            'model': 'openai',  # or 'local' for scikit-learn based
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
        'storage': {
            'embeddings_dir': '.coda/embeddings',
            'logs_dir': '.coda/logs',
            'cache_dir': '.coda/cache'
        },
        'safety': {
            'always_preview': True,
            'safe_mode_default': True,
            'backup_before_apply': True
        },
        'plugins': {
            'enabled': [],
            'directory': '.coda/plugins'
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._ensure_directories()
    
    def _find_config_file(self) -> str:
        """Find configuration file in current directory or home"""
        candidates = [
            '.coda/config.yaml',
            '.coda/config.yml',
            os.path.expanduser('~/.coda/config.yaml'),
            os.path.expanduser('~/.coda/config.yml')
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        # Return default location
        return '.coda/config.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f) or {}
                return self._merge_config(self.DEFAULT_CONFIG, loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
        
        return self.DEFAULT_CONFIG.copy()
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _ensure_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config['storage']['embeddings_dir'],
            self.config['storage']['logs_dir'],
            self.config['storage']['cache_dir'],
            self.config['plugins']['directory']
        ]
        
        for directory in dirs_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'llm.default_provider')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    @property
    def coda_dir(self) -> str:
        """Get the .coda directory path"""
        return '.coda'