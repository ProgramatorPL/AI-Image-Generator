import os
import glob
from typing import List, Dict

# Importuj scentralizowany detektor
from v_prediction_utils import VPredictionDetector

class ModelManager:
    """Manager do zarządzania modelami w folderze models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.ensure_models_directory()
    
    def ensure_models_directory(self):
        """Tworzy folder models jeśli nie istnieje"""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def scan_models(self) -> List[str]:
        """Skanuje folder models i zwraca listę plików modeli"""
        models = []
        patterns = ["*.ckpt", "*.safetensors"]
        
        for pattern in patterns:
            full_pattern = os.path.join(self.models_dir, pattern)
            model_files = glob.glob(full_pattern)
            models.extend([os.path.basename(f) for f in model_files])
        
        models.sort()
        return models
    
    def get_model_path(self, model_name: str) -> str:
        """Zwraca pełną ścieżkę do wybranego modelu"""
        return os.path.join(self.models_dir, model_name)
    
    def model_exists(self, model_name: str) -> bool:
        """Sprawdza czy model istnieje"""
        if not model_name or model_name == "No models found":
            return False
        model_path = self.get_model_path(model_name)
        return os.path.exists(model_path)
    
    def get_model_info(self, model_name: str) -> Dict:
        """Zwraca informacje o modelu"""
        model_path = self.get_model_path(model_name)
        info = {
            'name': model_name,
            'path': model_path,
            'size': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            'is_sdxl': self._is_sdxl_model(model_name),
            'is_v_prediction': VPredictionDetector.is_v_prediction_model(model_name), # Użycie scentralizowanego detektora
            'extension': os.path.splitext(model_name)[1].lower()
        }
        
        size_gb = info['size'] / (1024 ** 3)
        info['size_formatted'] = f"{size_gb:.1f}GB" if size_gb >= 1 else f"{size_gb * 1024:.1f}MB"
        
        return info
    
    def _is_sdxl_model(self, model_name: str) -> bool:
        """Sprawdza czy model to SDXL na podstawie nazwy"""
        name_lower = model_name.lower()
        return any(keyword in name_lower for keyword in ['xl', 'sdxl', 'xlarge'])
    
    def get_models_with_info(self) -> List[Dict]:
        """Zwraca listę modeli z informacjami"""
        models = self.scan_models()
        return [self.get_model_info(model) for model in models]

# Globalna instancja managera modeli
model_manager = ModelManager()