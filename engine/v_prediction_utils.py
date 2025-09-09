import re
from typing import List, Dict

class VPredictionDetector:
    """Klasa do wykrywania modeli V-Prediction"""
    
    # Lista słów kluczowych wskazujących na V-Prediction
    V_PREDICTION_KEYWORDS = [
        'v-pred', 'vpred', 'v_pred', 'vp',
        'vector', 'velocity', 'v prediction',
        '768-v', '2m-v', '2v', 'v2.0', 'v20',
        'v-parameterization', 'v_param',
        'v-parameter', 'velocity-pred'
    ]
    
    # Wzorce regex dla V-Prediction
    V_PREDICTION_PATTERNS = [
        r'v[-_]?pred',
        r'v\d+',
        r'vp\d*',
        r'velocity',
        r'vector',
        r'parameterization'
    ]
    
    @classmethod
    def is_v_prediction_model(cls, model_name: str) -> bool:
        """
        Sprawdza czy model wymaga V-Prediction na podstawie nazwy
        
        Args:
            model_name: Nazwa pliku modelu
            
        Returns:
            bool: True jeśli model wymaga V-Prediction
        """
        name_lower = model_name.lower()
        
        # Sprawdź słowa kluczowe
        for keyword in cls.V_PREDICTION_KEYWORDS:
            if keyword in name_lower:
                return True
        
        # Sprawdź wzorce regex
        for pattern in cls.V_PREDICTION_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        
        return False
    
    @classmethod
    def suggest_v_prediction_models(cls, model_list: List[str]) -> List[Dict]:
        """
        Sugeruje które modele mogą wymagać V-Prediction
        
        Args:
            model_list: Lista nazw modeli
            
        Returns:
            List[Dict]: Lista modeli z sugestiami
        """
        suggestions = []
        
        for model_name in model_list:
            is_v_pred = cls.is_v_prediction_model(model_name)
            suggestions.append({
                'name': model_name,
                'suggest_v_prediction': is_v_pred,
                'confidence': 'high' if is_v_pred else 'low'
            })
        
        return suggestions
    
    @classmethod
    def get_v_prediction_config(cls) -> Dict:
        """
        Zwraca konfigurację dla V-Prediction
        
        Returns:
            Dict: Konfiguracja prediction_type
        """
        return {
            'prediction_type': 'v_prediction',
            'recommended_guidance': 4.5,
            'recommended_steps': 20,
            'notes': 'Use V-Prediction for models that require velocity parameterization'
        }

# Przykłady modeli V-Prediction
V_PREDICTION_EXAMPLES = [
    "model-vprediction.safetensors",
    "model-vp.ckpt", 
    "model-v20.safetensors",
    "model-768-v.ckpt",
    "velocity-prediction-model.safetensors",
    "vectorized-model.ckpt"
]

def test_v_prediction_detection():
    """Testuje detekcję modeli V-Prediction"""
    detector = VPredictionDetector()
    
    test_models = V_PREDICTION_EXAMPLES + [
        "normal-model.safetensors",
        "standard.ckpt",
        "model-1.5.ckpt"
    ]
    
    print("V-Prediction Detection Test:")
    print("-" * 40)
    
    for model in test_models:
        is_v_pred = detector.is_v_prediction_model(model)
        status = "V-Prediction" if is_v_pred else "Standard"
        print(f"{model:30} -> {status}")

if __name__ == "__main__":
    test_v_prediction_detection()