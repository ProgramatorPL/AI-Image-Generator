import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

class BatchGenerator:
    """Klasa do zarządzania generowaniem batchy obrazów"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.total_generated = 0
        self.batch_history = []
        
    def setup_batch_directory(self, batch_name: Optional[str] = None) -> str:
        """Przygotowuje katalog dla batcha"""
        today = datetime.now().strftime("%Y-%m-%d")
        base_dir = os.path.join(self.output_dir, today)
        
        if batch_name:
            batch_dir = os.path.join(base_dir, batch_name)
        else:
            # Automatyczna nazwa batcha z timestampem
            timestamp = datetime.now().strftime("%H-%M-%S")
            batch_dir = os.path.join(base_dir, f"batch_{timestamp}")
        
        os.makedirs(batch_dir, exist_ok=True)
        return batch_dir
    
    def generate_batch_name(self, prompt: str, batch_size: int) -> str:
        """Generuje nazwę batcha na podstawie prompta"""
        # Skróć prompt do pierwszych 3 słów
        words = prompt.split()[:3]
        short_prompt = "_".join(words).lower()[:20]
        
        timestamp = datetime.now().strftime("%H%M")
        return f"{short_prompt}_{batch_size}x_{timestamp}"
    
    def get_batch_progress(self, current: int, total: int) -> Dict:
        """Zwraca informacje o postępie batcha"""
        percentage = (current / total) * 100
        return {
            'current': current,
            'total': total,
            'percentage': percentage,
            'remaining': total - current,
            'estimated_time': self.estimate_remaining_time(current, total)
        }
    
    def estimate_remaining_time(self, current: int, total: int, 
                              avg_time_per_image: float = 10.0) -> str:
        """Szacuje pozostały czas batcha"""
        remaining_images = total - current
        remaining_seconds = remaining_images * avg_time_per_image
        
        if remaining_seconds < 60:
            return f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            return f"{int(remaining_seconds / 60)}m {int(remaining_seconds % 60)}s"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def record_batch(self, prompt: str, batch_size: int, successful: int, 
                    output_dir: str, parameters: Dict):
        """Zapisuje informacje o wykonanym batchu"""
        batch_info = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'batch_size': batch_size,
            'successful': successful,
            'output_dir': output_dir,
            'parameters': parameters,
            'duration': parameters.get('total_time', 0)
        }
        
        self.batch_history.append(batch_info)
        self.total_generated += successful
        
        # Zapisz historię do pliku
        self.save_batch_history()
        
        return batch_info
    
    def save_batch_history(self):
        """Zapisuje historię batchy do pliku"""
        history_file = os.path.join(self.output_dir, "batch_history.json")
        try:
            import json
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.batch_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving batch history: {e}")
    
    def get_batch_summary(self) -> Dict:
        """Zwraca podsumowanie wszystkich batchy"""
        total_batches = len(self.batch_history)
        total_images = sum(batch['successful'] for batch in self.batch_history)
        
        return {
            'total_batches': total_batches,
            'total_images': total_images,
            'average_per_batch': total_images / total_batches if total_batches > 0 else 0,
            'last_batch': self.batch_history[-1] if self.batch_history else None
        }

# Globalna instancja batch generatora
batch_generator = BatchGenerator()

def create_batch_directory(prompt: str, batch_size: int) -> str:
    """Tworzy katalog dla batcha i zwraca ścieżkę"""
    batch_name = batch_generator.generate_batch_name(prompt, batch_size)
    return batch_generator.setup_batch_directory(batch_name)

def format_progress(current: int, total: int) -> str:
    """Formatuje informację o postępie"""
    progress = batch_generator.get_batch_progress(current, total)
    return f"{current}/{total} ({progress['percentage']:.1f}%) - ETA: {progress['estimated_time']}"

if __name__ == "__main__":
    # Testowanie funkcji batch
    bg = BatchGenerator()
    test_dir = bg.setup_batch_directory("test_batch")
    print(f"Batch directory: {test_dir}")
    
    for i in range(5):
        progress = bg.get_batch_progress(i, 5)
        print(f"Progress: {progress}")
        time.sleep(0.5)