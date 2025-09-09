from PIL import Image
import json
from datetime import datetime
from typing import Dict, Any, Optional
import torch

class MetadataWriter:
    """Klasa do zapisywania metadanych EXIF w obrazach"""
    
    EXIF_TAGS = {
        'user_comment': 37510,       # UserComment (dla formatu Civitai/A1111)
        'image_description': 270,   # ImageDescription (tutaj schowamy pełny JSON)
    }
    
    @classmethod
    def create_metadata(cls, 
                       prompt: str,
                       negative_prompt: str,
                       width: int,
                       height: int,
                       steps: int,
                       guidance_scale: float,
                       seed: Optional[int],
                       scheduler: str,
                       model_name: str,
                       v_prediction: bool,
                       batch_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Tworzy słownik z metadanymi dla wygenerowanego obrazu.
        """
        metadata = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': guidance_scale,
            'sampler': scheduler,
            'model': model_name,
            'v_prediction': v_prediction,
            'generation_date': datetime.now().isoformat(),
            'software': 'StableDiffusionGUI',
            'device': 'CUDA' if torch.cuda.is_available() else 'CPU'
        }
        
        if seed is not None:
            metadata['seed'] = seed
        
        if batch_index is not None:
            metadata['batch_index'] = batch_index
        
        return metadata

    @classmethod
    def metadata_to_civitai_string(cls, metadata: Dict[str, Any]) -> str:
        """
        Konwertuje metadane na string w formacie kompatybilnym z Civitai/A1111.
        """
        prompt_str = metadata.get('prompt', '')
        negative_prompt_str = f"Negative prompt: {metadata.get('negative_prompt', '')}"
        
        settings = [
            f"Steps: {metadata.get('steps', 'N/A')}",
            f"Sampler: {metadata.get('sampler', 'N/A')}",
            f"CFG scale: {metadata.get('cfg_scale', 'N/A')}",
            f"Seed: {metadata.get('seed', 'N/A')}",
            f"Size: {metadata.get('width', 'N/A')}x{metadata.get('height', 'N/A')}",
            f"Model: {metadata.get('model', 'N/A').split('.')[0]}", # Nazwa modelu bez rozszerzenia
        ]
        
        settings_str = ", ".join(settings)
        
        return f"{prompt_str}\n{negative_prompt_str}\n{settings_str}"

    @classmethod
    def add_metadata_to_image(cls, image: Image.Image, metadata: Dict[str, Any]) -> Image.Image:
        """
        Dodaje metadane EXIF do obrazu w dwóch formatach:
        1. Tekstowym dla Civitai w UserComment.
        2. Pełnym JSON w ImageDescription.
        """
        try:
            exif_data = image.getexif()

            # Format 1: String dla Civitai
            civitai_string = cls.metadata_to_civitai_string(metadata)
            charset_code = b'\x00\x00\x00\x00\x00\x00\x00\x00' # UNDEFINED dla UTF-8
            comment_bytes = civitai_string.encode('utf-8')
            exif_data[cls.EXIF_TAGS['user_comment']] = charset_code + comment_bytes

            # Format 2: Pełny JSON dla innych narzędzi
            json_string = json.dumps(metadata, ensure_ascii=False)
            exif_data[cls.EXIF_TAGS['image_description']] = json_string

            # Zapisz metadane w kopii obrazu
            image_with_exif = image.copy()
            image_with_exif.info["exif"] = exif_data.tobytes()
            
            return image_with_exif
        except Exception as e:
            print(f"Warning: Could not write EXIF data. Error: {e}")
            return image