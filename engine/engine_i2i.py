import torch
from PIL import Image
from compel import Compel
from copy import deepcopy

class ImageToImageEngine:
    """Silnik odpowiedzialny za operacje na obrazach (Image-to-Image)."""

    def process(self, mode: str, pipeline, compel: Compel, is_sdxl: bool, original_image: Image.Image, metadata: dict, device: str, dtype: torch.dtype):
        """
        Przetwarza obraz wejściowy w trybie 'upscale' lub 'variation'.

        Args:
            mode (str): Tryb operacji ('upscale' lub 'variation').
            pipeline: Aktywny potok (pipeline) img2img.
            compel: Instancja Compel do przetwarzania promptów.
            is_sdxl (bool): Flaga modelu SDXL.
            original_image (Image.Image): Obraz wejściowy.
            metadata (dict): Metadane oryginalnego obrazu.

        Returns:
            Tuple[Image.Image, dict]: Przetworzony obraz i jego nowe metadane.
        """
        new_metadata = deepcopy(metadata)
        prompt = metadata['prompt']
        negative_prompt = metadata['negative_prompt']
        seed = metadata['seed']
        generator = torch.Generator(device=device).manual_seed(seed)
        
        params = {
            "generator": generator,
            "guidance_scale": metadata['cfg_scale'],
        }

        # Przetwarzanie promptów z Compel
        if compel:
            if is_sdxl:
                prompt_embeds, pooled_prompt_embeds = compel(prompt)
                negative_prompt_embeds, negative_pooled_prompt_embeds = compel(negative_prompt)
                prompt_embeds, negative_prompt_embeds = self._pad_embeddings(prompt_embeds, negative_prompt_embeds, device, dtype)
                params.update({"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds, "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds})
            else:
                prompt_embeds = compel(prompt)
                negative_prompt_embeds = compel(negative_prompt)
                prompt_embeds, negative_prompt_embeds = self._pad_embeddings(prompt_embeds, negative_prompt_embeds, device, dtype)
                params.update({"prompt_embeds": prompt_embeds, "negative_prompt_embeds": negative_prompt_embeds})
        else:
            params.update({"prompt": prompt, "negative_prompt": negative_prompt})

        # Ustawienia specyficzne dla trybu
        if mode == 'upscale':
            new_width = int(original_image.width * 1.5)
            new_height = int(original_image.height * 1.5)
            input_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            params.update({"image": input_image, "num_inference_steps": 18, "strength": 0.4})
            new_metadata.update({'upscaled_from_seed': seed, 'upscale_factor': 1.5, 'width': new_width, 'height': new_height})
        
        elif mode == 'variation':
            params.update({"image": original_image, "num_inference_steps": metadata['steps'], "strength": 0.55})
            new_metadata.update({'variation_from_seed': seed, 'variation_strength': 0.55})

        # Generowanie
        new_image = pipeline(**params).images[0]
        
        return new_image, new_metadata

    def _pad_embeddings(self, prompt_embeds, negative_prompt_embeds, device: str, dtype: torch.dtype):
        """Wyrównuje długość embeddingów."""
        max_length = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])

        def pad_tensor(tensor, length):
            if tensor.shape[1] < length:
                padding_size = length - tensor.shape[1]
                padding = torch.zeros((1, padding_size, tensor.shape[2]), device=device, dtype=dtype)
                return torch.cat([tensor, padding], dim=1)
            return tensor

        prompt_embeds = pad_tensor(prompt_embeds, max_length)
        negative_prompt_embeds = pad_tensor(negative_prompt_embeds, max_length)
        return prompt_embeds, negative_prompt_embeds