from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
    DDPMScheduler
)
from typing import Dict, Any

class SchedulerManager:
    """Manager do zarządzania schedulerami z obsługą V-Prediction"""
    
    # Rozszerzona mapa schedulerów
    SCHEDULER_CONFIGS = {
        "euler": {
            "class": EulerDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "euler_a": {
            "class": EulerAncestralDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "dpm_++_2m": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "dpm_++_2s_a": {
            "class": DPMSolverSinglestepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "dpm_++_sde": {
            "class": DPMSolverMultistepScheduler,
            "repo": "runwayml/stable-diffusion-v1-5",
            "kwargs": {"algorithm_type": "sde-dpmsolver++"}
        },
        "lms": {
            "class": LMSDiscreteScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "ddim": {
            "class": DDIMScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        },
        "pndm": {
            "class": PNDMScheduler,
            "repo": "runwayml/stable-diffusion-v1-5"
        }
    }
    
    @classmethod
    def create_scheduler(cls, 
                        scheduler_type: str, 
                        v_prediction: bool = False,
                        is_sdxl: bool = False) -> Any:
        """
        Tworzy scheduler z odpowiednim prediction_type
        """
        scheduler_type = scheduler_type.lower()
        if scheduler_type not in cls.SCHEDULER_CONFIGS:
            scheduler_type = "euler" # Domyślny fallback
        
        config = cls.SCHEDULER_CONFIGS[scheduler_type]
        SchedulerClass = config["class"]
        
        prediction_type = "v_prediction" if v_prediction else "epsilon"
        
        repo_id = "stabilityai/stable-diffusion-xl-base-1.0" if is_sdxl else config["repo"]

        # Dodatkowe argumenty dla niektórych schedulerów
        kwargs = config.get("kwargs", {})
        
        return SchedulerClass.from_pretrained(
            repo_id,
            subfolder="scheduler",
            prediction_type=prediction_type,
            **kwargs
        )
    
    @classmethod
    def get_available_schedulers(cls) -> list:
        """Zwraca listę dostępnych schedulerów"""
        return sorted(list(cls.SCHEDULER_CONFIGS.keys()))