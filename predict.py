from typing import List, Optional
from cog import BasePredictor, Input, Path
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

CACHE_DIR = 'weights'

BASE_MODEL_NAME = 'stabilityai/stable-diffusion-xl-base-1.0'
REFINER_MODEL_NAME = 'stabilityai/stable-diffusion-xl-refiner-1.0'

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=CACHE_DIR
        )
        self.model.to(self.device)
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=CACHE_DIR
        )
        self.refiner.to(self.device)

    def predict(
        self,
        prompt: str = Input(description=f"Text prompt to send to the model."),
        n_steps: int = Input(
            description="Maximum number of steps",
            default=40
        ),
        high_noise_frac: float = Input(
            description="Steps repartition over experts",
            default=0.8
        )
    ) -> Path:
        # run both experts
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]
        
        output_path = f"/tmp/out.png"
        image.save(output_path)
        
        return Path(output_path)