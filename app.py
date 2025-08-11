import torch
from diffusers import FluxPipeline
from typing import Optional
import base64, io
from pydantic import BaseModel, Field
import inferless
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="A frog holding a sign that says hello world")
    guidance_scale: Optional[float] = 4.5
    height: Optional[int] = 1024
    width: Optional[int] = 1024

@inferless.response
class ResponseObjects(BaseModel):
    image_base64: str = Field(default="Test Output")

class InferlessPythonModel:
    def initialize(self):
        self.pipe = pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16).to("cuda")

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        image = self.pipe(
                    inputs.prompt,
                    height=inputs.height,
                    width=inputs.width,
                    guidance_scale=inputs.guidance_scale,
        ).images[0]
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return ResponseObjects(image_base64=encoded)

    def finalize(self):
        self.pipe = None
