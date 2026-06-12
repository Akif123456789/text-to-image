from huggingface_hub import login

# Replace "YOUR_HF_TOKEN_HERE" with your actual HF token
login("")#by hugging face you can get the token 
from diffusers import StableDiffusionPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"#always use gpu 

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)


prompt = "art a cat"
image = pipe(prompt).images[0]
image