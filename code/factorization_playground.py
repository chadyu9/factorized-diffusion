# Color-Factorized Diffusion with DeepFloyd IF (CUDA-safe version)
# ------------------------------------------------------------
# Same functionality as before, now avoids CUDA-specific calls when the
# installed PyTorch doesn’t support CUDA.
# ------------------------------------------------------------

from __future__ import annotations
import importlib
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_ID = "DeepFloyd/IF-I-M-v1.0"  # 64×64 base model
PROMPT_LUMA = "bee"
PROMPT_CHROMA = "barn"
NUM_STEPS = 50
OUTPUT_PATH = "color_factorized_image.png"

# ------------------------------------------------------------
# Device & dtype
# ------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS doesn’t support fp16 well
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Running on {DEVICE} (dtype={DTYPE})")

# ------------------------------------------------------------
# Load DeepFloyd IF stage-1 pipeline
# ------------------------------------------------------------
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    variant="fp16" if DTYPE == torch.float16 else None,
    torch_dtype=DTYPE,
)

# Optional memory-savers ------------------------------------------------------
if importlib.util.find_spec("xformers") is not None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers found – enabled memory-efficient attention.")
    except Exception as e:
        print(f"xformers present but couldn’t be enabled ({e}). Continuing without it.")
else:
    print(
        "xformers not installed – running with standard attention. To install: pip install xformers"
    )

# Move pipeline to the chosen device and optionally enable CPU-offload
if DEVICE == "cuda":
    pipe.to("cuda")
    # Only call CPU-offload when CUDA is actually available
    pipe.enable_model_cpu_offload()
else:
    pipe.to(DEVICE)

unet = pipe.unet
scheduler = pipe.scheduler

# ------------------------------------------------------------
# Encode prompts (on correct device)
# ------------------------------------------------------------
with torch.inference_mode():
    text_ids_luma = pipe.tokenizer(PROMPT_LUMA, return_tensors="pt").input_ids.to(
        DEVICE
    )
    text_ids_chroma = pipe.tokenizer(PROMPT_CHROMA, return_tensors="pt").input_ids.to(
        DEVICE
    )
    text_embed_luma = pipe.text_encoder(text_ids_luma)[0].to(DEVICE)
    text_embed_chroma = pipe.text_encoder(text_ids_chroma)[0].to(DEVICE)


# ------------------------------------------------------------
# Helper: luminance–chroma factorization
# ------------------------------------------------------------
@torch.no_grad()
def color_factorization(x: torch.Tensor):
    rgb = x[:, :3]
    extra = x[:, 3:] if x.shape[1] > 3 else None
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y_rgb = torch.cat([y, y, y], dim=1)
    chroma = rgb - y_rgb
    if extra is not None:
        y_rgb = torch.cat([y_rgb, extra], dim=1)
        chroma = torch.cat([chroma, torch.zeros_like(extra)], dim=1)
    return y_rgb, chroma


# ------------------------------------------------------------
# Denoising loop
# ------------------------------------------------------------
@torch.no_grad()
def run_color_factorized_diffusion():
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    x = torch.randn(
        1,
        unet.config.in_channels,
        unet.config.sample_size,
        unet.config.sample_size,
        device=DEVICE,
        dtype=DTYPE,
    )
    for t in scheduler.timesteps:
        y_comp, c_comp = color_factorization(x)
        pred_y = unet(y_comp, t, encoder_hidden_states=text_embed_luma).sample
        pred_c = unet(c_comp, t, encoder_hidden_states=text_embed_chroma).sample
        x = scheduler.step(pred_y + pred_c, t, x).prev_sample
    return x[:, :3]


# ------------------------------------------------------------
# Decode & save result
# ------------------------------------------------------------
final_rgb = run_color_factorized_diffusion()
image = (final_rgb / 2 + 0.5).clamp(0, 1)
image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
Image.fromarray((image * 255).astype(np.uint8)).save(OUTPUT_PATH)
print(f"Image saved to {OUTPUT_PATH}")
