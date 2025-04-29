# -----------------------------------------------------------
#  color_hybrid_if.py
# -----------------------------------------------------------
import os, torch, torch.mps
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil

# ─────────── prompts & params ───────────
prompt_gray = "a black-and-white painting of a barn"
prompt_color = "a painting of a bumblebee"
steps_stage1 = 50
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
# ────────────────────────────────────────

# ---------- device ----------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.float16 if device == "cuda" else torch.float32
print(f"Running on {device}  (dtype={dtype})")

# ========== Stage-1 : pixel-space diffusion ==========
stage_1 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-M-v1.0",
    variant="fp16" if dtype is torch.float16 else None,
    torch_dtype=dtype,
)
stage_1.to(device)
stage_1.enable_attention_slicing()
try:
    stage_1.enable_xformers_memory_efficient_attention()
except (ModuleNotFoundError, ValueError):
    print("xformers not installed – stage-1 uses standard attention.")
if torch.cuda.is_available():
    stage_1.enable_model_cpu_offload()


# ---- helper: pad to full length so all embeds share the same shape ----
def embed(text: str):
    ids = stage_1.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        max_length=stage_1.tokenizer.model_max_length,
        truncation=True,
    ).input_ids.to(device)
    return stage_1.text_encoder(ids)[0]  # (1, L, 4096)


text_emb_gray = embed(prompt_gray)
text_emb_color = embed(prompt_color)
neg_emb_blank = embed("")  # same shape as positive

# ---- factorized sampling ----
stage_1.scheduler.set_timesteps(steps_stage1)
latents = torch.randn(
    1, stage_1.unet.config.in_channels, 64, 64, device=device, dtype=dtype
)

with torch.inference_mode():
    for t in stage_1.scheduler.timesteps:
        eps_g = stage_1.unet(latents, t, encoder_hidden_states=text_emb_gray).sample
        eps_c = stage_1.unet(latents, t, encoder_hidden_states=text_emb_color).sample
        f_gray = eps_g.mean(dim=1, keepdim=True).expand_as(eps_g)
        f_color = eps_c - eps_c.mean(dim=1, keepdim=True)
        latents = stage_1.scheduler.step(f_gray + f_color, t, latents).prev_sample

rgb64 = (latents[:, :3].clamp(-1, 1) + 1) / 2
image_64 = pt_to_pil(rgb64)[0]
image_64.save(f"{out_dir}/stage1_64.png")
print("stage-1 saved ➜ outputs/stage1_64.png")

if device == "mps":
    torch.mps.empty_cache()

# ========== Stage-2 : 256 × 256 up-sampler ==========
print("\nAttempting Stage-2 (256×256 up-sampler)…")
try:
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0",
        text_encoder=None,
        variant="fp16" if dtype is torch.float16 else None,
        torch_dtype=dtype,
    )
    stage_2.to(device)
    stage_2.enable_attention_slicing()
    if torch.cuda.is_available():
        stage_2.enable_model_cpu_offload()
    try:
        stage_2.enable_xformers_memory_efficient_attention()
    except (ModuleNotFoundError, ValueError):
        print("xformers not installed – stage-2 uses standard attention.")

    image_256 = stage_2(
        image=image_64,
        prompt_embeds=text_emb_gray,
        negative_prompt_embeds=neg_emb_blank,  # same shape
        num_inference_steps=30,
        output_type="pil",
    ).images[0]
    image_256.save(f"{out_dir}/stage2_256.png")
    print("stage-2 saved ➜ outputs/stage2_256.png")
except Exception as e:
    print(f"Stage-2 skipped ({e}).")
    image_256 = None

# ========== Stage-3 : 1024 × 1024 up-scaler ==========
if image_256 is not None:
    print("\nAttempting Stage-3 (x4 up-scaler)…")
    try:
        safety = {
            "feature_extractor": stage_1.feature_extractor,
            "safety_checker": stage_1.safety_checker,
            "watermarker": stage_1.watermarker,
        }
        stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=dtype, **safety
        )
        stage_3.to(device)
        stage_3.enable_attention_slicing()
        if torch.cuda.is_available():
            stage_3.enable_model_cpu_offload()
        try:
            stage_3.enable_xformers_memory_efficient_attention()
        except (ModuleNotFoundError, ValueError):
            print("xformers not installed – stage-3 uses standard attention.")

        image_1k = stage_3(
            image=image_256,
            prompt=prompt_color,
            num_inference_steps=40,
            guidance_scale=0,
        ).images[0]
        image_1k.save(f"{out_dir}/stage3_1024.png")
        print("stage-3 saved ➜ outputs/stage3_1024.png")
    except Exception as e:
        print(f"Stage-3 skipped ({e}).")
