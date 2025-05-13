import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline
import argparse

from factorize_helpers import motion_blur_factorization, color_factorization, spatial_frequency_factorization

# ────────────────────────────────────────────────────────────────
# Parse command line arguments
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Run factorized diffusion with custom prompts')
parser.add_argument('--prompt1', type=str, default="a photo of houseplants", 
                    help='First prompt for the factorization')
parser.add_argument('--prompt2', type=str, default="a photo of marilyn monroe", 
                    help='Second prompt for the factorization')
parser.add_argument('--factorization', type=str, choices=['motion', 'color', 'spatial'], 
                    default='spatial', help='Type of factorization to use')

args = parser.parse_args()

# ────────────────────────────────────────────────────────────────
# 0) Config
# ────────────────────────────────────────────────────────────────
device = "cpu"  # or "cuda"
num_steps = 50  # DDIM timesteps
H = W = 64  # IF-I produces 64×64
guidance = True  # use classifier-free guidance
guidance_scale = 5.0  # strength of guidance

# ────────────────────────────────────────────────────────────────
# 1) Custom Stage I: hybrid noise factorization -> 64×64
# ────────────────────────────────────────────────────────────────
# Load IF-I and move to device
stage1_pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-M-v1.0",
    token=True,  # authenticated token for gated repo
    torch_dtype=torch.float32,
).to(device)
print("Loaded stage 1 pipeline!")

stage1_unet = stage1_pipe.unet
stage1_sched = stage1_pipe.scheduler
stage1_sched.set_timesteps(num_steps)

# prompts
prompt1 = args.prompt1 #high freq,
prompt2 = args.prompt2

# build prompt embeddings (with CFG)
pe1, ne1 = stage1_pipe.encode_prompt(prompt1, do_classifier_free_guidance=guidance)
pe2, ne2 = stage1_pipe.encode_prompt(prompt2, do_classifier_free_guidance=guidance)
print("Done buildnig prompt embeddings.")

# Specify which hybrid factorization to use
hybrid_factorization = args.factorization

# initialize pixel-space noise
latents = torch.randn(1, stage1_unet.config.in_channels, H, W, device=device)
latents = latents * stage1_sched.init_noise_sigma

print(f"Beginning stage 1, generating image with... {hybrid_factorization} factorization, and prompts: {prompt1} and {prompt2}")
# denoising loop with correct guidance
with torch.no_grad():
    for t in stage1_sched.timesteps:
        print("Denoising timestep", t.item())

        # duplicate latents
        model_in = torch.cat([latents, latents], dim=0)
        model_in = stage1_sched.scale_model_input(model_in, t)

        # combine prompt embeddings
        prompt_embeds1 = torch.cat([ne1, pe1], dim=0)  # (neg, pos)
        prompt_embeds2 = torch.cat([ne2, pe2], dim=0)

        # UNet forward
        n1 = stage1_unet(model_in, t, encoder_hidden_states=prompt_embeds1).sample
        n2 = stage1_unet(model_in, t, encoder_hidden_states=prompt_embeds2).sample

        # split and apply guidance
        n1_uncond, n1_cond = n1.chunk(2)
        n2_uncond, n2_cond = n2.chunk(2)

        n1_guided = n1_uncond + guidance_scale * (n1_cond - n1_uncond)
        n2_guided = n2_uncond + guidance_scale * (n2_cond - n2_uncond)

        # hybrid composition
        if hybrid_factorization == "motion":
            blur, _ = motion_blur_factorization(n1_guided)
            _, residual = motion_blur_factorization(n2_guided)
            composite = blur + residual
        elif hybrid_factorization == "color":
            h, _ = color_factorization(n1_guided)
            _, l = color_factorization(n2_guided)
            composite = h + l
        elif hybrid_factorization == "spatial":
            g, _ = spatial_frequency_factorization(n1_guided, 1.0)
            _, c = spatial_frequency_factorization(n2_guided, 1.0)
            composite = g + c

        # denoising step
        latents = stage1_sched.step(composite, t, latents).prev_sample

# Stage I output: 64×64 hybrid image tensor
stage1_output = latents.clamp(-1, 1)  # ensure proper range before Stage II

# save Stage I result
img_i = (stage1_output / 2 + 0.5).clamp(0, 1)
arr_i = (img_i[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr_i).save("if_stage_I_hybrid.png")
print("Saved stage I hybrid image, as if_stage_I_hybrid.png!")

# ────────────────────────────────────────────────────────────────
# 2) Official Stage II: 64→256 super-resolution
# ────────────────────────────────────────────────────────────────
stage2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-M-v1.0", token=True, torch_dtype=torch.float32, text_encoder=None
).to(device)

with torch.no_grad():
    stage2_output = stage2(
        image=stage1_output,  # B×3×64×64
        prompt_embeds=pe1,  # only positive embedding
        negative_prompt_embeds=ne1,  # still using guided setup
        generator=torch.Generator(device).manual_seed(0),
        output_type="pt",
    ).images  # B×3×256×256

# save Stage II result
img_ii = (stage2_output / 2 + 0.5).clamp(0, 1)
print(type(img_ii))
arr_ii = (img_ii[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr_ii).save("if_stage_II.png")

# save the hybrid image
if hybrid_factorization == "motion":
    # save the motion blurred image
    blur_ii = motion_blur_factorization(img_ii)[0]
    arr_m = (blur_ii[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr_m).save("if_stage_II_motion.png")


print(
    "All stages complete: saved if_stage_I_hybrid.png, if_stage_II.png, if_stage_III.png"
)
