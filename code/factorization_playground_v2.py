import torch, torch.fft, numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler
import torch.nn.functional as F

# ────────────────────────────────────────────────────────────────
# 0) Config
# ────────────────────────────────────────────────────────────────
device = "mps"  # or "cuda"
num_steps = 50  # DDIM timesteps
H = W = 64  # IF-I produces 64×64
guidance = True  # use classifier-free guidance
guidance_scale = 5.0  # strength of guidance

# ────────────────────────────────────────────────────────────────
# 1) Custom Stage I: hybrid noise factorization -> 64×64
# ────────────────────────────────────────────────────────────────
# Load IF-I and move to device
print("here1")
stage1_pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-M-v1.0",
    token=True,  # authenticated token for gated repo
    torch_dtype=torch.float32,
).to(device)
print("here2")

stage1_unet = stage1_pipe.unet
stage1_sched = stage1_pipe.scheduler
stage1_sched.set_timesteps(num_steps)

# prompts
prompt1 = "a photo of an old man"
prompt2 = "a photo of a rabbit"

# build prompt embeddings (with CFG)
pe1, ne1 = stage1_pipe.encode_prompt(prompt1, do_classifier_free_guidance=guidance)
pe2, ne2 = stage1_pipe.encode_prompt(prompt2, do_classifier_free_guidance=guidance)
print("here3")


# helper to split noise into gray/color
def color_factorization(x):
    gray = x.mean(dim=1, keepdim=True)  # B×1×H×W
    color = x - gray  # B×C×H×W
    gray = gray.expand_as(x)  # no extra memory
    return gray, color


def spatial_frequency_factorization(
    x: torch.Tensor, sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose an image tensor into high and low spatial frequency components
    using a Gaussian blur with standard deviation sigma.

    Args:
        x (torch.Tensor): input tensor of shape (B, C, H, W)
        sigma (float): standard deviation for Gaussian kernel

    Returns:
        high (torch.Tensor): high-frequency component (x - G_sigma(x))
        low (torch.Tensor): low-frequency component (G_sigma(x))
    """
    # Approximate a Gaussian blur using repeated small blurs
    if sigma == 0:
        low = x
    else:
        # Size of the Gaussian kernel — typical choice: 6*sigma rounded up to odd
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, device=x.device) - kernel_size // 2
        gaussian_kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # Expand to 2D separable convolution
        gaussian_kernel_x = gaussian_kernel.view(1, 1, 1, -1)
        gaussian_kernel_y = gaussian_kernel.view(1, 1, -1, 1)

        # Apply depthwise (channelwise) convolution: first x, then y
        padding = kernel_size // 2
        x_blurred = F.conv2d(
            x,
            gaussian_kernel_x.expand(x.shape[1], -1, -1, -1),
            padding=(0, padding),
            groups=x.shape[1],
        )
        x_blurred = F.conv2d(
            x_blurred,
            gaussian_kernel_y.expand(x.shape[1], -1, -1, -1),
            padding=(padding, 0),
            groups=x.shape[1],
        )

        low = x_blurred

    high = x - low
    return high, low


# initialize pixel-space noise
latents = torch.randn(1, stage1_unet.config.in_channels, H, W, device=device)
latents = latents * stage1_sched.init_noise_sigma

print("here4")
# denoising loop with correct guidance
with torch.no_grad():
    for t in stage1_sched.timesteps:
        print("timestep", t.item())

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
arr_ii = (img_ii[0].cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
Image.fromarray(arr_ii).save("if_stage_II.png")

# ────────────────────────────────────────────────────────────────
# 3) Official Stage III: 256→1024 x4 upscaler
# ────────────────────────────────────────────────────────────────
# carry over safety modules from Stage I
safety = {
    "feature_extractor": stage1_pipe.feature_extractor,
    "safety_checker": stage1_pipe.safety_checker,
    "watermarker": stage1_pipe.watermarker,
}
stage3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    token=True,
    torch_dtype=torch.float32,
    **safety
).to(device)

with torch.no_grad():
    stage3_images = stage3(
        prompt=prompt1,  # final prompt
        image=stage2_output,  # B×3×256×256
        noise_level=100,  # default noise for x4 upscaling
        generator=torch.Generator(device).manual_seed(0),
    ).images  # a list of PIL Images

# save Stage III result
stage3_images[0].save("if_stage_III.png")

print(
    "All stages complete: saved if_stage_I_hybrid.png, if_stage_II.png, if_stage_III.png"
)
