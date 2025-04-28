import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.fft
import numpy as np
from PIL import Image

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
device = "mps"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to(device)

unet = pipe.unet
vae = pipe.vae
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

scheduler.set_timesteps(50)
latents = torch.randn(1, unet.config.in_channels, 15, 15)

latents = latents.to(device)

prompt1 = "a photo of a rabbit"
prompt2 = "a photo of an old man"

text_input1 = pipe.tokenizer(prompt1, return_tensors="pt").input_ids.to(device)
text_input2 = pipe.tokenizer(prompt2, return_tensors="pt").input_ids.to(device)

text_embed1 = pipe.text_encoder(text_input1)[0]
text_embed2 = pipe.text_encoder(text_input2)[0]

# Return Highest and Lowest Frequency Components
def signal_factorization(latents):
    # Perform 2D-FFT on Latents
    fft_latents = torch.fft.fft2(latents)

    # Shifts Low Frequency Components to Center
    fshift = torch.fft.fftshift(fft_latents)

    # Shape of FFT, Define Center, and Initialize Mask
    _, _, H, W = fshift.shape
    center_h, center_w = H // 2, W // 2
    mask = torch.zeros_like(fshift, dtype=torch.bool)

    # Generates a Low-Frequency Mask
    r = 16
    for i in range(H):
      for j in range(W):
        if (i - center_h)**2 + (j - center_w)**2 < r**2:
          mask[:, :, i, j] = True

    # Initialize Low and High Frequency Components
    low_freq = torch.zeros_like(fshift)
    high_freq = torch.zeros_like(fshift)

    # Apply Mask to FFT (Low Freq gets Center, High Freq gets Rest)
    low_freq[mask] = fshift[mask]
    high_freq[~mask] = fshift[~mask]

    # Inverse Shift and Inverse FFT (Reverse Process of FFT to Get Images)
    low = torch.real(torch.fft.ifft2(torch.fft.ifftshift(low_freq), norm="ortho"))
    high = torch.real(torch.fft.ifft2(torch.fft.ifftshift(high_freq), norm="ortho"))

    return low, high

with torch.no_grad():
  for t in scheduler.timesteps:
      # Factorize the Latents
      low, high = signal_factorization(latents)

      # Get the noise predictions from the UNet
      noise_pred_low = unet(low, t, encoder_hidden_states=text_embed1).sample
      noise_pred_high = unet(high, t, encoder_hidden_states=text_embed2).sample

      # Combine the noise predictions
      noise_pred = noise_pred_low + noise_pred_high

      # Denoise the latent
      latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode the latents to an image
latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = image = vae.decode(latents).sample

# Save Image
image = (image / 2 + 0.5).clamp(0, 1)
image = image.squeeze(0) 
image = image.cpu().permute(1, 2, 0).numpy()
image = (image * 255).astype(np.uint8)
image = Image.fromarray(image)
image.save("factorized_image.png")