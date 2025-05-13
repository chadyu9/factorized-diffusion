# Factorized Diffusion

A re-implementation of "Factorized Diffusion" by Geng, Park, and Owens (2025), focusing on creating perceptual illusions through controlled image generation using diffusion models.

## Introduction

This project reimplements the Factorized Diffusion technique, which enables the creation of hybrid images that change appearance based on viewing distance. The implementation leverages the DeepFloyd IF diffusion model to factorize noise estimates during the denoising process, creating compelling visual illusions without requiring specialized fine-tuning.

## Chosen Result

We focused on reproducing the hybrid image generation results from the original paper, specifically implementing three types of factorizations:
- Spatial frequency hybrids (high/low frequency components)
- Color hybrids (grayscale/color components)
- Motion blur hybrids (blurred/residual components)

![Example Results](code/if_stage_I_hybrid.png)

## GitHub Contents

- `code/`: Contains the main implementation files
  - `main.py`: Core implementation using DeepFloyd IF model
  - `factorize_helpers.py`: Helper functions for different factorization techniques
- `results/`: Generated images organized by factorization type
  - `spatial/`: Frequency-based hybrid images
  - `color/`: Color-based hybrid images
  - `motion/`: Motion blur-based hybrid images

## Re-implementation Details

- **Model**: DeepFloyd IF (IF-I-M-v1.0 and IF-II-M-v1.0)
- **Framework**: PyTorch with Hugging Face Diffusers
- **Key Modifications**:
  - Implemented factorization in pixel space rather than latent space
  - Used CPU-based inference due to GPU memory constraints
  - Simplified the pipeline to focus on core factorization techniques

## Reproduction Steps

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the required models:
```bash
huggingface-cli download DeepFloyd/IF-I-M-v1.0
huggingface-cli download DeepFloyd/IF-II-M-v1.0
```

3. Generate hybrid images:
```bash
python code/main.py --prompt1 "A photo of houseplants" --prompt2 "A photo of Marilyn Monroe" --factorization spatial
```

**Note**: The first prompt corresponds to high frequency/grayscale/blurry components, while the second prompt corresponds to low frequency/color/non-blurry components.

## Results/Insights

Our implementation successfully reproduces the core functionality of Factorized Diffusion, with particularly strong results in spatial frequency factorization. The quality of color and motion blur factorizations is slightly lower than the original paper, likely due to computational constraints and model differences.

## Conclusion

This re-implementation demonstrates that perceptual factors can be explicitly manipulated in diffusion models through noise decomposition, enabling more controllable and interpretable image synthesis. The project highlights how existing generative models can be leveraged creatively without requiring specialized training.

## References

1. Geng, Park, and Owens (2025) - "Factorized Diffusion"
2. DeepFloyd IF - https://huggingface.co/DeepFloyd/IF-I-M-v1.0

## Acknowledgements

This project was completed as part of Cornell CS 5782 (Spring 2025). We thank the course staff and our peers for their valuable feedback and support throughout the implementation process.