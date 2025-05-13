# factorized-diffusion
Final project for Spring 2025 Cornell CS 4782 - an implementation of Factorized Diffusion

To download the model that we are using for this project, run ```huggingface-cli download DeepFloyd/IF-I-M-v1.0``` and ```huggingface-cli download DeepFloyd/IF-II-M-v1.0``` on your local terminal.

After downloading the model, you can generate hybrid images by running (For example): 
``` python code/main.py --prompt1 "A photo of houseplants" --prompt2 'A photo of Marilyn Monroe' --factorization 'spatial' ```

Prompt one corresponds to high frequency, grayscale, blurry, and prompt 2 corresponds to low frequency, color and not-blurry.