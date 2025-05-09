# helper to split noise into gray/color
import torch
import torch.nn.functional as F


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


def motion_blur_factorization(x: torch.Tensor, blur_length: int = 7):
    """
    Factorizes image(s) into motion blurred and residual components using a diagonal motion blur kernel.

    Args:
        x (torch.Tensor): Input image tensor of shape (C, H, W) or (B, C, H, W).
        blur_length (int): Size of the square diagonal blur kernel. Default is 7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (motion_blurred, residual), both same shape as input.
    """

    # Ensure batch dimension
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Shape: (1, C, H, W)

    B, C, H, W = x.shape

    # Create a diagonal blur kernel
    kernel = torch.zeros(1, 1, blur_length, blur_length, device=x.device)
    for i in range(blur_length):
        kernel[0, 0, i, i] = 1.0 / blur_length  # normalized diagonal

    # Expand for depthwise convolution: (C, 1, kH, kW)
    kernel = kernel.expand(C, 1, blur_length, blur_length)

    # Apply depthwise convolution
    motion_blurred = F.conv2d(x, kernel, padding="same", groups=C)

    residual = x - motion_blurred

    return motion_blurred, residual