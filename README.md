# ControlNet with Canny Edge Detection

This repository implements **ControlNet** using **Canny edge detection** as a control signal to generate realistic images based on latent representations.

## Features
- **ControlNet Architecture**: Leverages a pretrained UNet with custom modifications to handle Canny edge hints.
- **Latent Input**: Dataset loads precomputed latent representations for training instead of raw images.
- **Canny Edge Control**: Uses Canny edge-detected images to guide the generation process.
