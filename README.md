# Egyptian Relief Image Colorization using Deep Learning

## Overview

This project explores automatic colorization of ancient Egyptian relief images using deep learning.

Ancient reliefs appear mostly grayscale today due to pigment degradation.
This project reconstructs plausible color representations using a combination of segmentation, hint generation, and transformer-based colorization.

The system is built using PyTorch and is based on the **IColoriT Vision Transformer model**.

---

## Method Pipeline

The proposed system follows a multi-stage pipeline:

1. **Pigment Segmentation** using a ResNet50-UNet model
2. **Automatic Hint Generation** from segmented pigment regions
3. **Image Colorization** using the IColoriT transformer model

### Pipeline Overview

Input Relief Image
↓
Pigment Segmentation (ResNet50-UNet)
↓
Hint Generation
↓
IColoriT Transformer Colorization
↓
Final Colorized Relief Image

---

## Repository Structure

color2/iColoriT/ → Transformer-based colorization implementation

data/test_images → Example grayscale relief images

data/test_masks → Segmentation masks

results → Example colorized outputs

---

## Example Results

## Example Result

Below is an example showing the full pipeline from grayscale relief image to final colorized output.

| Input Image | Pigment Mask | Colorized Output |
|-------------|--------------|------------------|
| ![](color2/iColoriT/data/test_images/egypt_04.jpg) | ![](color2/iColoriT/data/test_masks/egypt_04.png) | ![](color2/iColoriT/results/egypt_04_colorized.png) |
## Technologies Used

Python
PyTorch
OpenCV
Vision Transformers

---

## Thesis

This repository contains the implementation developed for the MSc thesis:

**“Colorization of Egyptian Relief Images using Deep Learning”**

University of Würzburg  
Author: Sapana Gupta
