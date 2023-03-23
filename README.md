# loopback_scaler
Automatic1111 python script to enhance image quality

## Overview
The Loopback Scaler is an Automatic1111 Python script that enhances image resolution and quality using an iterative process. The code takes an input image and performs a series of image processing steps, including denoising, resizing, and applying various filters. The algorithm loops through these steps multiple times, with user-defined parameters controlling how the image evolves at each iteration. The result is an improved image, often with more detail, better color balance, and fewer artifacts than the original.

## Key features
- **Iterative enhancement**: The script processes the input image in several loops, with each loop increasing the resolution and refining the image quality. The image result from one loop is then inserted as the input image for the next loop which continually builds on what has been created.
- **Denoise Change**: The denoising strength can be adjusted for each loop, allowing users to strike a balance between preserving details and reducing artifacts.
- **Adaptive change**: The script adjusts the amount of resolution increase per loop based on the average intensity of the input image. This helps to produce more natural-looking results.
- **Image filters**: Users can apply various PIL Image Filters to the final image, including detail enhancement, blur, smooth, and contour filters.
- **Image adjustments**: The script provides sliders to fine-tune the sharpness, brightness, color, and contrast of the final image.

Recommended settings for img2img processing are provided in the script, including resize mode, sampling method, width/height, CFG scale, denoising strength, and seed.

Please note that the performance of the Loopback Scaler depends on the GPU, input image, and user-defined parameters. Experimenting with different settings can help you achieve the desired results.

## Tips, Tricks, and Advice
- Do **NOT** expect to recreate images with prompts using this method.
- You can start from txt2img with a prompt. Generate your image and then send it over to img2img. When creating images for this process, shoot for lower resolution images (512x768, 340x512, etc.)
- **ALWAYS** have a prompt in your img2img tab when doing this process, unless you are interested in creating chaos :D. Your results will usually be poor, but you CAN put a different prompt in img2img than what you created the source image with. Pretty interesting results come from this method.
- When using models that require VAE, keep the number of loops lower than normal because it will cause the image to fade each iteration. Luckily you can add Color and Sharpness back in with the PIL enhancements if you need.
- Don't set your maximum Width/Height higher than what you can normally generate. This script is not an upscaler model and isn't intended to make giant images. It is intended to give you detailed quality images that you can send to an upscaler.
- Once installed, there is an Info panel at the bottom of the script interface to help you understand the settings and what they do.

## Manual Installation
1. Unzip the `loopback_scaler.py` script.
2. Move the script to the `\stable-diffusion-webui\scripts` folder.
3. Close the Automatic1111 webui console window.
4. Relaunch the webui by running the `webui-user.bat` file.
5. Open your web browser and navigate to the Automatic1111 page or refresh the page if it's already open.

# Settings Guide

## Loops
The number of times the script will inference your image and increase the resolution in increments. The amount the resolution is increased each loop is determined by this number and the maximum image width/height. The more loops, the more chances of your image picking up more detail, but also artifacts. 4 to 10 is what I find to work best, but you may like more or less.

## Denoise change
This setting will increase or decrease the denoising strength every loop. A higher value will increase the denoising strength, while a lower value will decrease it. A setting of 1 keeps the denoising strength as it is set on the img2img settings.

## Adaptive change
This setting changes the amount of resolution increase per loop, keeping the changes from being linear. The higher the value the more significant the resolution changes toward the end of the looping.

## Maximum Image Width/Height
These parameters set the maximum width and height of the final image. Always start with an image smaller than these dimensions. The smaller you start, the more impressive the results. I usually start at either 340x512 or 512x768.

## Detail, Blur, Smooth, Contour
These parameters are checkboxes that apply a PIL Image Filter to the final image.

## Sharpness, Brightness, Color, Contrast
These parameters are sliders that adjust the sharpness, brightness, color, and contrast of the image. 1 will result in no adjustments, less than one reduces these settings for the final image and greater than 1 increases these settings.

## Img2Img Settings
I recommend creating an image with txt2img and then sending the result to img2img with the prompt and settings. For this script, I use these settings:

- **Resize mode**: Crop and resize
- **Sampling method**: DDIM
- **Sampling steps**: 30
- **Width/Height**: 340x512 or 512x768. I’d try to keep to the aspect ratio of the original image, but these can be set lower than the resolution of the original image.
- **CFG Scale**: 6 to 8
- **Denoising strength**: 0.2 to 0.4 is usual. The lower you go, the less change between loops. The higher you go, the less the end result will look like the original image.
- **Seed**: This doesn’t matter too much, I usually keep it at -1
