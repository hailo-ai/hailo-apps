CLIP Zero-shot Classification
=============================

This example performs zero-shot classification using a Hailo8 device, allowing you to input an image and multiple text prompts to generate classification probabilities. The example receives HEF files for the text and image encoders, tokenized prompts, and an optional frame count, then outputs the most likely prompt for each input frame.

Requirements
------------

- `hailo_platform` >= 4.19.0
- `OpenCV` >= 4.2.X
- `CMake` >= 3.20

Usage
-----
0. Verify Dependencies: ensure you have the correct version of HailoRT and that all dependencies are installed

1. Download resources
	```shell script
    ./download_resources.sh
    ```
    The following files will be downloaded: 
    - bus.jpg
    - clip_text_encoder_vit_l_14_laion2B.hef
    - clip_vit_l_14_laion2B_image_encoder.hef
    - text_projection.bin
    - ViT-L-14_laion2b_s32b_b82k.npy
    - bpe_simple_vocab_16e6.txt

    Note: Some downloaded files (e.g., text_projection.bin, ViT-L-14_laion2b_s32b_b82k.npy, and bpe_simple_vocab_16e6.txt) are essential even if you are using your own HEF files.

2. Compile the project
	```shell script
    ./build.sh
    ```
	This will create a build/x86 directory containing the executable file zero_shot_classification .

3. Run the example:

	```shell script
    ./build/x86_64/zero_shot_classification  -te=<text-encoder-hef> -ie=<image-encoder-hef> -t=<path-to-tokenized-prompt> -i=<input-image-or-video> -n=<number-of-frames>
    ```
	
Arguments

- ``-te``: Path to text encoder HEF file.
- ``-ie``: Path to image encoder HEF file.
- ``-p``: input prompts for the text encoder.
- ``-i``: Path to the input image for the image encoder.
- ``-n (optional)``: Number of times to run same image.


Example
---------------

```shell script
./build/x86_64/zero_shot_classification  -te=clip_text_encoder_vit_l_14_laion2B.hef -ie=clip_vit_l_14_laion2B_image_encoder.hef -p="a bird","a bus","a boat" -i=bus.jpg -n=30
```

Notes
-----
- Supported image formats: .jpg, .png, .bmp.
- Ensure no spaces between the = and the argument value in command-line parameters.
- This example has been tested specifically with clip_vit_l14.
- The -n flag is ignored for video inputs.

Disclaimer
----------
This code example is provided by Hailo solely on an “AS IS” basis and “with all faults”. No responsibility or liability is accepted or shall be imposed upon Hailo regarding the accuracy, merchantability, completeness or suitability of the code example. Hailo shall not have any liability or responsibility for errors or omissions in, or any business decisions made by you in reliance on this code example or any part of it. If an error occurs when running this example, please open a ticket in the "Issues" tab.

This code was tested on specific versions and environments as listed in the requirements above. While it may work with other versions, environments, or HEF files, there is no guarantee.