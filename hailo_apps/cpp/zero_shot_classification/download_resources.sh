#!/bin/bash

wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/images/bus.jpg
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/clip_text_encoder_vit_l_14_laion2B.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/hefs/h8/clip_vit_l_14_laion2B_image_encoder.hef
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/external+bin+files/text_projection.bin
cd tokenizer
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy+files/embedding_weights.npy -O ViT-L-14_laion2b_s32b_b82k.npy
wget https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/txt+files/bpe_simple_vocab_16e6.txt
cd ..

