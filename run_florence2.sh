#!/bin/bash

# Variables
TRIGGER="your_trigger_word_or_sentence"
IMAGE_DIR="./path_to_your_image_directory"

# Run the caption generation script
python caption_with_florence2.py --trigger "$TRIGGER" --image-dir "$IMAGE_DIR"
