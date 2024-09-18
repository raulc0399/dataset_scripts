#!/bin/bash

# Variables
TRIGGER="oxen_style"
IMAGE_DIR="../teenyicons/to_train"

# Test run variable
TEST_RUN=true

# Output directory variable
OUTPUT_DIR=""

# Run the caption generation script
python caption_with_llava.py --trigger "$TRIGGER" --image-dir "$IMAGE_DIR" $( [ -n "$OUTPUT_DIR" ] && echo "--output-dir $OUTPUT_DIR" ) $( [ "$TEST_RUN" = true ] && echo "--test-run" )
