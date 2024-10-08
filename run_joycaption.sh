#!/bin/bash

# Variables
TRIGGER="oxen_style"
IMAGE_DIR="../teenyicons/to_train"

# Test run variable
TEST_RUN=true

# Run the caption generation script
python caption_with_joycaption.py --trigger "$TRIGGER" --image-dir "$IMAGE_DIR" $( [ "$TEST_RUN" = true ] && echo "--test-run" )
