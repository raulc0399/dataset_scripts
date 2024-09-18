#!/bin/bash

# Variables
TRIGGER="oxen_style"
IMAGE_DIR="../teenyicons/to_train"

# Test run variable
TEST_RUN=true

# Run the simple caption generation script
python simple_caption.py --trigger "$TRIGGER" --image-dir "$NEW_IMAGE_DIR" $( [ "$TEST_RUN" = true ] && echo "--test-run" )
