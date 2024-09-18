#!/bin/bash

# Function to check if a file is an image
is_image() {
    mime=$(file -b --mime-type "$1")
    [[ $mime == image/* ]]
}

# Function to get image dimensions
get_image_dimensions() {
    identify -format "%w %h" "$1" 2>/dev/null
}

# Initialize variables
TEST_MODE=false

# Parse options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--test) TEST_MODE=true ;;
        *) DIRECTORY="$1" ;;
    esac
    shift
done

# Check if a directory argument is provided
if [ -z "$DIRECTORY" ]; then
    echo "Usage: $0 [-t|--test] <directory>"
    exit 1
fi

# Loop through all files in the specified directory and subdirectories
find "$DIRECTORY" -type f | while read -r file; do
    if is_image "$file"; then
        # Get image dimensions
        read width height < <(get_image_dimensions "$file")
        
        # Check if either dimension is less than 1024
        if [[ -n $width && -n $height ]]; then
            if (( width < 1024 && height < 1024 )); then
                echo "Removing small image: $file ($width x $height)"
                if [ "$TEST_MODE" = false ]; then
                    rm "$file"
                fi
            fi
        else
            echo "Failed to get dimensions for $file, skipping"
        fi
    else
        # Remove non-image files
        echo "Removing non-image file: $file"
        if [ "$TEST_MODE" = false ]; then
            rm "$file"
        fi
    fi
done

echo "Cleanup complete."
