#!/bin/bash

# Check if the number of files is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <number_of_files>"
    exit 1
fi

# Number of files to move
num_files=$1

source_dir="imgs"

# Name of the subdirectory
subdir_simple="$1_simple"
subdir_florence2="$1_florence2"
subdir_llava="$1_llava"

# Create the subdirectory if it doesn't exist
mkdir -p "$subdir_simple"
mkdir -p "$subdir_florence2"
mkdir -p "$subdir_llava"

# Get the list of files in the current directory
files=($source_dir/*.png)

# Check if there are enough files
if [ ${#files[@]} -lt $num_files ]; then
    echo "Error: Not enough files in the current directory"
    exit 1
fi

# Randomly select and move files
for i in $(shuf -i 0-$((${#files[@]} - 1)) -n $num_files); do
    # Copy the .png file to the subdir_simple
    cp "${files[i]}" "$subdir_simple/"
    echo "Copied ${files[i]} to $subdir_simple/"

    # Determine the corresponding .txt file name
    txt_file="$(basename "${files[i]%.png}.txt")"

    # Copy the .txt file from simple to subdir_simple
    cp "simple/$txt_file" "$subdir_simple/"
    echo "Copied simple/$txt_file to $subdir_simple/"

    # Copy the .txt file from florence2 to subdir_florence2
    cp "florence2/$txt_file" "$subdir_florence2/"
    echo "Copied florence2/$txt_file to $subdir_florence2/"

    # Copy the .txt file from llava to subdir_llava
    cp "llava/$txt_file" "$subdir_llava/"
    echo "Copied llava/$txt_file to $subdir_llava/"
done

echo "Completed copying $num_files files."
