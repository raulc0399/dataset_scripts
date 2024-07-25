from datasets import load_dataset, concatenate_datasets
import requests
from tqdm import tqdm
import os
import time

# Load the dataset
dataset = load_dataset("ptx0/photo-concept-bucket")
dataset = dataset['train']

keywords_to_filter_for = ["person", "man", "woman", "men", "women", "people", "boys", "girls", "boy", "girl"]
# keywords_to_filter_for = ["architecture", "house", "skyscraper", "facade"]

def contains_keywords(text):
    if text is None:
        return False

    text = text.lower()

    return any(keyword in text for keyword in keywords_to_filter_for)

def contains_keyword(keyword, text):
    if text is None:
        return False

    text = text.lower()

    return keyword in text


filtered_dataset = dataset.filter(
    lambda entry: contains_keywords(entry.get("title")) or 
                  contains_keywords(entry.get("cogvlm_caption"))
)

print(f"Number of entries containing keywords: {filtered_dataset.num_rows}")

files = os.listdir("../photo-concept-bucket-images")
filtered_dataset = filtered_dataset.filter(lambda entry: f"image_{entry["id"]}.jpg" in files)

print(f"Number of entries containing keywords and having images: {filtered_dataset.num_rows}")

# filtered_dataset = filtered_dataset.shuffle(seed=42).select(range(0, 20000))

# filtered_dataset_arch = filtered_dataset.filter(lambda entry: contains_keyword("architecture", entry.get("title")) or contains_keyword("architecture", entry.get("cogvlm_caption")))
# filtered_dataset_other = filtered_dataset.filter(lambda entry: not contains_keyword("architecture", entry.get("title")) and not contains_keyword("architecture", entry.get("cogvlm_caption")))

# print(f"Number of entries containing 'architecture': {filtered_dataset_arch.num_rows}")
# print(f"Number of entries not containing 'architecture': {filtered_dataset_other.num_rows}")

# filtered_dataset = concatenate_datasets([filtered_dataset_arch, filtered_dataset_other.shuffle(seed=42).select(range(0, 20000))])

# Define the keywords
keywords = keywords_to_filter_for

# Initialize a dictionary to store the statistics
statistics = {}

# Iterate over the filtered dataset
for entry in filtered_dataset:
    # Iterate over the keywords
    for keyword in keywords:
        if(contains_keyword(keyword, entry.get("title")) or contains_keyword(keyword, entry.get("cogvlm_caption"))):
            # If the keyword is present in the title or the caption, increment the count
            count = statistics.get(keyword, 0)
            count += 1
        
            # Add the count to the statistics dictionary
            statistics[keyword] = count

# Print the statistics
for keyword, count in statistics.items():
    print(f"Number of entries containing '{keyword}': {count}")

print(f"Total number of entries: {filtered_dataset.num_rows}")

filtered_dataset.to_json("../persons_dataset.jsonl", orient="records", lines=True)