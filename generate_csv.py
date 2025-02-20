import os
import csv

def generate_csv(txt_file, csv_file):
    """
    Reads a text file containing image file paths and writes a CSV file
    mapping image filenames to label filenames.
    """
    if not os.path.exists(txt_file):
        print(f"Warning: {txt_file} not found. Skipping...")
        return

    with open(txt_file, "r") as f, open(csv_file, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        for line in f:
            image_file = os.path.basename(line.strip())  # Extract filename
            text_file = image_file.replace(".jpg", ".txt")  # Convert to label file
            writer.writerow([image_file, text_file])  # Write row to CSV

    print(f"Successfully created {csv_file}")

# Generate CSVs for training and testing sets
generate_csv("train.txt", "train.csv")
generate_csv("test.txt", "test.csv")
