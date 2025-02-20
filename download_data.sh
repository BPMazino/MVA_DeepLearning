#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define dataset URLs
VOC2007_TRAINVAL="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC2007_TEST="http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
VOC2012_TRAINVAL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
LABEL_SCRIPT="https://pjreddie.com/media/files/voc_label.py"

# Function to download files if they do not already exist
download_if_not_exists() {
    local url=$1
    local filename=$(basename "$url")
    
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        wget -q --show-progress "$url"
    else
        echo "$filename already exists, skipping download."
    fi
}

# Download datasets
download_if_not_exists "$VOC2007_TRAINVAL"
download_if_not_exists "$VOC2007_TEST"
download_if_not_exists "$VOC2012_TRAINVAL"

# Extract datasets
echo "Extracting datasets..."
for tar_file in VOCtrainval_11-May-2012.tar VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar; do
    if [ -f "$tar_file" ]; then
        tar xf "$tar_file"
        rm "$tar_file"  # Remove tar file after extraction to save space
    fi
done

# Download voc_label.py
download_if_not_exists "$LABEL_SCRIPT"

# Run Python script to convert XML labels to TXT format
echo "Processing XML labels into TXT..."
python voc_label.py

# Merge train sets from 2007 and 2012
echo "Merging train and validation sets..."
cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
cp 2007_test.txt test.txt

# Move old text files to archive folder
mkdir -p old_txt_files
mv 2007_* 2012_* old_txt_files/

# Generate CSV files
python generate_csv.py

# Create required directories if not exist
mkdir -p data/images
mkdir -p data/labels

# Move images and labels
echo "Organizing dataset..."
mv VOCdevkit/VOC2007/JPEGImages/*.jpg data/images/ 2>/dev/null || true
mv VOCdevkit/VOC2012/JPEGImages/*.jpg data/images/ 2>/dev/null || true
mv VOCdevkit/VOC2007/labels/*.txt data/labels/ 2>/dev/null || true
mv VOCdevkit/VOC2012/labels/*.txt data/labels/ 2>/dev/null || true

# Cleanup: Remove unused VOCdevkit folder
echo "Cleaning up..."
rm -rf VOCdevkit/
mv test.txt old_txt_files/
mv train.txt old_txt_files/

echo "Dataset preparation complete! 🎉"
