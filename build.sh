#!/usr/bin/env bash
# exit on error
set -o errexit

# Add this line to upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

# Create the checkpoints directory if it doesn't exist
mkdir -p main/static/checkpoints

# Download the AI model if it's not already there
MODEL_FILE="main/static/checkpoints/sam_vit_l_0b3195.pth"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading AI model (this may take a few minutes)..."
    wget -O "$MODEL_FILE" "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
else
    echo "AI model already exists."
fi

python manage.py collectstatic --no-input
python manage.py migrate