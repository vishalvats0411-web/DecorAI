#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip for compatibility
pip install --upgrade pip

# Install PyTorch and Torchvision first from the official CPU index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Now, install the rest of the packages from your requirements file
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

# Run Django management commands
python manage.py collectstatic --no-input
python manage.py migrate