#!/usr/bin/env bash
# exit on error
set -o errexit

# Install numpy separately to ensure compatibility with opencv
pip install numpy==1.23.5

# Now install the rest of the packages
pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate