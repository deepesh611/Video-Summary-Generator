#!/usr/bin/env bash

# Install Modules
# shellcheck disable=SC1019
# shellcheck disable=SC1073
# shellcheck disable=SC1072
if [ -f "requirement.txt"]; then
    pip install -r requirement.txt
else
    echo "requirement.txt not found!"
    echo "Ensure that your current directory is Type-1"
    exit 1
fi

# Create .env
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    touch .env
    echo "API_URL=< your-api-url >" >> .env
    echo "API_KEY=< your-api-key >" >> .env
    echo ".env file created."
else
    echo ".env file already exists."
fi

# Create Required Directories
mkdir "input"
mkdir "output/frames"

echo "Setup completed."
echo "Place your video files in the 'input' directory."
echo "Processed frames will be saved in the 'output/frames' directory."
echo "Don't forget to update the .env file with your API credentials."

exit 0
