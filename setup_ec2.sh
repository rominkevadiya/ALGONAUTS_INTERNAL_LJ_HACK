#!/bin/bash
# setup_ec2.sh - Script to setup the Streamlit app on a fresh Ubuntu EC2 instance

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Installing Python, pip, and venv..."
sudo apt install -y python3-pip python3-venv git tmux

echo "Cloning repository..."
# NOTE: Replace the URL below with your actual git repository URL if it's public/private
# If you don't use git, you can use SCP or SFTP to copy the files to the instance.
# git clone <your-repo-url> app-repo
# cd app-repo

# Let's assume the files are already in ~/app-repo
# Ensure we are in the right directory
# cd ~/app-repo

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
# This might take a while because of PyTorch
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To run the app in the background so it stays alive when you close the terminal:"
echo "1. Create a tmux session: tmux new -s streamlit"
echo "2. Activate venv: source .venv/bin/activate"
echo "3. Run app: streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0"
echo "4. Detach from tmux: Press Ctrl+b, then d"
echo ""
echo "Don't forget to create/copy your .env file with your API keys!"
