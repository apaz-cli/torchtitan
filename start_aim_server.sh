#!/bin/sh

# Activate virtual environment
. .venv/bin/activate

# Set the Aim repository path (default to .aim at repo root)
AIM_REPO="${AIM_REPO:-./.aim}"

# Create Aim repository directory if it doesn't exist
if [ ! -d "$AIM_REPO" ]; then
    echo "Creating directory: $AIM_REPO"
    mkdir -p "$AIM_REPO"
fi

# Check if Aim repository needs initialization
# An initialized Aim repo has a meta directory inside it
if [ ! -d "$AIM_REPO/meta" ]; then
    echo "Initializing Aim repository at $AIM_REPO"
    aim init --repo "$AIM_REPO" --yes
    if [ $? -ne 0 ]; then
        echo "Error: Failed to initialize Aim repository"
        exit 1
    fi
fi

echo "Starting Aim server..."
echo "Visit http://localhost:43800 to see the Aim UI"
echo "Press Ctrl+C to stop the server"
echo ""

# Start Aim server (blocking command)
aim up --host 0.0.0.0 --port 43800 --repo "$AIM_REPO"
