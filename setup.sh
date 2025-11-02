#!/usr/bin/env bash

# Video Summary Generator - Setup Script
# This script sets up the development environment for the project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Video Summary Generator - Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Found Python $PYTHON_VERSION"

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    echo "Please ensure you are in the project root directory."
    exit 1
fi

echo -e "${GREEN}✓${NC} Found requirements.txt"
echo ""

# Ask about virtual environment
read -p "Create a virtual environment? (recommended) [Y/n]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    USE_VENV=false
    echo -e "${YELLOW}⚠ Skipping virtual environment creation${NC}"
else
    USE_VENV=true
    VENV_DIR=".venv"
    
    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
        read -p "Remove and recreate? [y/N]: " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            USE_VENV=false
            echo "Using existing virtual environment."
        fi
    fi
    
    if [ "$USE_VENV" = true ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        
        # Activate virtual environment based on OS
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
            ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
        else
            ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
        fi
        
        echo -e "${GREEN}✓${NC} Virtual environment created"
        echo -e "${YELLOW}To activate it, run:${NC}"
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
            echo "  .venv\\Scripts\\activate  (Windows)"
        else
            echo "  source .venv/bin/activate  (Linux/Mac)"
        fi
    fi
    
    # Activate virtual environment for this script
    if [ -f "$ACTIVATE_SCRIPT" ]; then
        source "$ACTIVATE_SCRIPT"
        echo -e "${GREEN}✓${NC} Virtual environment activated"
    fi
fi

echo ""
echo "Installing dependencies from requirements.txt..."
$PIP_CMD install -r requirements.txt

echo -e "${GREEN}✓${NC} Dependencies installed"
echo ""

# Create backend/.env if it doesn't exist
if [ ! -f "backend/.env" ]; then
    echo "Creating backend/.env file..."
    mkdir -p backend
    cat > backend/.env << EOF
# Backend API Configuration (Optional)
# These are for final summarization API - leave empty if not needed
API_URL=
API_KEY=
MODEL_NAME=tngtech/deepseek-r1t2-chimera:free
EOF
    echo -e "${GREEN}✓${NC} Created backend/.env"
    echo -e "${YELLOW}⚠ Remember to add your API credentials to backend/.env if needed${NC}"
else
    echo -e "${YELLOW}⚠ backend/.env already exists, skipping creation${NC}"
fi

# Create app/.streamlit directory structure
if [ ! -d "app/.streamlit" ]; then
    echo "Creating app/.streamlit directory..."
    mkdir -p app/.streamlit
    echo -e "${GREEN}✓${NC} Created app/.streamlit directory"
fi

# Create example secrets.toml if it doesn't exist
if [ ! -f "app/.streamlit/secrets.toml" ]; then
    echo "Creating app/.streamlit/secrets.toml template..."
    cat > app/.streamlit/secrets.toml << EOF
# Streamlit Secrets Configuration
# For local development, set your Modal backend URL here
# For Streamlit Cloud, set this in the dashboard

BACKEND_API_URL = ""

# Example (uncomment and set your URL):
# BACKEND_API_URL = "https://your-username--video-summary-generator-fastapi-app.modal.run"
EOF
    echo -e "${GREEN}✓${NC} Created app/.streamlit/secrets.toml template"
else
    echo -e "${YELLOW}⚠ app/.streamlit/secrets.toml already exists, skipping creation${NC}"
fi

# Create upload directories (they'll be created automatically, but this ensures they exist)
mkdir -p app/uploads
mkdir -p backend/uploads

echo -e "${GREEN}✓${NC} Created upload directories"
echo ""

# Check for Modal CLI
if command -v modal &> /dev/null; then
    echo -e "${GREEN}✓${NC} Modal CLI is installed"
else
    echo -e "${YELLOW}⚠ Modal CLI not found${NC}"
    echo "  Install with: pip install modal"
    echo "  Then authenticate with: modal token new"
fi

# Check for Streamlit
if command -v streamlit &> /dev/null || $PIP_CMD show streamlit &> /dev/null; then
    echo -e "${GREEN}✓${NC} Streamlit is installed"
else
    echo -e "${YELLOW}⚠ Streamlit not found${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. ${YELLOW}Configure Backend (Optional):${NC}"
echo "   Edit backend/.env and add API credentials if needed"
echo ""
echo "2. ${YELLOW}Configure Frontend:${NC}"
echo "   Edit app/.streamlit/secrets.toml and set BACKEND_API_URL"
echo "   Or set environment variable: export BACKEND_API_URL='your-modal-url'"
echo ""
echo "3. ${YELLOW}Deploy Backend to Modal:${NC}"
echo "   modal deploy backend/modal_app.py"
echo ""
echo "4. ${YELLOW}Run Frontend Locally:${NC}"
if [ "$USE_VENV" = true ]; then
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
        echo "   .venv\\Scripts\\activate"
    else
        echo "   source .venv/bin/activate"
    fi
fi
echo "   cd app"
echo "   streamlit run main.py"
echo ""
echo "5. ${YELLOW}Or Deploy Frontend to Streamlit Cloud:${NC}"
echo "   Push to GitHub, then deploy via streamlit.io/cloud"
echo ""
echo -e "${BLUE}========================================${NC}"
