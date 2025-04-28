#!/bin/bash
# Setup script for YOLO Object Detection project

# Color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print section header
print_section() {
    echo -e "\n${YELLOW}=================================${NC}"
    echo -e "${YELLOW} $1 ${NC}"
    echo -e "${YELLOW}=================================${NC}\n"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create virtual environment
create_venv() {
    print_section "Creating virtual environment"
    
    if command_exists python3; then
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${RED}✗ Python 3 not found. Please install Python 3 first.${NC}"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_section "Activating virtual environment"
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${RED}✗ Virtual environment not found.${NC}"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_section "Installing dependencies"
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo -e "${GREEN}✓ Dependencies installed${NC}"
    else
        echo -e "${RED}✗ requirements.txt not found.${NC}"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_section "Creating project directories"
    
    mkdir -p data models logs results
    echo -e "${GREEN}✓ Project directories created${NC}"
}

# Check environment
check_environment() {
    print_section "Checking environment"
    
    python src/check_environment.py
}

# Main function
main() {
    print_section "Setting up YOLO Object Detection project"
    
    # Check if in the correct directory
    if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
        echo -e "${RED}✗ Please run this script from the project root directory.${NC}"
        exit 1
    fi
    
    # Ask user if they want to create a virtual environment
    read -p "Do you want to create a new virtual environment? (y/n): " create_venv_choice
    
    if [[ $create_venv_choice == "y" || $create_venv_choice == "Y" ]]; then
        create_venv
        activate_venv
    else
        echo -e "${YELLOW}Skipping virtual environment creation.${NC}"
    fi
    
    # Install dependencies
    read -p "Do you want to install dependencies? (y/n): " install_deps_choice
    
    if [[ $install_deps_choice == "y" || $install_deps_choice == "Y" ]]; then
        install_dependencies
    else
        echo -e "${YELLOW}Skipping dependency installation.${NC}"
    fi
    
    # Create directories
    create_directories
    
    # Check environment
    check_environment
    
    print_section "Setup Complete"
    echo -e "${GREEN}You're ready to start working with YOLO Object Detection!${NC}"
    echo -e "To explore the COCO dataset: ${YELLOW}jupyter notebook notebooks/01_Setup_and_COCO_Dataset_Exploration.ipynb${NC}"
}

# Run the main function
main