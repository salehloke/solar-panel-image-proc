#!/bin/bash

# SolarAI Backend - Hybrid Mode Cleanup
# This script stops the hybrid mode backend and cleans up Docker containers

set -e  # Exit on any error

echo "🛑 Stopping SolarAI Backend Hybrid Mode..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running."
    exit 1
fi

# Stop PostgreSQL container
if docker ps | grep -q "local-postgres"; then
    print_status "Stopping PostgreSQL container..."
    docker stop local-postgres
    print_success "PostgreSQL stopped"
else
    print_warning "PostgreSQL container is not running"
fi

# Stop Redis container
if docker ps | grep -q "local-redis"; then
    print_status "Stopping Redis container..."
    docker stop local-redis
    print_success "Redis stopped"
else
    print_warning "Redis container is not running"
fi

# Ask if user wants to remove containers
echo ""
read -p "Do you want to remove the containers? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if docker ps -a | grep -q "local-postgres"; then
        print_status "Removing PostgreSQL container..."
        docker rm local-postgres
        print_success "PostgreSQL container removed"
    fi
    
    if docker ps -a | grep -q "local-redis"; then
        print_status "Removing Redis container..."
        docker rm local-redis
        print_success "Redis container removed"
    fi
else
    print_status "Containers stopped but not removed. You can restart them later."
fi

print_success "Hybrid mode cleanup completed!" 