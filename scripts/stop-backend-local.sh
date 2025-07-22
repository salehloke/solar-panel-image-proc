#!/bin/bash

# SolarAI Backend - Local Mode Stop Script
# This script stops the hybrid backend (local FastAPI + Docker databases)

set -e

echo "🛑 Stopping SolarAI Backend Local Mode..."

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

# Stop local FastAPI server
print_status "Stopping local FastAPI server..."
if pkill -f "uvicorn app.main:app" 2>/dev/null; then
    print_success "FastAPI server stopped"
else
    print_warning "No FastAPI server was running"
fi

# Stop PostgreSQL container
print_status "Stopping PostgreSQL container..."
if docker ps | grep -q "local-postgres"; then
    docker stop local-postgres
    print_success "PostgreSQL container stopped"
else
    print_warning "PostgreSQL container is not running"
fi

# Stop Redis container
print_status "Stopping Redis container..."
if docker ps | grep -q "local-redis"; then
    docker stop local-redis
    print_success "Redis container stopped"
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

# Show volume information
echo ""
print_status "Volume Information:"
print_status "Models and logs are stored in Docker volumes (persistent)"
print_status "To manage volumes, use: ./scripts/manage-volumes.sh info"

# Check if any processes are still running on the ports
print_status "Checking for any remaining processes..."

if lsof -i :8000 > /dev/null 2>&1; then
    print_warning "Port 8000 is still in use:"
    lsof -i :8000
else
    print_success "Port 8000 is free"
fi

if lsof -i :5432 > /dev/null 2>&1; then
    print_warning "Port 5432 is still in use:"
    lsof -i :5432
else
    print_success "Port 5432 is free"
fi

if lsof -i :6379 > /dev/null 2>&1; then
    print_warning "Port 6379 is still in use:"
    lsof -i :6379
else
    print_success "Port 6379 is free"
fi

echo ""
print_success "Local backend shutdown completed!"
echo ""
echo "📋 To restart:"
echo "  • ./scripts/start-backend-local.sh"
echo ""
echo "📋 Other commands:"
echo "  • Start Docker backend: ./scripts/start-backend-docker.sh"
echo "  • Check status: ./scripts/backend-status.sh" 