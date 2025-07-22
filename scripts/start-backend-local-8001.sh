#!/bin/bash

# SolarAI Backend - Hybrid Mode (Local PyTorch + Docker Databases)
# This script starts the backend with PyTorch running locally and databases in Docker
# Modified to use port 8001 instead of 8000

set -e  # Exit on any error

echo "🚀 Starting SolarAI Backend in Local Mode (Port 8001)..."

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

# Check if we're in the right directory
if [ ! -f "backend/app/main.py" ]; then
    print_error "backend/app/main.py not found. Please run this script from the project root."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

print_success "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    print_status "Creating virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source backend/venv/bin/activate

# Install/upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Create necessary directories (for logs only, models will be in container)
print_status "Creating log directory..."
mkdir -p logs

# Set environment variables for local mode
export DATABASE_URL="postgresql+asyncpg://solarai:solarai123@localhost:5432/solarai"
export POSTGRES_PASSWORD="solarai123"
export JWT_SECRET="your-secret-key"
export JWT_ALGORITHM="HS256"
export JWT_EXPIRES_IN="30"
export MODEL_PATH="models/resnet18_solar_panel.pt"
export LOG_LEVEL="INFO"

print_status "Environment variables set for local mode"

# Check if PostgreSQL is already running (either local-postgres or solarai_postgres)
print_status "Checking PostgreSQL availability..."
if docker ps | grep -q "local-postgres"; then
    print_success "Local PostgreSQL container is running"
    POSTGRES_CONTAINER="local-postgres"
elif docker ps | grep -q "solarai_postgres"; then
    print_success "SolarAI PostgreSQL container is running"
    POSTGRES_CONTAINER="solarai_postgres"
    print_status "Using existing SolarAI PostgreSQL container"
else
    # Start PostgreSQL in Docker if not running
    print_status "Setting up PostgreSQL in Docker..."
    if docker ps -a | grep -q "local-postgres"; then
        print_status "Starting existing PostgreSQL container..."
        docker start local-postgres
        POSTGRES_CONTAINER="local-postgres"
    else
        print_status "Creating PostgreSQL container..."
        docker run --name local-postgres \
            -e POSTGRES_DB=solarai \
            -e POSTGRES_USER=solarai \
            -e POSTGRES_PASSWORD=solarai123 \
            -p 5432:5432 \
            -d postgres:15
        POSTGRES_CONTAINER="local-postgres"
    fi
    
    # Wait for PostgreSQL to be ready
    print_status "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker exec $POSTGRES_CONTAINER pg_isready -U solarai > /dev/null 2>&1; then
            print_success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "PostgreSQL failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
fi

# Check if Redis is already running (either local-redis or solarai_redis)
print_status "Checking Redis availability..."
if docker ps | grep -q "local-redis"; then
    print_success "Local Redis container is running"
    REDIS_CONTAINER="local-redis"
elif docker ps | grep -q "solarai_redis"; then
    print_success "SolarAI Redis container is running"
    REDIS_CONTAINER="solarai_redis"
    print_status "Using existing SolarAI Redis container"
else
    # Start Redis in Docker if not running
    print_status "Setting up Redis in Docker..."
    if docker ps -a | grep -q "local-redis"; then
        print_status "Starting existing Redis container..."
        docker start local-redis
        REDIS_CONTAINER="local-redis"
    else
        print_status "Creating Redis container..."
        docker run --name local-redis \
            -p 6379:6379 \
            -d redis:7.2-alpine
        REDIS_CONTAINER="local-redis"
    fi
    
    # Wait for Redis to be ready
    print_status "Waiting for Redis to be ready..."
    for i in {1..10}; do
        if docker exec $REDIS_CONTAINER redis-cli ping > /dev/null 2>&1; then
            print_success "Redis is ready"
            break
        fi
        if [ $i -eq 10 ]; then
            print_error "Redis failed to start within 10 seconds"
            exit 1
        fi
        sleep 1
    done
fi

# Start the backend
print_status "Starting SolarAI Backend..."
cd backend

# Run the FastAPI application
print_success "Starting FastAPI server..."
print_status "The server will be available at: http://localhost:8001"
print_status "Press Ctrl+C to stop the server"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down..."
    print_status "Note: PostgreSQL and Redis containers are still running."
    print_status "To stop them, run: docker stop local-postgres local-redis"
    print_status "To remove them, run: docker rm local-postgres local-redis"
    print_status "Docker backend is still available at: http://localhost:8000"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Using port 8001 instead of 8000
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
