#!/bin/bash

# SolarAI Model Loading Mode Test Script
# Tests both Solution 2 (volume) and Solution 3 (build-time) modes

set -e

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

# Function to stop all containers
stop_all_containers() {
    print_status "Stopping all SolarAI containers..."
    docker-compose -f docker-compose.backend.yml down 2>/dev/null || true
    docker-compose -f docker-compose.backend.build-time.yml down 2>/dev/null || true
    print_success "All containers stopped"
}

# Function to test Solution 2 (Volume Mode)
test_volume_mode() {
    print_status "Testing Solution 2: Volume Mode"
    print_status "Starting containers with volume mode..."
    
    # Set environment variable
    export MODEL_LOADING_MODE=volume
    
    # Start containers
    docker-compose -f docker-compose.backend.yml up -d
    
    # Wait for startup
    print_status "Waiting for backend to start..."
    sleep 30
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Health endpoint responding"
    else
        print_error "Health endpoint not responding"
        return 1
    fi
    
    # Test model info endpoint
    print_status "Testing model info endpoint..."
    MODEL_INFO=$(curl -s http://localhost:8000/model/info)
    echo "$MODEL_INFO" | jq .
    
    # Check loading mode
    LOADING_MODE=$(echo "$MODEL_INFO" | jq -r '.loading_mode')
    if [ "$LOADING_MODE" = "volume" ]; then
        print_success "Volume mode confirmed"
    else
        print_error "Expected volume mode, got: $LOADING_MODE"
        return 1
    fi
    
    print_success "Solution 2 (Volume Mode) test completed"
}

# Function to test Solution 3 (Build-time Mode)
test_buildtime_mode() {
    print_status "Testing Solution 3: Build-time Mode"
    print_status "Starting containers with build-time mode..."
    
    # Stop volume mode containers first
    docker-compose -f docker-compose.backend.yml down
    
    # Start build-time containers
    docker-compose -f docker-compose.backend.build-time.yml up -d
    
    # Wait for startup (build-time might take longer due to model download)
    print_status "Waiting for backend to start (build-time mode)..."
    sleep 60
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "Health endpoint responding"
    else
        print_error "Health endpoint not responding"
        return 1
    fi
    
    # Test model info endpoint
    print_status "Testing model info endpoint..."
    MODEL_INFO=$(curl -s http://localhost:8000/model/info)
    echo "$MODEL_INFO" | jq .
    
    # Check loading mode
    LOADING_MODE=$(echo "$MODEL_INFO" | jq -r '.loading_mode')
    if [ "$LOADING_MODE" = "build_time" ]; then
        print_success "Build-time mode confirmed"
    else
        print_error "Expected build_time mode, got: $LOADING_MODE"
        return 1
    fi
    
    print_success "Solution 3 (Build-time Mode) test completed"
}

# Function to test prediction endpoint
test_prediction() {
    print_status "Testing prediction endpoint..."
    
    # Find a test image
    TEST_IMAGE=$(find logs -name "*.png" | head -1)
    
    if [ -z "$TEST_IMAGE" ]; then
        print_warning "No test image found, skipping prediction test"
        return 0
    fi
    
    print_status "Using test image: $TEST_IMAGE"
    
    # Test prediction
    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@$TEST_IMAGE")
    
    if [ $? -eq 0 ]; then
        print_success "Prediction successful"
        echo "$RESPONSE" | jq .
    else
        print_error "Prediction failed"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  volume     Test Solution 2 (Volume Mode)"
    echo "  buildtime  Test Solution 3 (Build-time Mode)"
    echo "  both       Test both modes"
    echo "  stop       Stop all containers"
    echo "  status     Show container status"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 volume     # Test volume mode only"
    echo "  $0 buildtime  # Test build-time mode only"
    echo "  $0 both       # Test both modes"
}

# Function to show status
show_status() {
    print_status "Container Status:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep solarai || true
    
    print_status "Volume Status:"
    docker volume ls | grep solar-image-processing || true
}

# Main script logic
case "${1:-help}" in
    "volume")
        stop_all_containers
        test_volume_mode
        test_prediction
        print_success "Volume mode test completed successfully"
        ;;
    "buildtime")
        stop_all_containers
        test_buildtime_mode
        test_prediction
        print_success "Build-time mode test completed successfully"
        ;;
    "both")
        stop_all_containers
        print_status "Testing both modes..."
        
        print_status "=== Testing Volume Mode ==="
        test_volume_mode
        test_prediction
        
        print_status "=== Testing Build-time Mode ==="
        test_buildtime_mode
        test_prediction
        
        print_success "Both modes tested successfully"
        ;;
    "stop")
        stop_all_containers
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_usage
        ;;
esac 