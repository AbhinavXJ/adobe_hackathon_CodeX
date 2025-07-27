#!/bin/bash

echo "Building Adobe Hackathon Round 1A Docker Image..."

# Build Docker image for AMD64 platform (hackathon requirement)
docker build --platform linux/amd64 -t adobe-round1a:latest .

echo "Docker image built successfully!"
echo ""
echo "To test locally:"
echo "1. Create test directories:"
echo "   mkdir -p test_input test_output"
echo ""
echo "2. Copy a test PDF:"
echo "   cp your_test.pdf test_input/"
echo ""
echo "3. Run the container:"
echo "   docker run --rm -v \$(pwd)/test_input:/app/input -v \$(pwd)/test_output:/app/output --network none adobe-round1a:latest"
echo ""
echo "4. Check results:"
echo "   ls test_output/"
