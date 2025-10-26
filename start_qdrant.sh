#!/bin/bash

# Start Qdrant Vector Database

echo "🚀 Starting Qdrant Vector Database..."

# Check if Qdrant container already exists
if docker ps -a | grep -q qdrant; then
    echo "Qdrant container exists. Starting it..."
    docker start qdrant
else
    echo "Creating new Qdrant container..."
    docker run -d \
        --name qdrant \
        -p 6333:6333 \
        -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest
fi

echo "Waiting for Qdrant to be ready..."
sleep 5

# Check if Qdrant is running
if curl -s http://localhost:6333/collections > /dev/null; then
    echo "✅ Qdrant is running successfully!"
    echo "   Dashboard: http://localhost:6333/dashboard"
    echo "   To stop: docker stop qdrant"
else
    echo "❌ Qdrant failed to start. Check logs with: docker logs qdrant"
fi

