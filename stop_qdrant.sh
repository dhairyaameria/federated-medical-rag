#!/bin/bash

# Stop Qdrant Vector Database

echo "🛑 Stopping Qdrant..."

if docker ps | grep -q qdrant; then
    docker stop qdrant
    echo "✅ Qdrant stopped"
else
    echo "⚠️  Qdrant is not running"
fi

