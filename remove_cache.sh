find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete