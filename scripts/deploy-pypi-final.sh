#!/bin/bash
# Final PyPI deployment script for ML Assistant CLI v0.2.0

set -e

echo "🚀 Final PyPI Deployment - ML Assistant CLI v0.2.0"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "mlcli/__init__.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Get current version
VERSION=$(python -c "import mlcli; print(mlcli.__version__)")
echo "📋 Current version: $VERSION"

# Verify we have the built packages
if [ ! -d "dist" ] || [ -z "$(ls -A dist/)" ]; then
    echo "📦 Building packages..."
    rm -rf build/ dist/ *.egg-info/
    python -m build
else
    echo "📦 Using existing packages in dist/"
fi

# Show what we're deploying
echo ""
echo "📋 Ready to deploy:"
ls -la dist/
echo ""

# Final checks
echo "🔍 Running final checks..."
python -c "import mlcli; print(f'✓ Version: {mlcli.__version__}')"
twine check dist/*
echo "✓ Package quality check passed"

# Deployment confirmation
echo ""
echo "🚨 FINAL CONFIRMATION"
echo "This will deploy ML Assistant CLI v$VERSION to PyPI"
echo "Once deployed, users worldwide can install with:"
echo "  pip install ml-assistant-cli"
echo ""
read -p "Are you absolutely sure you want to deploy to PyPI? (yes/no): " -r

if [[ $REPLY == "yes" ]]; then
    echo ""
    echo "🚀 Deploying to PyPI..."
    echo "You will be prompted for your PyPI API token"
    echo ""
    
    # Deploy to PyPI
    twine upload dist/*
    
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "================================"
    echo "✅ ML Assistant CLI v$VERSION is now live on PyPI!"
    echo ""
    echo "📦 Package URL: https://pypi.org/project/ml-assistant-cli/"
    echo "📥 Install command: pip install ml-assistant-cli"
    echo "🐳 Docker image: docker run -it --rm -v \$(pwd):/home/mlcli/workspace santhoshkumar0918/ml-assistant-cli:latest"
    echo ""
    echo "🌍 Your ML Assistant CLI is now available to millions of users worldwide!"
    echo ""
    echo "Next steps:"
    echo "1. Monitor PyPI for download statistics"
    echo "2. Update documentation with new installation instructions"
    echo "3. Announce the release to your community"
    echo "4. Start planning Phase 2 (Cloud MVP) features"
    
elif [[ $REPLY == "no" ]]; then
    echo "❌ Deployment cancelled"
    echo "The packages are ready in dist/ when you're ready to deploy"
else
    echo "❌ Invalid response. Please type 'yes' or 'no'"
    exit 1
fi