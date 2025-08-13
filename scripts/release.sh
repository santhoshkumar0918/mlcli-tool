#!/bin/bash
# Complete release script for ML Assistant CLI
# Handles version bumping, building, testing, and deployment

set -e

echo "🚀 ML Assistant CLI Release Script"
echo "=================================="

# Configuration
CURRENT_VERSION=$(python -c "import mlcli; print(mlcli.__version__)")
DOCKER_USERNAME="santhoshkumar0918"
IMAGE_NAME="ml-assistant-cli"
REPO_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo "📋 Current version: $CURRENT_VERSION"

# Version options
echo ""
echo "Version bump options:"
echo "1. Patch (0.2.0 → 0.2.1) - Bug fixes"
echo "2. Minor (0.2.0 → 0.3.0) - New features"
echo "3. Major (0.2.0 → 1.0.0) - Breaking changes"
echo "4. Custom version"
echo "5. Keep current version ($CURRENT_VERSION)"

read -p "Choose option (1-5): " -n 1 -r
echo

case $REPLY in
    1)
        # Patch version bump
        NEW_VERSION=$(python -c "
import mlcli
v = mlcli.__version__.split('.')
v[2] = str(int(v[2]) + 1)
print('.'.join(v))
")
        ;;
    2)
        # Minor version bump
        NEW_VERSION=$(python -c "
import mlcli
v = mlcli.__version__.split('.')
v[1] = str(int(v[1]) + 1)
v[2] = '0'
print('.'.join(v))
")
        ;;
    3)
        # Major version bump
        NEW_VERSION=$(python -c "
import mlcli
v = mlcli.__version__.split('.')
v[0] = str(int(v[0]) + 1)
v[1] = '0'
v[2] = '0'
print('.'.join(v))
")
        ;;
    4)
        read -p "Enter custom version: " NEW_VERSION
        ;;
    5)
        NEW_VERSION=$CURRENT_VERSION
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo "📋 New version: $NEW_VERSION"

# Update version if changed
if [ "$NEW_VERSION" != "$CURRENT_VERSION" ]; then
    echo "📝 Updating version in mlcli/__init__.py..."
    sed -i "s/__version__ = \"$CURRENT_VERSION\"/__version__ = \"$NEW_VERSION\"/g" mlcli/__init__.py
    echo "✅ Version updated to $NEW_VERSION"
fi

# Pre-release checks
echo ""
echo "🔍 Running pre-release checks..."

# Test import
python -c "import mlcli; print(f'✓ Import successful - v{mlcli.__version__}')"

# Test CLI
mlcli --help > /dev/null && echo "✓ CLI help works"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Build package
echo "🔨 Building Python package..."
python -m build

# Check package quality
echo "🔍 Checking package quality..."
twine check dist/*

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t ${REPO_NAME}:${NEW_VERSION} -t ${REPO_NAME}:latest .

# Test Docker image
echo "🧪 Testing Docker image..."
docker run --rm ${REPO_NAME}:${NEW_VERSION} --help > /dev/null && echo "✓ Docker image works"

# Show what will be deployed
echo ""
echo "📦 Ready to deploy:"
echo "  Python package: ml-assistant-cli v$NEW_VERSION"
echo "  Docker image: ${REPO_NAME}:${NEW_VERSION}"
echo "  Files:"
ls -la dist/

# Deployment options
echo ""
echo "🚀 Deployment options:"
echo "1. Deploy to PyPI only"
echo "2. Deploy to Docker Hub only"
echo "3. Deploy to both PyPI and Docker Hub"
echo "4. Create GitHub release (requires git tag)"
echo "5. Full deployment (PyPI + Docker + GitHub)"
echo "6. Skip deployment"

read -p "Choose option (1-6): " -n 1 -r
echo

case $REPLY in
    1)
        echo "📤 Deploying to PyPI..."
        twine upload dist/*
        echo "✅ Deployed to PyPI: https://pypi.org/project/ml-assistant-cli/"
        ;;
    2)
        echo "📤 Deploying to Docker Hub..."
        docker push ${REPO_NAME}:${NEW_VERSION}
        docker push ${REPO_NAME}:latest
        echo "✅ Deployed to Docker Hub: https://hub.docker.com/r/${REPO_NAME}"
        ;;
    3)
        echo "📤 Deploying to PyPI..."
        twine upload dist/*
        echo "✅ PyPI deployment complete"
        
        echo "📤 Deploying to Docker Hub..."
        docker push ${REPO_NAME}:${NEW_VERSION}
        docker push ${REPO_NAME}:latest
        echo "✅ Docker Hub deployment complete"
        ;;
    4)
        echo "🏷️  Creating GitHub release..."
        git add mlcli/__init__.py
        git commit -m "Bump version to $NEW_VERSION" || true
        git tag "v$NEW_VERSION"
        git push origin "v$NEW_VERSION"
        echo "✅ GitHub tag created: v$NEW_VERSION"
        ;;
    5)
        echo "🚀 Full deployment starting..."
        
        # PyPI
        echo "📤 Deploying to PyPI..."
        twine upload dist/*
        echo "✅ PyPI deployment complete"
        
        # Docker Hub
        echo "📤 Deploying to Docker Hub..."
        docker push ${REPO_NAME}:${NEW_VERSION}
        docker push ${REPO_NAME}:latest
        echo "✅ Docker Hub deployment complete"
        
        # GitHub
        echo "🏷️  Creating GitHub release..."
        git add mlcli/__init__.py
        git commit -m "Release v$NEW_VERSION" || true
        git tag "v$NEW_VERSION"
        git push origin "v$NEW_VERSION"
        echo "✅ GitHub tag created"
        
        echo "🎉 Full deployment complete!"
        ;;
    6)
        echo "⏭️  Skipping deployment"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

# Post-deployment verification
if [[ $REPLY =~ ^[1-3,5]$ ]]; then
    echo ""
    echo "🔍 Post-deployment verification:"
    echo "  PyPI: https://pypi.org/project/ml-assistant-cli/"
    echo "  Docker: https://hub.docker.com/r/${REPO_NAME}"
    echo "  Install: pip install ml-assistant-cli==$NEW_VERSION"
    echo "  Docker: docker run --rm ${REPO_NAME}:${NEW_VERSION} --help"
fi

echo ""
echo "🎉 Release script complete!"
echo "📋 Version: $NEW_VERSION"
echo "📦 Package: ml-assistant-cli"
echo "🐳 Docker: ${REPO_NAME}"