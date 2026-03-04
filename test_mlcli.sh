mlcli init --name my-demo --description "Demo ML project" --plugin tabular#!/bin/bash
set -e

# Setup
PROJECT_DIR="test_workspace"
rm -rf "$PROJECT_DIR"
mkdir "$PROJECT_DIR"

# Activate venv
source .venv/bin/activate

# 1. Initialize Project
echo "Testing mlcli init..."
mlcli --project-dir "$PROJECT_DIR" init --name test-project

cd "$PROJECT_DIR"

# 2. Create Dummy Data
echo "Creating dummy data..."
mkdir -p data/raw
cat <<EOF > data/raw/data.csv
feature1,feature2,target
1.0,2.0,0
1.1,2.1,0
0.9,1.9,0
1.2,2.2,0
0.8,1.8,0
1.0,2.0,0
1.1,2.1,0
0.9,1.9,0
1.2,2.2,0
0.8,1.8,0
5.0,6.0,1
5.1,6.1,1
4.9,5.9,1
5.2,6.2,1
4.8,5.8,1
5.0,6.0,1
5.1,6.1,1
4.9,5.9,1
5.2,6.2,1
4.8,5.8,1
10.0,11.0,2
10.1,11.1,2
9.9,10.9,2
10.2,11.2,2
9.8,10.8,2
10.0,11.0,2
10.1,11.1,2
9.9,10.9,2
10.2,11.2,2
9.8,10.8,2
EOF

# 3. Preprocess
echo "Testing mlcli preprocess..."
mlcli preprocess --input data/raw/data.csv --target target

# 4. Train
echo "Testing mlcli train..."
mlcli train --target target

# 5. Evaluate
echo "Testing mlcli evaluate..."
mlcli evaluate --target target

# 6. Suggest
echo "Testing mlcli suggest..."
mlcli suggest

echo "All tests passed!"
