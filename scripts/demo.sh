#!/bin/bash
# End-to-end demonstration of the Goalie prediction framework

set -e  # Exit on error

echo "=========================================="
echo "Goalie Prediction Framework - Demo"
echo "=========================================="
echo ""

# Step 1: Generate synthetic test data (in real use, you'd ingest from API)
echo "Step 1: Generating synthetic test data..."
python scripts/generate_test_data.py
echo ""

# Step 2: Feature engineering
echo "Step 2: Engineering features..."
python -m src.cli features
echo ""

# Step 3: Train models and evaluate
echo "Step 3: Training models..."
python -m src.cli train --run-id demo_run
echo ""

# Step 4: Generate analysis report
echo "Step 4: Generating analysis report..."
python -m src.cli report
echo ""

# Step 5: List generated artifacts
echo "Step 5: Listing generated artifacts..."
python -m src.cli list-data
echo ""

echo "=========================================="
echo "Demo complete!"
echo ""
echo "Check the following for results:"
echo "  - Report: artifacts/runs/demo_run/report.md"
echo "  - Predictions: artifacts/parquet/predictions/"
echo "  - Features: artifacts/parquet/features/"
echo "=========================================="
