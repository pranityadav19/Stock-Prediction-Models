#!/bin/bash
# Stock Results Tracker
# Run this to check predictions from 5 trading days ago

# Set up environment for cron
export PATH="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export HOME="/Users/pranityadav"

cd /Users/pranityadav/Downloads/Stock-Prediction-Models

# Create logs directory if it doesn't exist
mkdir -p /Users/pranityadav/Downloads/Stock-Prediction-Models/daily_alerts/logs/predictions
mkdir -p /Users/pranityadav/Downloads/Stock-Prediction-Models/daily_alerts/logs/results

# Get date from 7 days ago (approximately 5 trading days)
CHECK_DATE=$(date -v-7d +%Y-%m-%d)

echo "=========================================="
echo "Checking stock results for $CHECK_DATE at $(date)"
echo "=========================================="

# Run results tracker
/opt/anaconda3/bin/python -m daily_alerts.results_tracker $CHECK_DATE

echo "Results tracking completed at $(date)"
