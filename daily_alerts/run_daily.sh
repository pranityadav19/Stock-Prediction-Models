#!/bin/bash
# Daily Stock Prediction Alerts
# Run this in the morning before market opens (6:30 AM ET)

# Set up environment for cron (cron runs in minimal environment)
export PATH="/opt/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export HOME="/Users/pranityadav"

# SendGrid API key - set in environment or .env file (not hardcoded for security)
# export SENDGRID_API_KEY="your-key-here"  # Set this before running locally

cd /Users/pranityadav/Downloads/Stock-Prediction-Models

# Create logs directory if it doesn't exist
mkdir -p /Users/pranityadav/Downloads/Stock-Prediction-Models/daily_alerts/logs

echo "=========================================="
echo "Starting Stock analysis at $(date)"
echo "=========================================="

# Run the daily analysis with full python path
/opt/anaconda3/bin/python -m daily_alerts.daily_runner

echo "Stock analysis completed at $(date)"
