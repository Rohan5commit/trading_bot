#!/bin/bash
# Script to start the trading bot scheduler if not already running

# Check if scheduler is already running
if pgrep -f "scheduler.py" > /dev/null; then
    echo "Scheduler is already running"
    exit 0
fi

# Start the scheduler in the background
cd /Users/rohan/trading_bot
nohup python3 scheduler.py > scheduler_cron.log 2>&1 &

echo "Scheduler started at $(date)"