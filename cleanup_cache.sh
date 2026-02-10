#!/bin/bash
# Script to clean up cache and temporary files while preserving P&L and strategy

echo "Starting cleanup of cache files..."

# Clean up feature store (regenerated daily)
echo "Cleaning up feature store..."
rm -f /Users/rohan/trading_bot/feature_store/*.csv
echo "Feature store cleaned."

# Clean up old logs (keep last 1000 lines of scheduler log)
echo "Cleaning up logs..."
tail -n 1000 /Users/rohan/trading_bot/scheduler.log > /Users/rohan/trading_bot/scheduler.log.tmp
mv /Users/rohan/trading_bot/scheduler.log.tmp /Users/rohan/trading_bot/scheduler.log

# Clean up old result files (keep only last 3 days)
echo "Cleaning up old result files..."
cd /Users/rohan/trading_bot/results/
find . -name "*.csv" -type f -mtime +3 -delete
find . -name "*.ok" -type f -mtime +3 -delete

# Clean up data archive (keep only latest)
echo "Cleaning up data archives..."
cd /Users/rohan/trading_bot/data/
ls -t trading_bot_archive_*.db | tail -n +2 | xargs rm -f

echo "Cleanup completed. Storage reduced while preserving P&L and strategy."