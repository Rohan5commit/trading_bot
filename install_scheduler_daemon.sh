#!/bin/bash
# Script to install and start the trading bot scheduler as a macOS launch daemon

# Copy the plist file to the LaunchDaemons directory
sudo cp /Users/rohan/trading_bot/com.tradingbot.scheduler.plist /Library/LaunchDaemons/

# Set proper permissions
sudo chown root:wheel /Library/LaunchDaemons/com.tradingbot.scheduler.plist
sudo chmod 644 /Library/LaunchDaemons/com.tradingbot.scheduler.plist

# Load the daemon
sudo launchctl load /Library/LaunchDaemons/com.tradingbot.scheduler.plist

echo "Trading bot scheduler daemon installed and started successfully!"
echo "The scheduler will now automatically start on system boot."