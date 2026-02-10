#!/bin/bash
# Script to uninstall the trading bot scheduler launch daemon

# Unload the daemon
sudo launchctl unload /Library/LaunchDaemons/com.tradingbot.scheduler.plist

# Remove the plist file
sudo rm /Library/LaunchDaemons/com.tradingbot.scheduler.plist

echo "Trading bot scheduler daemon uninstalled successfully!"