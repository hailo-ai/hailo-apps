#!/bin/bash
# Uninstall all Hailo packages (pip, apt, kernel modules)

set -euo pipefail

echo "Uninstalling Hailo packages..."

# Uninstall pip packages
echo "Removing pip packages..."
# Try removing as regular user (in case installed with --user or --break-system-packages)
if [ -n "$SUDO_USER" ]; then
    echo "Trying to remove pip packages as user $SUDO_USER..."
    sudo -u "$SUDO_USER" pip uninstall -y --break-system-packages hailo-tappas-core-python-binding hailort 2>/dev/null || true
fi
# Try removing as root (in case installed system-wide with sudo)
echo "Trying to remove pip packages as root..."
pip uninstall -y --break-system-packages hailo-tappas-core-python-binding hailort 2>/dev/null || true

# Uninstall apt packages
echo "Removing apt packages..."
sudo apt purge -y hailo-tappas-core hailort hailort-pcie-driver 2>/dev/null || true

# Remove kernel modules
echo "Removing kernel modules..."
sudo find /lib/modules -type f \( -name 'hailo*.ko' -o -name 'hailo*.ko.xz' \) -print -delete
sudo find /lib/modules -type d -name 'hailo' -print -exec rm -rf {} + 2>/dev/null || true

# Update kernel modules and initramfs
echo "Updating kernel modules..."
sudo depmod -a
echo "Updating initramfs..."
sudo update-initramfs -u

# Remove resources directory
echo "Removing resources directory..."
sudo rm -rf /usr/local/hailo/resources/

echo "Hailo packages uninstalled successfully!"