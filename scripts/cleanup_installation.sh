#!/bin/bash
# Cleanup installation artifacts from hailo-apps

set -euo pipefail

sudo rm -rf resources hailo_apps.egg-info/ venv_hailo_apps/ hailort.log 
sudo rm -rf /usr/local/hailo/resources/
