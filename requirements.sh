#!/bin/bash

# Set strict mode for better error handling
set -euo pipefail

# Display initial message
echo "update and installation of all necessary packages"

# Add necessary repositories
add-apt-repository -y cloud-archive:antelope
apt-add-repository -y ppa:ansible/ansible

# Update system to incorporate new repositories
apt-get update -y

# Install packages to set up the OpenStack environment and Ansible
apt-get install -y python3-openstackclient python3-novaclient python3-keystoneclient ansible

# Install Snap packages
snap install openstackclients

# Final update and upgrade after all installations
apt-get upgrade -y

# SSH key generation (no passphrase)
mkdir -p /home/ubuntu/cluster-keys
ssh-keygen -t rsa -b 4096 -f /home/ubuntu/cluster-keys/RSA-key -N ""

# Adjust file ownership and permissions
chown -R ubuntu:ubuntu /home/ubuntu/cluster-keys
chmod 700 /home/ubuntu/cluster-keys
chmod 600 /home/ubuntu/cluster-keys/RSA-key
chmod 644 /home/ubuntu/cluster-keys/RSA-key.pub

# Clone the specified Git repository
cd /home/ubuntu
git clone  https://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@github.com/vaughankraska/de_2_project.git

# Final message
echo "Installation and setup completed successfully!"

