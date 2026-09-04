#!/bin/bash
set -e
yum update -y
yum install -y python3 python3-pip git
pip3 install --upgrade pip
mkdir -p /home/ec2-user/experiments
chown -R ec2-user:ec2-user /home/ec2-user/experiments
