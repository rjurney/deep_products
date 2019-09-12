#!/bin/bash
set -x -e

# Install all required modules
sudo pip-3.6 install lxml frozendict ipython pandas boto3 bs4 nltk

# Download nltk data
python -m nltk.downloader punkt
python -m nltk.downloader stopwords

# Install Mosh
sudo yum -y install mosh
