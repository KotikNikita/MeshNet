#!/bin/bash
echo "Start downloading"
ls -la
sudo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz" -O data.zip && rm -rf /tmp/cookies.txt
ls -la
chmod 777 ./data.zip

sudo unzip data.zip
chmod ./ModelNet40_MeshNet
rm -rf data.zip
rm -rf ./ModelNet40_MeshNet/.DS_Store

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1l8Ij9BODxcD1goePBskPkBcgKW76Ewcs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1l8Ij9BODxcD1goePBskPkBcgKW76Ewcs" -O MeshNet_best_9192.pkl && rm -rf /tmp/cookies.txt
