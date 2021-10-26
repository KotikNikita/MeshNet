#!/bin/bash
echo "Start downloading"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1o9pyskkKMxuomI5BWuLjCG2nSv5iePZz" -O data.zip && rm -rf /tmp/cookies.txt

unzip data.zip

rm -rf data.zip
rm -rf ./ModelNet40_MeshNet/.DS_Store
git clone https://github.com/iMoonLab/MeshNet

sed -i 's!ModelNet40_MeshNet/!../ModelNet40_MeshNet/!' ./MeshNet/config/test_config.yaml

sed -i 's!ModelNet40_MeshNet/!../ModelNet40_MeshNet/!' ./MeshNet/config/train_config.yaml

sed -i 's!MeshNet_best_9192.pkl!../MeshNet_best_9192.pkl!' ./MeshNet/config/test_config.yaml

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1l8Ij9BODxcD1goePBskPkBcgKW76Ewcs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1l8Ij9BODxcD1goePBskPkBcgKW76Ewcs" -O MeshNet_best_9192.pkl && rm -rf /tmp/cookies.txt




