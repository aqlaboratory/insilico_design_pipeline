#!/bin/bash

# Install pipeline package
pip install -e .

# Set up TMscore/TMalign
mkdir -p packages/TMscore
cd packages/TMscore
wget https://zhanggroup.org/TM-score/TMscore.cpp
g++ -O3 -ffast-math -lm -o TMscore TMscore.cpp
chmod +x TMscore
wget https://zhanggroup.org/TM-align/TMalign.cpp
g++ -O3 -ffast-math -lm -o TMalign TMalign.cpp
chmod +x TMalign