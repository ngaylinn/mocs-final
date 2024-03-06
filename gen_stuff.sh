#!/bin/bash

# Define the path to the Python script
VIS_SCRIPT="vis.py"

# Define the experiment file path with "t0" as a template
EXP_FILE="experiments_paper/exp_diamond_4layer/base3_diamond_64.32.16.8/base3_diamond_64.32.16.8_t0.pkl"

# Loop from 1 to 20
for i in {1..20}; do
    # Replace "t0" with "t#" where "#" is the current number in the loop
    EXP_FILE_MODIFIED="${EXP_FILE//t0/t${i}}"

    # Run the vis.py command with the modified experiment file
    python3 $VIS_SCRIPT --exp $EXP_FILE_MODIFIED

    # You can add any additional commands or processing here as needed
done
