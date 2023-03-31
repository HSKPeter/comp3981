#!/bin/bash

# Store the output of `ls` in a variable
log_files=$(ls log)

# Get the last filename using string manipulation
last_log_file=${log_files##*/}

# Define color codes
BLUE='\033[0;34m'
NC='\033[0m'

# Print your custom message with the filename included
echo -e "Log file of the most recent exploration is ${BLUE}${last_log_file}${NC}\n"

tail -n 10 log/$last_log_file
