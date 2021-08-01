#!/bin/bash
# Define a string array as words separated by spaces
ModelNames="events.out.tfevents.1627507784.72ee09c3799e log.csv log.log status.json"
MyDir = "/data/graceduansu/logs/docker_baseline_seed1_32768"
# Iterate the string variable using for loop
for val in $ModelNames; do
    docker cp 6b6ec68f5b44:babyai/logs/BabyAI-GoToImpUnlock-v0_seed1_32768/$val $MyDir/$val
done