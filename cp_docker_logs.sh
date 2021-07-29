#!/bin/bash
# Define a string array as words separated by spaces
ModelNames="BossLevel-1_best GoTo-1_best GoToLocal-GoToObjMaze-1_best GoToObj-1_best GoToObjMaze-1_best GoToRedBallGrey-1_best GoToSeq-1_best Pickup-1_best PickupLoc-1_best PutNext-1_best PutNextLocal-1_best Synth-1_best SynthLoc-1_best UnblockPickup-1_best"

# Iterate the string variable using for loop
for val in $ModelNames; do
    mkdir /data/graceduansu/models/$val
    docker cp busy_brown:babyai/models/$val/model.pt /data/graceduansu/models/$val
    docker cp busy_brown:babyai/models/$val/vocab.json /data/graceduansu/models/$val
done