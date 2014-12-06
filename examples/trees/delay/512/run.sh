#!/bin/bash

#Start the tree simulation with bandwidth detection

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


#### Parameters ####
# file defining variables of simulation : $1
####


"$DIR/../run-delay.sh" "$1"
