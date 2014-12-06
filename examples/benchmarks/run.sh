#!/bin/bash


#Runs topology tree.json which includes benchmarks of delay and bw measurement tools


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


start="$DIR/../../start.sh"
topo="$DIR/tree.json"


${start} --no-command -cc --topo "$topo"
