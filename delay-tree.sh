#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

if [[ -z $1 ]]
then
    topo="delay-tree"
else
    topo=$1
fi
if [[ -z $2 ]]
then
    delay="10"
else
    delay=$2
fi

#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
"$prog" --command "$DIR/start-probe.sh {commandOpts}" --vars dtime=1000000ms delay="$delay"ms --topo "$topo"
# --auto-start-events 200
