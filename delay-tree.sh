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

if [[ -z $3 ]]
then
    granularity="0.3"
else
    granularity=$3
fi

#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
"$prog" --command "$DIR/start-probe.sh {commandOpts}" --vars dtime=1000000ms --vars delay="$delay"ms --vars granularity="$granularity" --vars x="$delay" --topo "$topo"
# --auto-start-events 200
