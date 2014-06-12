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
if [[ -z $3 ]]
then
    delay="10"
else
    delay=$3
fi

if [[ -z $4 ]]
then
    granularity="0.3"
else
    granularity=$4
fi

if [[ -z $5 ]]
then
    randWeight="1"
else
    randWeight=$5
fi

if [[ -z $6 ]]
then
    ipWeight="1"
else
    ipWeight=$6
fi

if [[ -z $7 ]]
then
    delayWeight="1"
else
    delayWeight=$7
fi

if [[ -z $8 ]]
then
    balanceWeight="1"
else
    balanceWeight=$8
fi

watcherprobe=$2
#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
"$prog" -q --command "$DIR/start-probe.sh {commandOpts}" --vars dtime=1000000ms --vars delay="$delay"ms --vars granularity="$granularity" --vars x="$delay" --vars delayWeight="$delayWeight" --vars balanceWeight="$balanceWeight" --vars ipWeight="$ipWeight" --vars randWeight="$randWeight" --topo "nox-$topo" --watcher-output "$HOME/netprobes/data/watcher-output/delay.json" --watcher-probe "$watcherprobe" --watcher-post-event "$HOME/netprobes/commander.sh -ip $watcherprobe -c 'watcher delay run'" --watcher-start-event "$HOME/netprobes/data/watcher-output/events" --watcher-log "$HOME/netprobes/data/logs/watchers/${watcherprobe}watcher.log"
# --auto-start-events 200
