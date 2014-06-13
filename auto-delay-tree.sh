#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog=$DIR"/start.sh"

delay="100"
granularity="3"
sampleSize="0.1"
randWeight="1"
ipWeight="1"
delayWeight="1"
balanceWeight="1"
bucketType="ordered-bucket"


if [[ -z $1 ]]
then
    topo="tree-128"
else
    topo=$1
fi

if [[ -z $2 ]]
then
    watcherprobe="h128"
else
    watcherprobe=$2
fi

if [[ -z $3 ]]
then
    vars="--vars dtime=1000000ms --vars delay=${delay}ms --vars granularity=${granularity} --vars x=${delay} --vars delayWeight=${delayWeight} --vars balanceWeight=${balanceWeight} --vars ipWeight=${ipWeight} --vars randWeight=${randWeight} --vars sampleSize=${sampleSize} --vars bucketType=${bucketType}"
else
    vars=$3
fi

#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
"$prog" -q --command "$DIR/start-probe.sh {commandOpts}" ${vars} --topo "$topo" --watcher-output "$HOME/netprobes/data/watcher-output/delay.json" --watcher-probe "$watcherprobe" --watcher-post-event "$HOME/netprobes/commander.sh -ip $watcherprobe -c 'watcher delay run'" --watcher-start-event "$HOME/netprobes/data/watcher-output/events" --watcher-log "$HOME/netprobes/data/logs/watchers/${watcherprobe}watcher.log"
# --auto-start-events 200
