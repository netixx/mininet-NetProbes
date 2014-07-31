#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
prog="$DIR/start.sh"


#topo="nox-tree-128.json"
#watcherprobe="h128"
#topo="delay-tree.json"
#watcherprobe="h1"
topo=$1
watcherprobe=$2
vars=$3
sims=$4

#"$prog" --no-command --vars dtime=100000ms delay=300ms --start 30 --topo delay-tree --monitor usages/monitor.txt
#set -x

cmd="$prog -q --command '$DIR/start-probe.sh {commandOpts}'  --topo '$HOME/experiments/mininet/data/$topo' --watcher-output '$HOME/netprobes/data/watcher-output/delay.json' --watcher-probe '$watcherprobe' --watcher-post-event '$HOME/netprobes/commander.sh -ip $watcherprobe -c \"watcher delay run\"' --watcher-start-event '$HOME/netprobes/data/watcher-output/events' --watcher-log '$HOME/netprobes/data/logs/watchers/${watcherprobe}watcher.log' --watcher-reset-event '$HOME/netprobes/data/watcher-output/done' --watcher-wait-up '$HOME/netprobes/data/watcher-output/up' --sim-prepend 'python3 $HOME/netprobes/app/commander/main.py -ip $watcherprobe -c \"watcher delay init --options {sim}\"' $vars $sims"

eval "$cmd"
