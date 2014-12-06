#!/bin/bash

# Start the tree simulation with bandwidth watcher

#### Parameters ####
# directory where start.sh and start-probe.sh are: $1
# topology json to use : $2
# id of the watcher probe : $3
# variables for $substitution in json : $4
# parameters for multiple simulation to run : $5
# file to source to override default values (optionnal) : $6
####

start_dir=$1
topo=$2
watcherprobe=$3
vars=$4
sims=$5


#### Default values ####
watcher_output="$HOME/netprobes/data/watcher-output/bandwidth.json"
commander_prog="$HOME/netprobes/commander.sh"
watcher_start_event="$HOME/netprobes/data/watcher-output/events"
watcher_log="$HOME/netprobes/data/logs/watchers/${watcherprobe}watcher.log"
watcher_reset_event="$HOME/netprobes/data/watcher-output/done"
watcher_wait_up="$HOME/netprobes/data/watcher-output/up"
watcher_type="delay"
sim_prepend_command="$commander_prog -ip $watcherprobe -c \"watcher bandwidth init --options {sim}\""

if [[ ! -z $6 ]] && [[ -f "$6" ]]; then
    source "$6"
fi


mininet_prog="$start_dir/start.sh"
start_probe_prog="$start_dir/start-probe.sh"


# Prepare command (quoting, default values...)
cmd="$mininet_prog -q --command '$start_probe_prog {commandOpts}'  --topo '$topo' --watcher-output '
$watcher_output' --watcher-probe '$watcherprobe' --watcher-post-event '$commander_prog -ip $watcherprobe -c \"watcher bandwidth run\"' --watcher-start-event '$watcher_start_event' --watcher-log '$watcher_log' --watcher-reset-event '$watcher_reset_event' --watcher-wait-up '$watcher_wait_up' --watcher-type '$watcher_type' --sim-prepend '$sim_prepend_command' $vars $sims"


# Start the simulation (with eval, for quote resolution)
eval "$cmd"
