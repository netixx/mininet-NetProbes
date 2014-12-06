#!/bin/bash

#Start the tree simulation with bandwidth detection

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

granularity="1"
#impose bw of 10 wrt to 100Mbps initially
bw="10"
sampleSizes="5 10 25"
bucket="probabilistic-power-bucket"
#detect 20Mbps difference
x="20"

randWeight="1"
ipWeight="0"
balanceWeight="0"
repeat="1 2 3 4 5 6 7 8"
global_repeat="1"


function getSims() {
sims=""
for rep in ${repeat}
do
    for sampleSize in ${sampleSizes}
    do
     sims="--sim 'balanced-metric-weight=$balanceWeight,ip-metric-weight=$ipWeight,random-metric-weight=$randWeight,sample-size=$sampleSize,bucket-type=$bucket' $sims"
    done
done

echo "$sims"
}

#read variables for this run in file
if [[ -f "$1" ]]; then
    source "$1"
else
    echo "Variables could not be set because vars file was not given or does not exist."
fi

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
run_prog="$THIS_DIR/../run-watcher-simulation-once.sh"
start_dir="$THIS_DIR/../../.."
vars_override="$THIS_DIR/bw-tree-vars"


for gr in ${global_repeat}
do
for topo in ${topos}
do
    for link in ${links}
    do
        vars="--vars dtime=1000000ms --vars bw=$bw --vars sampleSize=5 --vars granularity=$granularity --vars x=$x --vars balanceWeight=0 --vars ipWeight=0 --vars randWeight=1 --vars bucketType=$bucket --vars link=$link --vars overlay_size=$overlay_size"

        sims="$(getSims)"
        ${run_prog} "$start_dir" "$topo" "$watcher" "$vars" "$sims" "$vars_override" 2>&1 | tee -a "$LOG_FILE"
        ret=${PIPESTATUS[0]}
        echo ">>>> Run Finished, returned $ret"
    done
done
done
