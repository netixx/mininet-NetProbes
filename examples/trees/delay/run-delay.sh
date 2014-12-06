#!/bin/bash

#Starts the simulation with automatic interaction

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#granularities="0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
#delays="10 20 50 100 200 500"

granularity="1"
delay="100"
sampleSizes="10 20 50"
bucket="probabilistic-power-bucket"
#buckets="ordered-bucket"
x="180"

randWeights="0 1"
ipWeights="0 1"
delayWeights="0"
balanceWeights="0 1"
repeat="1 2"
global_repeat="1 2 3"

#returns 0 if parameters are ok, else > 0
function weightSelection() {
    rand=$1
    balanced=$2
    delay=$3
    ip=$4
    w="$(expr $rand + $balance + $delay + $ip)"
    w2="$(expr $balance + $ip)"
    if [[ "$w" -ge "1" && "$w" -le "2" ]]
    then
        if [[ "$w" -eq "2" && "$w2" -lt "2" ]]
        then
        return 1
        fi
    return 0
    fi

    return 1
}


function getSims() {
sims=""
for rep in ${repeat}
do
    for ipWeight in ${ipWeights}
    do
    for randWeight in ${randWeights}
    do
    for delayWeight in ${delayWeights}
    do
    for balanceWeight in ${balanceWeights}
    do
    if type "weightSelection" > /dev/null
    then
        if ! weightSelection "$randWeight" "$balanceWeight" "$delayWeight" "$ipWeight"
        then
            continue
        fi
    fi
    for sampleSize in ${sampleSizes}
    do
        sims="--sim 'balanced-metric-weight=$balanceWeight,ip-metric-weight=$ipWeight,delay-metric-weight=$delayWeight,random-metric-weight=$randWeight,sample-size=$sampleSize,bucket-type=$bucket' $sims"
    done
    done
    done
done
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
vars_override="$THIS_DIR/delay-tree-vars"

for gr in ${global_repeat}
do
for topo in ${topos}
do
for prog in "$DIR/auto-delay-tree.sh"
do
    for link in ${links}
    do

        vars="--vars dtime=1000000ms --vars delay="$delay"ms --vars sampleSize=10 --vars granularity=$granularity --vars x=$x --vars delayWeight=0 --vars balanceWeight=0 --vars ipWeight=0 --vars randWeight=1 --vars bucketType=$bucket --vars link=$link --vars operator_probe='$operator_probe' --vars overlay_size=$overlay_size"

        sims="$(getSims)"
        ${run_prog} "$start_dir" "$topo" "$watcher" "$vars" "$sims" "$vars_override" 2>&1 | tee -a "$LOG_FILE"
        ret=${PIPESTATUS[0]}
        echo ">>>> Run Finished, returned $ret"
    done
done
done
done
