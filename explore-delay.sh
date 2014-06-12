#!/bin/bash

#Start the tree simulation @ data/delay-tree.json

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#granularities="0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
#delays="10 20 50 100 200 500"

granularities="0.1 0.3 0.5 0.8 1.0"
delays="10 20 50 100 200"

randWeights="0 1"
ipWeights="0 1"
delayWeights="0 1"
balanceWeights="0 1"
repeat="1 2"
exclusive="1"
#granularities="0.1 0.5"
#delays="10 20"

for rep in repeat
do
    for prog in "$DIR/auto-delay-tree.sh"
    do
        for ipWeight in ${ipWeights}
        do
            for randWeight in ${randWeights}
            do
                for delayWeight in ${delayWeights}
                do
                    for balanceWeight in ${balanceWeights}
                    do
                        if [[ "$exclusive" == "1" ]]
                        then
                            if [[ "$(expr $randWeight + $balanceWeight + $delayWeight + $ipWeight)" != "1" ]]
                            then
                                continue
                            fi
                        fi
                        for granularity in ${granularities}
                        do
                            for delay in ${delays}
                            do

                                echo ">>>> Run simulation $prog for ipWeight=$ipWeight, randWeight=$randWeight, delayWeight=$delayWeight, balanceWeight=$balanceWeight, granularity=$granularity, delay=$delay"
                                $prog "tree-128" "h128" "$delay" "$granularity" "$randWeight" "$ipWeight" "$delayWeight" "$balanceWeight"
                                ret=$?
                                echo ">>>> Run Finished, returned $ret"
                            done
                        done
                    done
                done
            done
        done
    done
done
