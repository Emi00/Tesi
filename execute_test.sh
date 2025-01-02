#!/bin/bash



if [ $# -lt 6 ]; then
    echo "Usage: $0 algo flag version order dim rounds"
    exit 1 
fi

algo=${1}
flag=${2}
version=${3}
order=${4}
dim=${5}
rounds=${6}
file=tmp_${algo}_${dim}_${flag}_v${version}_$order
rm data/$file


g++ $algo.cpp -o $algo -$flag -march=znver5
echo $algo $flag rounds $rounds dim $dim algo v$version
for round in $(seq 1 $rounds)
do
    ./$algo $dim $version 0 $order >> data/$file
    sleep 1
done

./statistics.py data/$file

exit 0