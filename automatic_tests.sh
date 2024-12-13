#!/bin/bash


rounds=100
dim_array=10 #(10 100 1000 10000 100000)
flags=("O1" "O2" "O3")
orders=(0 1 2)


for flag in ${flags[@]}
do
    # SelectionSort
    versions=("v0" "v1" "v2" "v3" "v4" "v5")

    g++ selectionSort.cpp -o selectionSort -$flag -march=znver5

    for dim in ${dim_array[@]}
    do
        for order in ${orders[@]}
        do

            for version in 0 1 2 3 4 5
            do
                echo selectionSort $flag rounds $rounds dim $dim algo ${versions[$version]}
                file=tmp_selectionSort_${dim}_${flag}_${versions[$version]}_$order
                rm data/$file
                for i in $(seq 1 $rounds)
                do
                    ./selectionSort $dim $version 0 $order >> data/$file
                    sleep 0.01
                done
                ./statistics.py data/$file
            done
        done
    done
    # BubbleSort
    versions=("v0" "v1" "v2")

    g++ bubbleSort.cpp -o bubbleSort -$flag -march=znver5

    for dim in ${dim[@]}
    do
        for order in ${orders[@]}
        do
            for version in 0 1 2
            do
                echo bubbleSort $flag rounds $round dim $dim algo ${versions[$version]}
                file=tmp_bubbleSort_${dim}_${flag}_${versions[$version]}_$order
                rm data/$file
                for i in $(seq 1 $rounds)
                do
                    ./bubbleSort $dim $version 0 $order >> data/$file
                    sleep 0.01
                done
                ./statistics data/$file
            done
        done
    done
done
