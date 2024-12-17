#!/bin/bash


rounds=20
dim_array=1000 #(10 100 1000 10000 100000)
flags=("O1" "O2" "O3")
orders=(0 3) #(0 1 2 3)


for flag in ${flags[@]}
do
    # SelectionSort
    versions=("v0" "v1" "v2" "v3" "v4" "v5" "v6" "v7")

    g++ selectionSort.cpp -o selectionSort -$flag -march=znver5

    for dim in ${dim_array[@]}
    do
        for order in ${orders[@]}
        do

            for version in 0 5  #0 1 2 3 4 5 6 7
            do
                echo selectionSort $flag rounds $rounds dim $dim algo ${versions[$version]}
                file=tmp_selectionSort_${dim}_${flag}_${versions[$version]}_$order
                rm data/$file
                for i in $(seq 1 $rounds)
                do
                    ./selectionSort $dim $version 0 $order >> data/$file
                    sleep 0.2
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
            for version in 0 3 #0 1 2 3
            do
                echo bubbleSort $flag rounds $rounds dim $dim algo ${versions[$version]}
                file=tmp_bubbleSort_${dim}_${flag}_${versions[$version]}_$order
                rm data/$file
                for i in $(seq 1 $rounds)
                do
                    ./bubbleSort $dim $version 0 $order >> data/$file
                    sleep 0.2
                done
                ./statistics.py data/$file
            done
        done
    done

    # InsertionSort
    versions=("v0" "v1" "v2")
    g++ insertionSort.cpp -o insertionSort -$flag -march=znver5
    for dim in ${dim[@]}
    do
        for order in ${orders[@]}
        do
            for version in 0 2
            do
                echo insertionSort $flag rounds $rounds dim $dim algo ${versions[$version]}
                file=tmp_insertionSort_${dim}_${flag}_${versions[$version]}_$order
                rm data/$file
                for i in $(seq 1 $rounds)
                do
                    ./insertionSort $dim $version 0 $order >> data/$file
                    sleep 0.2
                done
                ./statistics.py data/$file
            done
        done
    done
done
