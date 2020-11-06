#!/bin/bash
NUM_TESTS=20
base_path="./configs/synthetic_clusters/synthetic_clusters_"
base_path_generation="./configs/synthetic_generation_clusters/synthetic_generation_clusters_"

for i in $(seq -w 1 $NUM_TESTS);
do
    cfg_path_generation="${base_path_generation}${i}.yaml"
    cfg_path="${base_path}${i}.yaml"

#    echo $cfg_path
    if [ $i -le $1 ];
    then
        case $2 in
            -g)
                echo "Using file: ${cfg_path_generation} to generate data.";
                python3 kcurves/generate_synthetic.py $cfg_path_generation;
            ;;
            -t)
                echo "Using file: ${cfg_path} to perform training and test.";
                python3 kcurves train $cfg_path;
                python3 kcurves test $cfg_path;
            ;;
            -a)
                echo "Using file: ${cfg_path_generation} to generate data.";
                python3 kcurves/generate_synthetic.py $cfg_path_generation;
                echo "Using file: ${cfg_path} to perform training and test.";
                python3 kcurves train $cfg_path;
                python3 kcurves test $cfg_path;
            ;;
        esac
        echo "------------------------------------------------------";
    fi

done