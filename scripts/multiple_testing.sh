#!/bin/bash
base_path="./configs/synthetic_$1/synthetic_$1_"
base_path_generation="./configs/synthetic_$1_generation/synthetic_$1_generation_"

for i in $(seq -w 1 $2);
do
    cfg_path_generation="${base_path_generation}${i}.yaml"
    cfg_path="${base_path}${i}.yaml"
#    echo $cfg_path
    if [ $i -le $3 ];
    then
        case $4 in
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