#!/bin/bash
#------------------------------config---------------------------------
for config_version in 101 102
do
    model="sn6_v${config_version}"
    epoch=12
    dataset='sn6'

    echo "================================ New Mode: ${dataset} v${config_version} =============================="

    #------------------------------train-----------------------------------

    if [ $1 == 1 ]
    then
        # train but not debug
        echo "==== start no debug training ===="
        ./tools/dist_train.sh configs/${dataset}/${model}.py 4
    elif [ $1 == 2 ]
    then
        # train and debug
        echo "==== start debug training ===="
        python tools/train.py configs/${dataset}/${model}.py --gpus 1
    elif [ $1 == 0 ]
    then
        # skip training
        echo "==== skip training ===="
    fi


    #------------------------------inference and coco eval-----------------------------------
    if [ $2 == 1 ]
    then
        echo "==== start 4 GPU coco test ===="
        mkdir -p results/${dataset}/${model}

        ./tools/dist_test.sh configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth 4 --out results/${dataset}/${model}/coco_results.pkl --eval bbox segm
    elif [ $2 == 2 ]
    then
        echo "==== start 1 GPU coco test ===="
        export CUDA_VISIBLE_DEVICES=0
        mkdir -p results/${model}

        python tools/test.py configs/${dataset}/${model}.py work_dirs/${model}/epoch_${epoch}.pth --out results/${dataset}/${model}/coco_results.pkl --eval bbox segm
    elif [ $2 == 0 ]
    then
        # read the results file
        echo "==== skip inference ===="
    fi

    #------------------------------F1 Score and CSV generation
    if [ $3 == 1 ]
    then
        echo "==== start training set F1 Score calculating ===="
        python tools/sn6/sn6_evaluation.py --config_version ${model} --imageset train
    elif [ $3 == 2 ]
    then
        echo "==== start testset csv file generation ===="
        python tools/sn6/sn6_test.py --config_version ${model} --imageset test --epoch 24
    elif [ $3 == 0 ]
    then
        # read the results file
        echo "==== skip CSV generation ===="
    fi

    echo "================================ New Mode: ${dataset} v${config_version} =============================="
done

cd ../wwtool
python tools/utils/send_email.py