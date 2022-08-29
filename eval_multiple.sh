#!/bin/bash


testeval () {
    echo $@
    ds_dirpath=$1
    ckpt=$2
    test_part_name=$3
    test_part_list=$4

    model_name=$(basename $ckpt .pth)
    eval_root_dirpath=$ds_dirpath/predictions/$test_part_name/$model_name
    preds_dirpath=$eval_root_dirpath/predictions

    rm -rf $eval_root_dirpath

    .venv/bin/python test.py -ckpt $ckpt --data_root $ds_dirpath -sl $test_part_list -output $preds_dirpath --bs 16
    .venv/bin/python scripts/plot_confusion_matrix.py $preds_dirpath/error_matrix.txt $preds_dirpath/error_matrix.png

}

for model_ckpt in $(ls saved/*.pth); do
    model_name=$(basename $ckpt .pth)

    if [ model_name = "real" ]; then
        testeval data/sets/real $model_ckpt retrain data/sets/real/partitions/test.csv
    else
        testeval data/sets/real $model_ckpt full-test data/sets/real/partitions/test_full.csv
    fi
    testeval data/sets/nbid $model_ckpt real-synth data/sets/nbid/partitions/$model_name.csv
done

DS_DIR_PATH=data/sets/nbid
for test_part_list in $(ls $DS_DIRPATH/partitions); do
    testeval $DS_DIR_PATH saved/full.pth real-synth $DS_DIRPATH/partitions/$test_part_list
done
