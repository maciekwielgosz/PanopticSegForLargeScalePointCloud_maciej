#!/bin/bash

export PYTHONPATH='/home/nibio/mutable-outside-world'

# provide path to the input and output directories and also information if to clean the output directory from command line
SOURCE_DIR=$1
DEST_DIR=$2
CLEAN_OUTPUT_DIR=$3

# set default values if not provided
if [ -z "$SOURCE_DIR" ]
then
    SOURCE_DIR="/home/nibio/mutable-outside-world/data_for_test"
fi

if [ -z "$DEST_DIR" ]
then
    DEST_DIR="/home/nibio/mutable-outside-world/data_for_test_results"
fi

if [ -z "$CLEAN_OUTPUT_DIR" ]
then
    CLEAN_OUTPUT_DIR="true"
fi

# if someone fails to provide this correctly then print the usage and exit
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CLEAN_OUTPUT_DIR" ]
then
    echo "Usage: run_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>"
    exit 1
fi

# clear the output directory if requested
if [ "$CLEAN_OUTPUT_DIR" = "true" ]
then
    rm -rf $DEST_DIR/*
fi


# rename the input files to the format that the inference script expects 
# change '-' to '_' in the file names
python3 nibio_inference/fix_naming_of_input_files.py $SOURCE_DIR 

# utm normalization 
python3 nibio_inference/pipeline_utm2local_parallel.py -i $SOURCE_DIR -o $DEST_DIR/utm2local

# update the eval.yaml file with the correct paths
cp /home/nibio/mutable-outside-world/conf/eval.yaml $DEST_DIR
python3 nibio_inference/modify_eval.py $DEST_DIR/eval.yaml $DEST_DIR/utm2local $DEST_DIR

# run the inference script with the config file
python3 eval.py --config-name $DEST_DIR/eval.yaml

# rename the output files result_0.ply , result_1.ply, ... to the original file names but with the prefix "inference_"
python3 /home/nibio/mutable-outside-world/nibio_inference/rename_result_files_instance.py $DEST_DIR/eval.yaml $DEST_DIR

# rename segmentation files
python3 /home/nibio/mutable-outside-world/nibio_inference/rename_result_files_segmentation.py $DEST_DIR/eval.yaml $DEST_DIR


FINAL_DEST_DIR=$DEST_DIR/final_results

# remove the old final results directory if it exists
if [ -d "$FINAL_DEST_DIR" ]; then rm -rf $FINAL_DEST_DIR; fi

# run merge script
python3 /home/nibio/mutable-outside-world/nibio_inference/merge_pt_ss_is_in_folders.py -i $DEST_DIR/utm2local -s $DEST_DIR -o $FINAL_DEST_DIR -v

echo "Number of files in the final results directory: $(ls $FINAL_DEST_DIR | wc -l)"




