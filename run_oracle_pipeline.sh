#!/bin/bash

# Set DEBUG_MODE (change this to true or false as needed)
DEBUG_MODE=false

# Set the path (change this to the path taken from the config file)
# if [ "$DEBUG_MODE" = true ]; then
#     PATH_DATA='/home/nibio/mutable-outside-world' #TODO: change this to the path
# else
#     PATH_DATA='/home/datascience'
# fi

PATH_DATA='/home/datascience'

# Set the input and output folders in the oracle
ORACLE_IN_DATA_FOLDER="$PATH_DATA/docker_in_folder" # This is the folder where the input data is stored on the oracle
ORACLE_OUT_DATA_FOLDER="$PATH_DATA/docker_out_folder" # This is the folder where the output data is stored on the oracle

TMP_IN_DATA_FOLDER="$PATH_DATA/tmp_in_folder" # This is the folder where the input data is stored on the oracle temporarily
TMP_OUT_DATA_FOLDER="$PATH_DATA/tmp_out_folder" # This is the folder where the output data is stored on the oracle temporarily

# Set the input and output folders which mimic the bucket
DOCKER_IN_FOLDER='/home/nibio/mutable-outside-world/bucket_in_folder' # this just mimics the input bucket
DOCKER_OUT_FOLDER='/home/nibio/mutable-outside-world/bucket_out_folder' # this just mimics the output bucket

# function to read the input from the oracle
run_oracle_wrapper_input() {
    if [ "$DEBUG_MODE" = true ]; then
        # This is mapped in the docker run
        bucket_location=${DOCKER_IN_FOLDER}
    else
        # Get the input location from the environment variable
        bucket_location=${OBJ_INPUT_LOCATION}

        # Remap the input location
        bucket_location=${bucket_location//@axqlz2potslu/}
        bucket_location=${bucket_location//oci:\/\//\/mnt\/}
    fi

    # Create the input folder if it does not exist in the docker container
    mkdir -p "$ORACLE_IN_DATA_FOLDER"

    # Copy files from bucket_location to the input folder
    shopt -s nullglob # Enable nullglob to handle empty directories
    cp -r "$bucket_location"/* "$ORACLE_IN_DATA_FOLDER"
}

# function to write the output to the oracle
run_oracle_wrapper_output() {
    if [ "$DEBUG_MODE" = true ]; then
        # This is mapped in the docker run
        bucket_location=${DOCKER_OUT_FOLDER}
    else
        # Get the output location from the environment variable
        bucket_location=${OBJ_OUTPUT_LOCATION}

        # Remap the output location
        bucket_location=${bucket_location//@axqlz2potslu/}
        bucket_location=${bucket_location//oci:\/\//\/mnt\/}
    fi

    # Create the output folder if it does not exist in the docker container
    mkdir -p "$bucket_location"

    # Zip the output folder
    zip -r "$ORACLE_OUT_DATA_FOLDER/final_results.zip" "$ORACLE_OUT_DATA_FOLDER/final_results"

    # Copy the zipped folder to the output_location
    cp "$ORACLE_OUT_DATA_FOLDER/final_results.zip" "$bucket_location"
}

### Main execution ###

# Run the input script
run_oracle_wrapper_input

# # Run the inference script
bash run_inference.sh "$ORACLE_IN_DATA_FOLDER" "$ORACLE_OUT_DATA_FOLDER"

# # Run the output script
run_oracle_wrapper_output
