#!/bin/bash

# Path to the eval.yaml file   ###adapt
DEST_DIR="$1"

: "${DEST_DIR:=/home/nibio/mutable-outside-world/data_for_test_results}"

# Base path   ###adapt
base_path="/home/nibio/mutable-outside-world"
# base_path=$(dirname "$DEST_DIR")


yaml_file="$DEST_DIR/eval.yaml"

# Function to read paths from eval.yaml
read_yaml() {
    eval $(python3 -c "
import yaml
with open('$yaml_file') as file:
    config = yaml.safe_load(file)
    paths = config['data']['fold']
    print('large_ply_files=(' + ' '.join(paths) + ')')
")
}

# Read paths from YAML file
read_yaml

# Iterate over each large point cloud file
for i in "${!large_ply_files[@]}"; do
    large_ply_file="${large_ply_files[$i]}"
    file_name=$(basename "$large_ply_file" .ply)
    directory=$(dirname "$large_ply_file")
    index=$((i + 1))

    echo "Processing file $large_ply_file (index $index)..."

    # Step 1: Split large point cloud to tiles
    python3 ${base_path}/split_largePC_to_tiles.py --file_path "$large_ply_file" --tile_size 50 --overlap 5

    # Get the base_dir and ply_output_file_paths.txt path
    tile_size=50
    overlap=5
    new_folder_name="tiles_${tile_size}_${file_name}"
    base_dir="${directory}/${new_folder_name}"
    ply_output_file_paths="${base_dir}/ply_output_file_paths.txt"

    # Clear the eval directory using src_dir before Step 2   ###adapt
    src_dir="${base_path}/outputs/binbin/5_classes/eval"
    rm -rf "${src_dir}"

    # Delete all folders containing "test" in the name
    data_set_path="${base_path}/data_set1_5classes/treeinsfused/"
    test_folders=$(find "$data_set_path" -type d -name '*test*')
    
    echo "Found test folders:"
    echo "$test_folders"
    
    if [ -n "$test_folders" ]; then
        find "$data_set_path" -type d -name '*test*' -exec rm -rf {} +
        echo "Deleted test folders."
    else
        echo "No test folders found."
    fi


    # Step 2: Generate eval commands
    python3 ${base_path}/generate_eval_command.py --ply_output_file "$ply_output_file_paths"

    # Wait for eval commands to complete, add some sleep time to ensure commands are executed
    # sleep 60  # Adjust the time according to the actual situation

    # Step 3: Merge  
    # all_tiles_outputs="${base_path}/outputs/binbin/5_classes/all_tiles_outputs_${file_name}"
    # merged_result="${base_path}/outputs/binbin/5_classes/merged_result_${file_name}.ply"

    all_tiles_outputs="${DEST_DIR}/all_tiles_outputs_${file_name}"
    merged_result="${DEST_DIR}/merged_result_${file_name}.ply"
    
    python3 ${base_path}/merge_tiles.py \
        --src_dir "$src_dir" \
        --ply_output_file "$ply_output_file_paths" \
        --base_dir "$all_tiles_outputs" \
        --original_point_cloud_file "$large_ply_file" \
        --output_file "$merged_result" \
        --overlap "$overlap"

    cp "$merged_result" "${DEST_DIR}/inference_${file_name}.ply"

    rm -rf "${all_tiles_outputs}"
    rm -rf "${merged_result}"
    
    # Clear the eval directory
    rm -rf "${src_dir}"
done
