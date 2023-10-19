# Description: Run the paper test for the agnostic instance segmentation method

#!/bin/bash

# echo "Running paper test for austrian_plot"
# bash run_inference.sh ~/data/test_data_agnostic_instanceSeg/original_as_is/austrian_plot/ ~/data/test_data_agnostic_instanceSeg/results/austrian_plot_out/

# echo "Running paper test for english_plot"
# bash run_inference.sh ~/data/test_data_agnostic_instanceSeg/original_as_is/english_plot/ ~/data/test_data_agnostic_instanceSeg/results/english_plot_out/

# echo "Running paper test for for_instance"
# bash run_inference.sh ~/data/test_data_agnostic_instanceSeg/original_as_is/for_instance/ ~/data/test_data_agnostic_instanceSeg/results/for_instance_out/

# echo "Running paper test for german_plot"
# bash run_inference.sh ~/data/test_data_agnostic_instanceSeg/original_as_is/german_plot/ ~/data/test_data_agnostic_instanceSeg/results/german_plot_out/

# echo "Running paper test for mls"
# bash run_inference.sh ~/data/test_data_agnostic_instanceSeg/original_as_is/mls/ ~/data/test_data_agnostic_instanceSeg/results/mls_out/

# # metrics
# echo "Running metrics for austrian_plot"
# python3 metrics/instance_segmentation_metrics_in_folder.py --gt_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/austrian_plot_out/final_results/ --target_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/austrian_plot_out/final_results/ --output_folder_path ~/data/test_data_agnostic_instanceSeg/results/austrian_plot_out/metrics_out --remove_ground --verbose

# echo "Running metrics for english_plot"
# python3 metrics/instance_segmentation_metrics_in_folder.py --gt_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/english_plot_out/final_results/ --target_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/english_plot_out/final_results/ --output_folder_path ~/data/test_data_agnostic_instanceSeg/results/english_plot_out/metrics_out --remove_ground --verbose

# echo "Running metrics for for_instance"
# python3 metrics/instance_segmentation_metrics_in_folder.py --gt_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/for_instance_out/final_results/ --target_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/for_instance_out/final_results/ --output_folder_path ~/data/test_data_agnostic_instanceSeg/results/for_instance_out/metrics_out --remove_ground --verbose

# echo "Running metrics for german_plot"
# python3 metrics/instance_segmentation_metrics_in_folder.py --gt_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/german_plot_out/final_results/ --target_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/german_plot_out/final_results/ --output_folder_path ~/data/test_data_agnostic_instanceSeg/results/german_plot_out/metrics_out --remove_ground --verbose

# echo "Running metrics for mls"
# python3 metrics/instance_segmentation_metrics_in_folder.py --gt_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/mls_out/final_results/ --target_las_folder_path ~/data/test_data_agnostic_instanceSeg/results/mls_out/final_results/ --output_folder_path ~/data/test_data_agnostic_instanceSeg/results/mls_out/metrics_out --remove_ground --verbose

# echo "Done"



#!/bin/bash

# Description: Run the paper test for the agnostic instance segmentation method

BASE_DIR=~/data/test_data_agnostic_instanceSeg
ORIGINAL_DIR=${BASE_DIR}/sparse_1000
RESULTS_DIR=${BASE_DIR}/results_1000

declare -a PLOTS=("austrian_plot" "english_plot" "for_instance" "german_plot" "mls")

# Run inference tests
for PLOT in "${PLOTS[@]}"; do
    echo "Running paper test for ${PLOT}"
    bash run_inference.sh ${ORIGINAL_DIR}/${PLOT}/ ${RESULTS_DIR}/${PLOT}_out/
done

# Run metrics
for PLOT in "${PLOTS[@]}"; do
    echo "Running metrics for ${PLOT}"
    python3 metrics/instance_segmentation_metrics_in_folder.py \
        --gt_las_folder_path ${RESULTS_DIR}/${PLOT}_out/final_results/ \
        --target_las_folder_path ${RESULTS_DIR}/${PLOT}_out/final_results/ \
        --output_folder_path ${RESULTS_DIR}/${PLOT}_out/metrics_out \
        --remove_ground --verbose
done

echo "Done"
