import argparse
import json
import os
from joblib import Parallel, delayed


# local imports
from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.pandas_to_las import pandas_to_las

class PrepareFinal(object):
    def __init__(self, input_folder, output_folder, utf_folder, verbose=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.utf_folder = utf_folder    
        self.verbose = verbose
  
    
    def process_files(self):
        filenames = os.listdir(self.input_folder)
        # get olny the ply files
        filenames = [filename for filename in filenames if filename.endswith('.ply')]
        
        print(f"filenames: {filenames}")
        
        if self.verbose:
            print(f"Processing {len(filenames)} files...")
        Parallel(n_jobs=4)(
            delayed(self.process_file)(filename) for filename in filenames
        )
        
        if self.verbose:
            print(f"Output files are saved in: {self.output_folder}")
        
    def process_file(self, filename):
        # read ply to pandas
        input_file_path = os.path.join(self.input_folder, filename)
        print(f"Processing in prepare_final: {input_file_path}")
        
        points_df = ply_to_pandas(input_file_path)
        
        min_values_file_name = filename.replace('.ply', '_min_values.json')
        
        # remove 'inferece_' from the beginning of the filename
        min_values_file_name = min_values_file_name.replace('inference_', '')
        
        min_values_path = os.path.join(self.utf_folder, min_values_file_name)
        
        with open(min_values_path, 'r') as f:
            min_values = json.load(f)
        
        min_x, min_y, min_z = min_values
        points_df['x'] = points_df['x'].astype(float) + min_x
        points_df['y'] = points_df['y'].astype(float) + min_y
        points_df['z'] = points_df['z'].astype(float) + min_z

        # add 1 to PredInstance column
        points_df['instance_preds'] = points_df['instance_preds'] + 1
        
        # save the file to las
        pandas_to_las(
            points_df,
            csv_file_provided=False,
            output_file_path=os.path.join(self.output_folder, filename.replace('.ply', '.laz').replace('inference_', '')),
            do_compress=True,
            verbose=self.verbose
            )
        
        
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Process las or laz files and save results as ply files.')
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder containing ply files.')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder to save las files.')
    parser.add_argument('-u', '--utf_folder', type=str, help='Path to the utf folder.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    prepare_final = PrepareFinal(args.input_folder, args.output_folder, args.utf_folder, args.verbose)
    
    prepare_final.process_files()
    
        
# Example usage:
# python3 /home/nibio/mutable-outside-world/nibio_inference/prepare_final.py -u "data_for_test_results/utm2local" -i "data_for_test_results" -o "data_for_test_results/final_results" -v