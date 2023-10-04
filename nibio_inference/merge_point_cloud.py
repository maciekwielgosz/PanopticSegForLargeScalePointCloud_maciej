import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply

def merge_pointclouds(chunks_list):
    """
    Merges list of DataFrames representing point clouds into a single DataFrame.
    
    Args:
    - chunks_list (list): List of pd.DataFrame chunks representing point clouds.
    
    Returns:
    - pd.DataFrame: Merged point cloud.
    """
    return pd.concat(chunks_list, ignore_index=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Merge point cloud chunks into a single point cloud.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input directory containing chunked point cloud files.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save merged point clouds.")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grouped_chunks = defaultdict(list)
    
    # Load all point cloud chunks in the directory and group by stem
    for file_name in tqdm(os.listdir(input_dir)):
        input_file = os.path.join(input_dir, file_name)
        if input_file.endswith('.laz') or input_file.endswith('.las'):
            df = las_to_pandas(input_file)
        elif input_file.endswith('.ply'):
            df = ply_to_pandas(input_file)
        else:
            print(f"Skipping unsupported file format: {input_file}")
            continue
        
        # Extracting the stem from the filename (e.g., "data_chunk_1.ply" -> "data")
        stem = "_".join(file_name.split("_")[:-1])
        if "_chunk" in stem:
            stem = stem.replace("_chunk", "")
        
        grouped_chunks[stem].append(df)
        
    # Merge and save for each group
    for stem, chunks in grouped_chunks.items():
        merged_df = merge_pointclouds(chunks)
        output_file = os.path.join(output_dir, f"{stem}.ply")
        pandas_to_ply(merged_df, csv_file_provided=False, output_file_path=output_file)
