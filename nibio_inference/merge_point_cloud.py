import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from joblib import Parallel, delayed


from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply

from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd


def distance(point1, point2):
    """Compute the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

def compare_chunks(border_trees, other_chunk, prediction, threshold_distance=0.02):
    for idx, row in border_trees.iterrows():
        for other_idx, other_row in other_chunk.iterrows():
            dist = distance((row['x'], row['y'], row['z']), (other_row['x'], other_row['y'], other_row['z']))
            if dist < threshold_distance:
                unified_label = min(row[prediction], other_row[prediction])
                border_trees.loc[idx, prediction] = unified_label
                other_chunk.loc[other_idx, prediction] = unified_label
    return border_trees, other_chunk

def unify_labels(chunks_list, prediction='preds', threshold_distance=0.02):
    n_jobs = -1  # Use all available cores
    results = []

    for i, chunk in enumerate(chunks_list):
        border_trees = chunk[(chunk['x_border'] == 1) | (chunk['y_border'] == 1)]
        results.extend(Parallel(n_jobs=n_jobs)(
            delayed(compare_chunks)(border_trees, other_chunk, prediction, threshold_distance) 
            for j, other_chunk in enumerate(chunks_list) if i != j
        ))

    # Update chunks_list with the results
    for updated_border_trees, updated_other_chunk in results:
        chunks_list[i].update(updated_border_trees)
        chunks_list[j].update(updated_other_chunk)

    return chunks_list


def merge_pointclouds(chunk_name, chunks_list, prediction='preds', verbose=False):
    """
    Merges list of DataFrames representing point clouds into a single DataFrame.
    
    Args:
    - chunks_list (list): List of pd.DataFrame chunks representing point clouds.
    
    Returns:
    - pd.DataFrame: Merged point cloud.
    """
    print("chunk number: ", len(chunks_list))

    def process_chunk(chunk):
        centroid = chunk[['x', 'y', 'z']].mean()
        centroid = [(int(x * 1000 + 0.5) / 1000) for x in centroid]  # round to 3 decimal places
        centroid = tuple(centroid)
        unique_values = chunk[prediction].unique()
        number_of_unique_values = len(unique_values)

         # Identify points close to the borders
        x_min, x_max = chunk['x'].min(), chunk['x'].max()
        y_min, y_max = chunk['y'].min(), chunk['y'].max()

        chunk['x_border'] = ((chunk['x'] <= x_min + 0.02) | (chunk['x'] >= x_max - 0.02)).astype(int)
        chunk['y_border'] = ((chunk['y'] <= y_min + 0.02) | (chunk['y'] >= y_max - 0.02)).astype(int)


        return [centroid, unique_values, number_of_unique_values]

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_chunk)(chunk) for chunk in tqdm(chunks_list, desc="Computing centroids")
    )

    results_df = pd.DataFrame(results, columns=["centroid", "unique_values", "number_of_unique_values"])

    # Generate a new prediction column for each chunk
    start_value = 0
    for index, row in results_df.iterrows():
        chunk = chunks_list[index]
        mapping = {old_val: new_val for old_val, new_val in zip(row['unique_values'], range(start_value, start_value + row['number_of_unique_values']))}
        chunk[prediction] = chunk[prediction].map(mapping)
        start_value += row['number_of_unique_values']

    # add new prediction column to the results_df
    results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_chunk)(chunk) for chunk in tqdm(chunks_list, desc="Computing centroids")
        )

    results_df = pd.DataFrame(results, columns=["centroid", "unique_values", "number_of_unique_values"])


    chunks_list = unify_labels(chunks_list, prediction=prediction)
    # save the results_df to csv
    results_df.to_csv(f"{chunk_name}_results.csv")


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
        merged_df = merge_pointclouds(stem, chunks)
        output_file = os.path.join(output_dir, f"{stem}.ply")
        pandas_to_ply(merged_df, csv_file_provided=False, output_file_path=output_file)
