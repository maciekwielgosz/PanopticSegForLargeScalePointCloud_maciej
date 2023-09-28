import argparse
import json
import sys
import jaklas
import laspy

import pandas as pd

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.pandas_to_las import pandas_to_las


class MergePtSsIs(object):
    def __init__(self, 
                 point_cloud, 
                 semantic_segmentation, 
                 instance_segmentation, 
                 output_file_path,
                 verbose=False
                 ):
        
        self.point_cloud = point_cloud
        self.semantic_segmentation = semantic_segmentation
        self.instance_segmentation = instance_segmentation
        self.output_file_path = output_file_path
        self.verbose = verbose


    def merge(self):
        if self.verbose:
            print('Merging point cloud, semantic segmentation and instance segmentation.')
        # read point cloud
        # point_cloud_df = las_to_pandas(self.point_cloud)

        point_cloud_df = ply_to_pandas(self.point_cloud)

        # change header names from X, Y, Z which are the default names in the las file to x, y, z
        if 'X' in point_cloud_df.columns:
            point_cloud_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)

        # read semantic segmentation
        semantic_segmentation_df = ply_to_pandas(self.semantic_segmentation)

        if 'X' in semantic_segmentation_df.columns:
            semantic_segmentation_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)

        # change all the names of the columns to have the suffix _semantic_segmentation except the x, y, z columns
        semantic_segmentation_df.columns = [f'{col}_semantic_segmentation' if col not in ['x', 'y', 'z'] else col for col in semantic_segmentation_df.columns]

        # read instance segmentation
        instance_segmentation_df = ply_to_pandas(self.instance_segmentation)

        if 'X' in instance_segmentation_df.columns:
            instance_segmentation_df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)

        # change all the names of the columns to have the suffix _instance_segmentation except the x, y, z columns
        instance_segmentation_df.columns = [f'{col}_instance_segmentation' if col not in ['x', 'y', 'z'] else col for col in instance_segmentation_df.columns]

        # create a new data frame with only the columns that contain pred in the name and the x, y, z columns
        # semantic_segmentation_df = semantic_segmentation_df[[col for col in semantic_segmentation_df.columns if 'pred' in col] + ['x', 'y', 'z']]
        # merge point cloud with semantic segmentation in a way that takes sum of the columns
        merged_df = pd.merge(point_cloud_df, semantic_segmentation_df, on=['x', 'y', 'z'], how='outer')

        # create a new data frame with only the columns that contain pred in the name and the x, y, z columns
        # instance_segmentation_df = instance_segmentation_df[[col for col in instance_segmentation_df.columns if 'pred' in col] + ['x', 'y', 'z']]

        merged_df = pd.merge(merged_df, instance_segmentation_df, on=['x', 'y', 'z'], how='outer')

        # bring back to utm coordinates
        min_values_path = self.point_cloud.replace('.ply', '_min_values.json')

        with open(min_values_path, 'r') as f:
            min_values = json.load(f)
        
        min_x, min_y, min_z = min_values

        # add the min values back to x, y, z
        merged_df['x'] = merged_df['x'].astype(float) + min_x
        merged_df['y'] = merged_df['y'].astype(float) + min_y
        merged_df['z'] = merged_df['z'].astype(float) + min_z

        # remove duplicate columns
        # merged_df = merged_df.T.drop_duplicates().T

        return merged_df
    
    def save(self, merged_df):
        # save the merged data frame to a file
        # merged_df.to_csv(self.output_file_path, index=False)

        # save the merged data frame to a file
        # pandas_to_ply(
        #     merged_df,
        #     csv_file_provided=False,
        #     output_file_path=self.output_file_path
        #     )
        
        # pandas_to_las(
        #     merged_df,
        #     csv_file_provided=False,
        #     output_file_path=self.output_file_path
        #     )
        
        # save the merged data frame to a file using jaklas as a .las file
        jaklas.write(merged_df, self.output_file_path, point_format=2, scale=(0.001, 0.001, 0.001))

        # convert to laz
        las = laspy.read(self.output_file_path)
        las.write(self.output_file_path.replace('.las', '.laz'), do_compress=True)


    def run(self):
        if self.verbose:
            print('point_cloud: {}'.format(self.point_cloud))
            print('semantic_segmentation: {}'.format(self.semantic_segmentation))
            print('instance_segmentation: {}'.format(self.instance_segmentation))
            print('output_file_path: {}'.format(self.output_file_path))
        merged_df = self.merge()
        if self.output_file_path is not None:
            self.save(merged_df)
        return merged_df
    
    def __call__(self):
        return self.run()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge point cloud, semantic segmentation and instance segmentation.')
    parser.add_argument('-pc', '--point_cloud', help='Path to the point cloud file.')
    parser.add_argument('-ss', '--semantic_segmentation', help='Path to the semantic segmentation file.')
    parser.add_argument('-is', '--instance_segmentation', help='Path to the instance segmentation file.')
    parser.add_argument('-o', '--output_file_path', help='Path to the output file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    
    # generate help message if no arguments are provided
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = vars(parser.parse_args())

    # get the arguments
    POINT_CLOUD = args['point_cloud']
    SEMANTIC_SEGMENTATION = args['semantic_segmentation']
    INSTANCE_SEGMENTATION = args['instance_segmentation']
    OUTPUT_FILE_PATH = args['output_file_path']
    VERBOSE = args['verbose']



    # run the merge
    merge_pt_ss_is = MergePtSsIs(
        point_cloud=POINT_CLOUD,
        semantic_segmentation=SEMANTIC_SEGMENTATION,
        instance_segmentation=INSTANCE_SEGMENTATION,
        output_file_path=OUTPUT_FILE_PATH,
        verbose=VERBOSE
        )()
    


    

    


