import argparse
import glob
import os
import sys
from typing import Any

from tqdm import tqdm

from nibio_inference.final_utm_integration import FinalUtmIntegration


class FinalUtmIntegrationInFolders(object):
    def __init__(self, 
                 input_data_folder_path,
                 segmented_data_folder_path,
                 output_data_folder_path,
                 verbose=False
                    ):
        self.input_data_folder_path = input_data_folder_path
        self.segmented_data_folder_path = segmented_data_folder_path
        self.output_data_folder_path = output_data_folder_path
        self.verbose = verbose

        # create the output_data_folder_path if it does not exist
        if not os.path.exists(self.output_data_folder_path):
            os.makedirs(self.output_data_folder_path)

    def merge(self):
        # find all the ply files in the input_data_folder_path
        input_data_files = glob.glob(os.path.join(self.input_data_folder_path, '*.ply'))

        # combine the input_data_folder_path and the core names of the files in the input_data_folder_path
        input_data = {os.path.basename(file).split('.')[0]: file for file in input_data_files}


        # find all the ply files in the segmented_data_folder_path
        segmented_data_files = glob.glob(os.path.join(self.segmented_data_folder_path, '*.ply'))

        # combine the segmented_data_folder_path and the core names of the files in the segmented_data_folder_path
        segmented_data = {os.path.basename(file).split('.')[0]: file for file in segmented_data_files}

        # split the segmented_data keys using '_' and remove the first element and join the rest of the elements
        # to get the core name of the file in the segmented_data_folder_path
        segmented_data = {('_'.join(key.split('_')[1:])): value for key, value in segmented_data.items()}


        print('input_data')
        print(input_data)
        print('segmented_data')
        print(segmented_data)


        list_of_matched_tupeles = []
        for item in input_data.keys(): 
            if item in segmented_data.keys():
                list_of_matched_tupeles.append((input_data[item], segmented_data[item]))

        print('list of matched tuples')
        print(list_of_matched_tupeles)


        # merge the point cloud, semantic segmentation and instance segmentation
        for item in tqdm(list_of_matched_tupeles):
            FinalUtmIntegration(
                point_cloud=item[0],
                semantic_segmentation=item[1],
                # save the merged file in the output_data_folder_path with the same name as the input file but .laz extension
                # output_file_path=os.path.join(self.output_data_folder_path, os.path.basename(item[0]).split('.')[0] + '.laz'),
                output_file_path=os.path.join(self.output_data_folder_path, os.path.basename(item[0]).split('.')[0] + '.las'),

                # output_file_path=os.path.join(self.output_data_folder_path, os.path.basename(item[0]).split('.')[0] + '.csv'),
                verbose=self.verbose
                )()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.verbose:
            print('Merging point cloud, semantic segmentation and instance segmentation.')
        return self.merge()


if __name__ == '__main__':
    # use argparse to get the input and output file paths
    parser = argparse.ArgumentParser(description='Merge point cloud, semantic segmentation and instance segmentation.')
    parser.add_argument('-i', '--input_data_folder_path', type=str, help='Path to the input data folder.')
    parser.add_argument('-s', '--segmented_data_folder_path', type=str, help='Path to the segmented data folder.')
    parser.add_argument('-o', '--output_data_folder_path', type=str, help='Path to the output data folder.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')

    # generate help message if no arguments are provided
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = vars(parser.parse_args())

    # get the arguments
    INPUT_DATA_FOLDER_PATH = args['input_data_folder_path']
    SEGMENTED_DATA_FOLDER_PATH = args['segmented_data_folder_path']
    OUTPUT_DATA_FOLDER_PATH = args['output_data_folder_path']
    VERBOSE = args['verbose']

    # run the merge
    merge_pt_ss_is_in_folders = FinalUtmIntegrationInFolders(
        input_data_folder_path=INPUT_DATA_FOLDER_PATH,
        segmented_data_folder_path=SEGMENTED_DATA_FOLDER_PATH,
        output_data_folder_path=OUTPUT_DATA_FOLDER_PATH,
        verbose=VERBOSE
        )()

        