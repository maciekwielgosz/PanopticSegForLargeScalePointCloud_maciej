import collections
import laspy
import pandas as pd
import numpy as np

# works with laspy 2.1.2

def pandas_to_las(csv, las_file_path, csv_file_provided=False, verbose=False):
    """
    Convert a pandas DataFrame to a .las file.

    Parameters
    ----------
    csv : pandas DataFrame
        The DataFrame to be converted to .las file.
        But if the csv_file_provided argument is true,
        the csv argument is considered as the path to the .csv file.
    las_file_path : str
        The path to the .las file to be created.
    csv_file_provided : str, optional
        The path to the .csv file to be converted to .las file.
        If None, the csv argument is used instead.
        The default is None.
    """
    # Check if the csv_file_provided argument is provided

    if csv_file_provided:
        df = pd.read_csv(csv)
    else:
        df = csv

    # Check if the DataFrame has the required columns
    required_columns = ['x', 'y', 'z']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'Column {col} not found in {csv}')
        
    # Create a new .las file
    las_header = laspy.LasHeader(point_format=3, version="1.2")

    standard_columns = list(las_header.point_format.dimension_names)

    # check which columns in the csv file are in the standard columns
    csv_standard_columns = list(set(standard_columns) & set(df.columns))

    # add them to required columns if they are not already there
    for col in csv_standard_columns:
        if col not in required_columns:
            required_columns.append(col)

    # read all the colum names from the csv file
    csv_columns = list(df.columns)

    if verbose:
        print('csv_columns: {}'.format(csv_columns))

    # get extra dimensions from target las file
    gt_extra_dimensions = list(set(csv_columns) - set(required_columns))

    # add extra dimensions to new las file
    for item in gt_extra_dimensions:
        las_header.add_extra_dim(laspy.ExtraBytesParams(name=item, type=np.int32))

    outfile = laspy.LasData(las_header)
    
    # Assign coordinates
    for col in required_columns:
        outfile[col] = df[col].values

    # Assign extra dimensions
    for col in gt_extra_dimensions:
        # if you are processing col=return_num limit the values to 0-7
        if col == 'return_num':
            outfile[col] = df[col].values % 8
        else:
            outfile[col] = df[col].values

    # Write the file
    outfile.write(las_file_path)

# Test the function
# CSV_FILE = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek/first_cc.csv'
# NEW_LAS_FILE = '/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/maciek/new_first.las'

# pandas_to_las(CSV_FILE, NEW_LAS_FILE)
