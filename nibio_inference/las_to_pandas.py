import numpy as np
import pandas as pd
import laspy

# works with laspy 2.1.2

def las_to_pandas(las_file_path, csv_file_path=None):
    """
    Reads a LAS file and converts it to a pandas dataframe, then saves it to a CSV file.

    Args:
        las_file_path (str): The path to the LAS file to be read.
        csv_file_path (str): The path to the CSV file to be saved.

    Returns:
        DataFrame: Pandas DataFrame containing the LAS data.
    """
    file_content = laspy.read(las_file_path)

    # Put x, y, z, label into a numpy array
    basic_points = np.vstack((file_content.x, file_content.y, file_content.z)).T

    # # multiple x, y, z by scale factor
    # basic_points[:, 0] = basic_points[:, 0] * file_content.header.scale[0]
    # basic_points[:, 1] = basic_points[:, 1] * file_content.header.scale[1]
    # basic_points[:, 2] = basic_points[:, 2] * file_content.header.scale[2]

    # Fetch any extra dimensions
    gt_extra_dimensions = list(file_content.point_format.extra_dimension_names)

    if gt_extra_dimensions:
        extra_points = np.vstack([getattr(file_content, dim) for dim in gt_extra_dimensions]).T
        # Combine basic and extra dimensions
        all_points = np.hstack((basic_points, extra_points))
        all_columns = ['x', 'y', 'z'] + gt_extra_dimensions
    else:
        all_points = basic_points
        all_columns = ['x', 'y', 'z']

    # Create dataframe
    points_df = pd.DataFrame(all_points, columns=all_columns)

    # Save pandas dataframe to csv
    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False, header=True, sep=',')

    return points_df