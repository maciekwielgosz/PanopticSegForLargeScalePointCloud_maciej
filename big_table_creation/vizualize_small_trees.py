import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse


from scipy.spatial import ConvexHull



import numpy as np

from nibio_inference.las_to_pandas import las_to_pandas
import json

HIGH_RANGES = [0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

# convert HIGH_RANGES to dictionary with keys as index and values as ranges so they sound nice like '0-5'
HIGH_RANGES_DICT = {}
for i in range(len(HIGH_RANGES) - 1):
    HIGH_RANGES_DICT[i] = f'{HIGH_RANGES[i]}-{HIGH_RANGES[i+1]}'



class VizualizeSmallTrees(object):

    def __init__(self) -> None:
        pass

    def compute_mean_metrics(self, gt_trees, df_csv):

        # exclude row from df_csv where 'pred_label' is 0
        df_csv = df_csv.loc[df_csv['pred_label'] != 0]

        # exclude '0' from gt_trees
        gt_trees = [x for x in gt_trees if x != 0]

        # compute mean for each metric
        mean_rmse = df_csv['rmse_hight'].mean()
        mean_f1 = df_csv['f1_score'].mean()

        # compute for trees_ok as the number of trees that IoU is greater than 0.5
        trees_ok_detected = df_csv.loc[df_csv['IoU'] > 0.5].shape[0]

        # compute detection rate as the number of trees that IoU is greater than 0.5 divided by the number of trees in the dataset
        if len(gt_trees) == 0:
            detection_rate = 0
        else:
            detection_rate = trees_ok_detected / len(gt_trees)

        ommision_rate = 1 - detection_rate

        if df_csv.shape[0] == 0:
            commission_rate = 0
        else:
            commission_rate = df_csv.loc[df_csv['IoU'] <= 0.5].shape[0] / df_csv.shape[0]

        # put metrics in a dictionary
        dict_metrics = {}
        # limit the number of decimals to 3
        dict_metrics['mean_rmse'] = round(mean_rmse, 3)
        dict_metrics['mean_f1'] = round(mean_f1, 3)
        dict_metrics['detection_rate'] = float('{:.3f}'.format(detection_rate))
        dict_metrics['ommision_rate'] = round(ommision_rate, 3)
        dict_metrics['commission_rate'] = round(commission_rate, 3)
        dict_metrics['num_gt_trees'] = len(gt_trees)
        dict_metrics['num_predicted_trees'] = df_csv.shape[0]
        dict_metrics['num_trees_ok_detected'] = trees_ok_detected
        dict_metrics['gt_trees'] = gt_trees
        dict_metrics['predicted_trees'] = [x-1 for x in df_csv['pred_label'].tolist()] # subtract 1 from pred_label to get the index of the tree
        dict_metrics['predicted_trees_ok'] = [x - 1 for x in df_csv.loc[df_csv['IoU'] > 0.5]['pred_label'].tolist()] # subtract 1 from pred_label to get the index of the tree

        return dict_metrics


    def read_single_csv_and_parse(self, laz_and_csv_file):

        laz_file_path, csv_file_path = laz_and_csv_file

        # read laz file to df_laz
        df_laz = las_to_pandas(laz_file_path)

        # subtract df_laz['Z'].min() from df_laz['Z'] to get the height of the tree
        # df_laz['Z'] = df_laz['Z'] - df_laz['Z'].min()

        gt_trees_with_height_max = df_laz[['treeID', 'Z']].groupby('treeID').max().reset_index()
        gt_trees_with_height_min = df_laz[['treeID', 'Z']].groupby('treeID').min().reset_index()

        # compute the tree hight by subtracting the min z value from the max z value grouped by treeID
        gt_trees_with_height = gt_trees_with_height_max.copy()
        gt_trees_with_height['Z'] = gt_trees_with_height_max['Z'] - gt_trees_with_height_min['Z']


        
        # compute the tree hight by subtracting the min z value from the max z value grouped by treeID
        

        # convert gt_trees_with_height to list of tuples
        gt_trees_with_height = list(zip(gt_trees_with_height['treeID'].tolist(), gt_trees_with_height['Z'].tolist()))

        print('gt_trees_with_height: ', gt_trees_with_height)

        df_csv = pd.read_csv(csv_file_path)

        # remove unaamed columns
        df_csv = df_csv.loc[:, ~df_csv.columns.str.contains('^Unnamed')]


        range_dict = {}

        for key, value in HIGH_RANGES_DICT.items():
            df_csv_in_range = df_csv.loc[(df_csv['high_of_tree_gt'] >= HIGH_RANGES[key]) & (df_csv['high_of_tree_gt'] < HIGH_RANGES[key+1])]

            # find the gt trees in gt_trees_with_height that are in the range
            gt_trees_in_range = [x[0] for x in gt_trees_with_height if x[1] >= HIGH_RANGES[key] and x[1] < HIGH_RANGES[key+1]]

            gt_trees = gt_trees_in_range

            # # find the max z value of df_laz
            # df_dict_laz_tmp = df_laz.loc[(df_laz['Z'] >= HIGH_RANGES[key]) & (df_laz['Z'] < HIGH_RANGES[key+1])]
            # # get treeID of gt trees
            # gt_trees =  df_dict_laz_tmp['treeID'].unique()

            # compute mean metrics    
            dict_metrics = self.compute_mean_metrics(gt_trees, df_csv_in_range)

            # add dict_metrics to range_dict
            range_dict[value] = dict_metrics
   
        # print range_dict in a pretty way
        # print(json.dumps(range_dict, indent=4))

        for key in range_dict.keys():
            print('range: ', key)
            print('detection rate: ', range_dict[key]['detection_rate'])
            print('ommision rate: ', range_dict[key]['ommision_rate'])
            print('commission rate: ', range_dict[key]['commission_rate'])
            print('f1 score: ', range_dict[key]['mean_f1'])
            print('gt trees: ', range_dict[key]['gt_trees'])
            print('predicted trees: ', range_dict[key]['predicted_trees'])
            print('predicted trees ok: ', range_dict[key]['predicted_trees_ok'])
            # print if there are any trees that are not detected
            print('')

        print('range_dict: ', range_dict.keys())

        for key in range_dict.keys():
            range_str = key.replace('.', '-')
            if len(range_dict[key]['gt_trees']) > 0 and len(range_dict[key]['predicted_trees']) > 0:
                print('len gt trees: ', len(range_dict[key]['gt_trees']))
                print('len predicted trees: ', len(range_dict[key]['predicted_trees']))
                self.generate_2D_plot_of_trees(
                    df_laz, range_dict[key]['gt_trees'], 
                    range_dict[key]['predicted_trees'],
                    range_dict[key],  # Pass the metrics dictionary
                    name_of_plot=f'2D_plot_{range_str}_{laz_file_path.split("/")[-1].split(".")[0]}',
                    height_range=key  # Pass the height range
                )
        return range_dict


    def generate_2D_plot_of_trees(self, 
                                  df_laz, 
                                  list_of_gt_trees, 
                                  list_of_pred_trees, 
                                  metrics, 
                                  name_of_plot='2D_plot',
                                  height_range=''):
    
        print('list_of_gt_trees in the visualization : ', list_of_gt_trees)
        print('list_of_pred_trees in the visulization: ', list_of_pred_trees)
        
        # Define the folder where plots will be saved
        save_folder = 'plots'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Define colors with less intensity
        color_gt = 'gray'  # Light green for gt trees
        color_pred = 'lightgreen'  # Salmon for pred trees

        plt.figure(figsize=(10, 8), dpi=300)  # Adjust the size and resolution
        plt.style.use('ggplot')  # Use a predefined style for a nicer look
    
        for gt_tree in list_of_gt_trees:
            df_laz_gt_tree = df_laz.loc[df_laz['treeID'] == gt_tree]
            x, y = df_laz_gt_tree['X'].to_numpy(), df_laz_gt_tree['Y'].to_numpy()
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            hull_path = np.append(hull.vertices, hull.vertices[0])  # Ensure closure
            plt.fill(points[hull_path, 0], points[hull_path, 1], 
                 color=color_gt, alpha=0.3, edgecolor='none', 
                 linewidth=2, label='Ground Truth Trees' if gt_tree == list_of_gt_trees[0] else "")
            
        for pred_tree in list_of_pred_trees:
            df_laz_pred_tree = df_laz.loc[df_laz['preds_instance_segmentation'] == pred_tree]
            x, y = df_laz_pred_tree['X'].to_numpy(), df_laz_pred_tree['Y'].to_numpy()
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            hull_path = np.append(hull.vertices, hull.vertices[0])  # Ensure closure
            plt.fill(points[hull_path, 0], points[hull_path, 1], 
                 color=color_pred, alpha=0.3, edgecolor='none', 
                 linewidth=2, label='Predicted Trees' if pred_tree == list_of_pred_trees[0] else "")
            

            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title(name_of_plot)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().set_aspect('equal', adjustable='box')

            # Automatically generate the legend from the labeled data
            plt.legend(loc='upper left')

        metrics_text = (
            f"Mean RMSE: {metrics['mean_rmse']}\n"
            f"Mean F1 Score: {metrics['mean_f1']}\n"
            f"Detection Rate: {metrics['detection_rate']}\n"
            f"Omission Rate: {metrics['ommision_rate']}\n"
            f"Commission Rate: {metrics['commission_rate']}\n"
            f"Num. GT Trees: {metrics['num_gt_trees']}\n"
            f"Num. Predicted Trees: {metrics['num_predicted_trees']}\n"
            f"Num. Trees (IoU > 0.5): {metrics['num_trees_ok_detected']}\n"
        )
        plt.text(0.95, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", alpha=0.5))

        # Additional text for plot name and height range
        plot_info_text = f"Plot: {name_of_plot}\nHeight Range: {height_range}"
        plt.text(0.05, 0.05, plot_info_text, transform=plt.gca().transAxes, fontsize=9, horizontalalignment='left', verticalalignment='bottom', bbox=dict(boxstyle="round", alpha=0.5))

        plt.savefig(os.path.join(save_folder, name_of_plot + '.png'), bbox_inches='tight')  # Save the plot with tight bounding box


    def read_multiple_files(self, folder_name):
        subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
        dataframes = {}

        for subfolder in subfolders:
            subfolder_name = subfolder.split('/')[-1]
            # get list of files in each subfolder in 'metrics_out' folder and exlude 'summary_metrics_all_plots.csv'
            csv_files = [f.path for f in os.scandir(os.path.join(subfolder, 'metrics_out')) if f.is_file() and f.name != 'summary_metrics_all_plots.csv']

            # get list of laz files in each subfolder
            laz_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith('.laz')]

            # laz_files and csv_files contain the same strings, so we can use the same index to get the corresponding laz file
            # index them and create a list of tuples
            laz_and_csv_files = []

            for i in range(len(csv_files)):
                # check if part of csv file name is in laz file name
                for laz_file in laz_files:
                    if csv_files[i].split('/')[-1].split('_')[0] in laz_file:
                        # create tuple with laz file and csv file
                        laz_and_csv_files.append((laz_file, csv_files[i]))

            print(laz_and_csv_files)


            for file in laz_and_csv_files:
                self.read_single_csv_and_parse(file)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vizualize small trees')
    parser.add_argument('-i', '--folder', type=str, required=True, help='Path to folder with csv files')
    args = parser.parse_args()

    viz = VizualizeSmallTrees()
    viz.read_multiple_files(args.folder)