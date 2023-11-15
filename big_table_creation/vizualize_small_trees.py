import os
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


import numpy as np

from nibio_inference.las_to_pandas import las_to_pandas
import json

HIGH_RANGES = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

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
        df_laz['Z'] = df_laz['Z'] - df_laz['Z'].min()

        df_csv = pd.read_csv(csv_file_path)

        # remove unaamed columns
        df_csv = df_csv.loc[:, ~df_csv.columns.str.contains('^Unnamed')]


        range_dict = {}

        for key, value in HIGH_RANGES_DICT.items():
            df_csv_in_range = df_csv.loc[(df_csv['high_of_tree_gt'] >= HIGH_RANGES[key]) & (df_csv['high_of_tree_gt'] < HIGH_RANGES[key+1])]
            df_dict_laz_tmp = df_laz.loc[(df_laz['Z'] >= HIGH_RANGES[key]) & (df_laz['Z'] < HIGH_RANGES[key+1])]
            # get treeID of gt trees
            gt_trees =  df_dict_laz_tmp['treeID'].unique()

            # compute mean metrics    
            dict_metrics = self.compute_mean_metrics(gt_trees, df_csv_in_range)

            # add dict_metrics to range_dict
            range_dict[value] = dict_metrics
   
        # print range_dict in a pretty way
        # print(json.dumps(range_dict, indent=4))

        for key in range_dict.keys():
            print('gt trees: ', range_dict[key]['gt_trees'])
            print('predicted trees: ', range_dict[key]['predicted_trees'])
            print('predicted trees ok: ', range_dict[key]['predicted_trees_ok'])
            # print if there are any trees that are not detected
            print('')

        print('range_dict: ', range_dict.keys())

        # generate 2D projection of the point cloud of the gt trees and predicted trees if both gt trees and predicted trees are in the range 0-5
        if len(range_dict['0-5.0']['gt_trees']) > 0 and len(range_dict['0-5.0']['predicted_trees']) > 0:
            # print length of gt trees and predicted trees
            print('len gt trees: ', len(range_dict['0-5.0']['gt_trees']))
            print('len predicted trees: ', len(range_dict['0-5.0']['predicted_trees']))
            self.generate_2D_plot_of_trees(
                df_laz, range_dict['0-5.0']['gt_trees'], 
                range_dict['0-5.0']['predicted_trees'],
                name_of_plot='2D_plot_0-5' + '_' + laz_file_path.split('/')[-1].split('.')[0]
                )
            
        # generate 2D projection of the point cloud of the gt trees and predicted trees if both gt trees and predicted trees are in the range 5-10
        if len(range_dict['5.0-10.0']['gt_trees']) > 0 and len(range_dict['5.0-10.0']['predicted_trees']) > 0:
            # print length of gt trees and predicted trees
            print('len gt trees: ', len(range_dict['5.0-10.0']['gt_trees']))
            print('len predicted trees: ', len(range_dict['5.0-10.0']['predicted_trees']))
            self.generate_2D_plot_of_trees(
                df_laz, range_dict['5.0-10.0']['gt_trees'], 
                range_dict['5.0-10.0']['predicted_trees'],
                name_of_plot='2D_plot_5-10' + '_' + laz_file_path.split('/')[-1].split('.')[0]
                )
        
        return range_dict


    def generate_2D_plot_of_trees(self, df_laz, list_of_gt_trees, list_of_pred_trees, name_of_plot='2D_plot'):
        # Define the folder where plots will be saved
        save_folder = 'plots'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Define colors with less intensity
        color_gt = 'lightgreen'  # Light green for gt trees
        color_pred = 'salmon'  # Salmon for pred trees

        # filter df_laz to only contain gt trees and pred trees
        df_laz_gt_trees = df_laz.loc[df_laz['treeID'].isin(list_of_gt_trees)]
        df_laz_pred_trees = df_laz.loc[df_laz['preds_instance_segmentation'].isin(list_of_pred_trees)]

        # Generate 2D array of gt trees and pred trees with x and y coordinates
        gt_trees_2D = df_laz_gt_trees[['X', 'Y']].to_numpy()
        pred_trees_2D = df_laz_pred_trees[['X', 'Y']].to_numpy()

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(gt_trees_2D[:, 0], gt_trees_2D[:, 1], c=color_gt, label='gt trees')
        ax.scatter(pred_trees_2D[:, 0], pred_trees_2D[:, 1], c=color_pred, label='pred trees')

        ax.legend()
        plt.title(name_of_plot.replace('_', ' ').title())
        plot_filename = f'{name_of_plot}.png'
        plt.savefig(os.path.join(save_folder, plot_filename))
        plt.close(fig)


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