#!/usr/bin/env python
# coding: utf-8

# ##### Load Python Modules

# In[ ]:


# import Python modules

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse


from glob import glob

import os, sys

import scipy.io as sio
import sklearn


from mpl_toolkits.mplot3d import Axes3D

from typing import List, Optional, Tuple, Union



import yaml
plt.ioff()


# ##### Entropy Helper Function

# In[ ]:

from fastcluster import linkage
from scipy.stats import entropy
import warnings


import numpy as np
import warnings

def entropy(Y, probs=False):
    """
    Also known as Shannon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    Calculates the entropy of a dataset Y. If probs is True, Y is treated as a set of probabilities.
    """
    Y = np.array(Y)
    if probs:
        if np.sum(Y) != 1:
            if np.sum(Y) == 0:
                # If the sum of Y is 0 when it's supposed to be probabilities, return None as it's invalid.
                return None
            else:
                # Normalize Y to sum to 1 if not already probabilities summing to 1.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    try:
                        Y = Y / np.sum(Y)
                    except RuntimeWarning as e:
                        print(f'Warning for {Y} and sum {np.sum(Y)}\n{e}')
                        return 0

        # Adjust calculation to handle p=0 case correctly by using np.where to replace 0 * log(0) with 0.
        non_zero_prob = Y[Y > 0]
        entropy_value = np.sum(-1 * non_zero_prob * np.log(non_zero_prob))
        return entropy_value
    else:
        _, count = np.unique(Y, return_counts=True, axis=0)
        prob = count / len(Y)
        non_zero_prob = prob[prob > 0]

        # Compute entropy for non-zero probabilities
        entropy_value = np.sum(-1 * non_zero_prob * np.log(non_zero_prob))
        # Adjust the calculation for the non-probability input case.
        return entropy_value

#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    """
    return jEntropy(Y, X) - entropy(X)


# Information Gain
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    Reference: https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Formal_definition
    """
    return entropy(Y) - cEntropy(Y,X)


# In[ ]:


def contigency_table(df, pred_cols, response_col, save=False, save_dir=None, file_type='xlsx', disable_tqdm=True):
    df_ = df.copy(deep=True)
    ct_list, ct_list_prop = [], []
    i = 1
    if isinstance(response_col, list):
        response_col = response_col[0]
    for col in tqdm(pred_cols, disable=disable_tqdm):
        
        curr_ct = pd.crosstab(df_[col], df_[response_col], margins=True, margins_name="Total")
        
        y_categories = curr_ct.columns[:-1]
        
        for category in curr_ct.index: 
            row_counts = curr_ct.loc[category, y_categories].values
            if i <= 2:
                print(f"Pred column: {col}, Category: {category}, Row Count:{row_counts}")
  
            curr_ct.loc[category, "Entropy"] = entropy(row_counts, probs=True)
            i +=1
        curr_ct_prop = curr_ct.copy(deep=True)
        
        grand_total = curr_ct.loc["Total", "Total"]
        curr_ct_prop[y_categories] = curr_ct[y_categories] / grand_total
        curr_ct_prop["Total"] = curr_ct["Total"] / grand_total

        # Assign multi-index
        multi_index = pd.MultiIndex.from_product([[col], curr_ct.index], names=['Variable', None])
        curr_ct.index = multi_index
        curr_ct_prop.index = multi_index
        
        ct_list.append(curr_ct)
        ct_list_prop.append(curr_ct_prop)

    ct = pd.concat(ct_list, axis=0)
    ct_prop = pd.concat(ct_list_prop, axis=0)
    if save:
        if save_dir is None:
            count_file_name = f'contigency_table_{response_col}.{file_type}'
            prop_file_name = f'contigency_table_prop_{response_col}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            count_file_name = os.path.join(save_dir, f'contigency_table_{response_col}.{file_type}')
            prop_file_name = os.path.join(save_dir, f'contigency_table_prop_{response_col}.{file_type}')
        if file_type == 'xlsx':
            ct.to_excel(count_file_name)
            ct_prop.to_excel(prop_file_name)
        elif file_type == 'csv':
            ct.to_csv(count_file_name)
            ct_prop.to_csv(prop_file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")

    return ct, ct_prop


# In[ ]:


def simulate_multinomial(n_list, probs, ct_type):
    """
    Simulate num_samples multinomial samples of size n from a distribution with probabilities p
    """
    if isinstance(n_list, int):
        n_list = [n_list]
    if ct_type == 'null':
        return np.array([np.random.multinomial(n, probs) for n in n_list])
    elif ct_type == 'proper':
        return np.array([np.random.multinomial(n_list[i], probs[i]) for i in range(len(n_list))])
    else:
        raise ValueError("type must be either 'null' or 'proper'")


# In[ ]:


def simulate_contigency_table(ct, ct_type, i=0, save=False, save_dir=None, file_type='xlsx'):
    """
    Simulate a null contigency table by permuting the values in the original contigency table
    """
    ct_ = ct.copy(deep=True)
    ct_ = ct_.droplevel(list(range(ct_.index.nlevels - 1)))
    columns_before_total = ct_.columns.tolist()[:ct_.columns.tolist().index('Total')]
    rows_before_total = ct_.index.tolist()[:ct_.index.tolist().index('Total')]
    n_list = ct_.loc["Total", columns_before_total]
    if ct_type == 'null':
        p_list = ct_.loc[rows_before_total, "Total"]/ct_.loc["Total", "Total"]
    elif ct_type == 'proper':
        p_list = ct_.loc[rows_before_total, columns_before_total].div(ct_.loc['Total', columns_before_total])
        p_list = p_list.T.values
    else:
        raise ValueError("ct_type must be either 'null' or 'proper'")
    
    ct_arr = simulate_multinomial(n_list, p_list, ct_type)

    df = pd.DataFrame(ct_arr.T, index=rows_before_total, columns=columns_before_total)

    df['Total'] = df.sum(axis=1)
    df.loc['Total'] = df.sum(axis=0)
    
    y_categories = df.columns[:-1]
    
    for category in df.index:  
        row_counts = df.loc[category, y_categories].values  
        df.loc[category, "Entropy"] = entropy(row_counts, probs=True)

    ct_ = ct.copy(deep=True)
    ct_.loc[(slice(None),), :] = df.values

    if save:
        if save_dir is None:
            file_name = f'simulate_contigency_table{ct_type}_i_{i}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'simulate_contigency_table{ct_type}_i_{i}.{file_type}')
        if file_type == 'xlsx':
            ct_.to_excel(file_name)
        elif file_type == 'csv':
            ct_.to_csv(file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")
    return(ct_)


# In[ ]:


def ensemble_ct(ct, n_samples, ct_type, disable_tqdm=True, save=True, save_dir=None, file_type='xlsx'):
    entropy_list = []
    rows_before_total = [idx for idx in ct.index if idx[-1] != 'Total']
    samples_to_save = np.random.choice(n_samples, 5, replace=False)

    for i in tqdm(range(n_samples), disable=disable_tqdm):
        if i in samples_to_save:
            three_save = True
        else:
            three_save = False
        sim_ct = simulate_contigency_table(ct, ct_type=ct_type, save=three_save, i=i, save_dir=save_dir, file_type=file_type)
        entropy_list.append(sim_ct.loc[ rows_before_total, "Entropy"])
    
    entropy_df = pd.concat(entropy_list, axis=1)
    entropy_df.columns = list(range(n_samples))
    if save:
        if save_dir is None:
            file_name = f'entropy_simulation_{ct_type}_alpha_{alpha}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'entropy_simulation_{ct_type}.{file_type}')
        if file_type == 'xlsx':
            entropy_df.to_excel(file_name)
        elif file_type == 'csv':
            entropy_df.to_csv(file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")

    return entropy_df


# In[ ]:


def ensemble_ct_CI(df, ct, ct_type, alpha=0.05, save=False, save_dir=None, file_type='xlsx'):
    ct_, df_ = ct.copy(), df.copy()
    lwr_qtl, upr_qtl = alpha/2, 1 - alpha/2
    quantiles = df_.apply(lambda row: [row.quantile(lwr_qtl), row.quantile(upr_qtl)], axis=1, result_type='expand')
    
    quantiles.columns = [f'{lwr_qtl*100}% Quantile', f'{upr_qtl*100}% Quantile']

    rows_before_total = [idx for idx in ct_.index if idx[-1] != 'Total']
    quantiles = ct_.loc[rows_before_total, "Entropy"].to_frame().join(quantiles)
    quantiles['alpha'] = alpha
    # quantiles['alpha_sigf'] = (quantiles.apply(lambda row: row['Entropy'] < row[f'{lwr_qtl*100}% Quantile'] or
    #                                             row['Entropy'] > row[f'{upr_qtl*100}% Quantile'], axis=1))

    probabilities = pd.DataFrame(index=df_.index)

    # Calculate the probabilities for each row
    for idx in df_.index:
        simulated_values = df_.loc[idx, :]  # get the row of simulations
        observed_value = ct_.loc[idx, 'Entropy']
        
        probabilities.loc[idx, 'Pr( <= Entropy)'] = np.mean(simulated_values <= observed_value)
        probabilities.loc[idx, 'Pr( >= Entropy)'] = np.mean(simulated_values >= observed_value)
        pval = np.min([probabilities.loc[idx, 'Pr( <= Entropy)'], probabilities.loc[idx, 'Pr( >= Entropy)']])
        probabilities.loc[idx, 'pval'] = pval
        probabilities.loc[idx, 'alpha_sigf'] = pval < alpha
        probabilities.loc[idx, 'ent(Y|x) < ent(Y)'] = observed_value < ct_.at[(idx[0], 'Total'), 'Entropy']

    quantiles = pd.concat([quantiles, probabilities], axis=1)

    if save:
        if save_dir is None:
            file_name = f'hypothesis_testing_{ct_type}_alpha_{alpha}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'hypothesis_testing_{ct_type}_alpha_{alpha}.{file_type}')
        if file_type == 'xlsx':
            quantiles.to_excel(file_name)
        elif file_type == 'csv':
            quantiles.to_csv(file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")
    return quantiles


# In[ ]:
#-------------------------------------------------------------------
# Finding the optimal threshold for the minimum sum of errors
#-------------------------------------------------------------------

from scipy.stats import gaussian_kde

def find_min_sum_off_overlap_area(sim_null, sim_proper, n_samples):
    concat_np = np.concatenate([sim_null, sim_proper])
    bin_width =  (np.max(concat_np) - np.min(concat_np))/100
    concat_np = np.arange(np.min(concat_np), np.max(concat_np),bin_width)
    sim_null_cut = pd.cut(sim_null, concat_np)
    sim_proper_cut = pd.cut(sim_proper, concat_np)
    sim_null_val_count = sim_null_cut.value_counts()
    sim_proper_val_count = sim_proper_cut.value_counts()
    val_count = pd.concat([sim_null_val_count, sim_proper_val_count], axis = 1)
    val_count_min = val_count.apply(min, axis = 1)
    min_se_val = np.sum(val_count_min)/n_samples

    return min_se_val, min_se_val, concat_np



#-------------------------------------------------------------------
# Plot  minimum sum of errors
#-------------------------------------------------------------------
def plot_distributions_with_error_areas(sim_null, sim_proper, n_samples, idx, bins=20, save_dir=None, file_type='pdf', save=False, show=False):
    sim_null = sim_null[~np.isnan(sim_null)]
    sim_proper = sim_proper[~np.isnan(sim_proper)]

    hist_range = (min(min(sim_null), min(sim_proper)), max(max(sim_null), max(sim_proper)))
    hist_null, bin_edges = np.histogram(sim_null, bins=bins, range=hist_range, density=True)
    hist_proper, _ = np.histogram(sim_proper, bins=bins, range=hist_range, density=True)
    
    # Determine the overlapping area by taking the minimum height of bars at each bin
    overlap_hist = np.minimum(hist_null, hist_proper)
    overlap_area = np.sum(overlap_hist * np.diff(bin_edges))

    # Plotting
    plt.figure(figsize=(8, 6))
    width = np.diff(bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(centers, hist_null, width=width, align='center', color='blue', alpha=0.5, label='Null')
    plt.bar(centers, hist_proper, width=width, align='center', color='red', alpha=0.5, label='Proper')
    
    # Highlighting the overlap area with a different color
    plt.bar(centers, overlap_hist, width=width, align='center', color='green', alpha=0.8, label=f'Overlap Area = {overlap_area:.4f}')
    
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Convert strings to floats and round them to 4 decimal places
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", str(idx[1]))
    rounded_numbers = [round(float(num), 4) for num in numbers]
    
    plt.title(f'Category {rounded_numbers} of {idx[0]} With Overlap Area = {overlap_area:.4f}')
    plt.legend()
    # Save the figure
    if save:
        if save_dir is None:
            file_name = f'Overlapping_area_Category {rounded_numbers}.pdf'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'Overlapping_area_Category {rounded_numbers}.pdf')
        plt.savefig(file_name, format=file_type, bbox_inches='tight')
        # print(f"Plot saved as {file_name}")

    if show:
        plt.show()
    else:
        plt.close()

    return overlap_area



# In[ ]:


def add_min_sum_errors(sim_null, sim_proper, quantiles, n_samples, alpha, save=False, save_dir=None, file_type='xlsx'):
    for idx in sim_null.index:
        print(f"Processing {idx} from {sim_null.index}")
        min_sum_errors = plot_distributions_with_error_areas(sim_null.loc[idx, :], sim_proper.loc[idx, :], n_samples, idx, save_dir=save_dir, file_type='pdf', save=True, show=False)
        # min_sum_errors, optimal_threshold = find_optimal_threshold(sim_null.loc[idx, :], sim_proper.loc[idx, :])
        quantiles.loc[idx, 'min_sum_errors'] = min_sum_errors
        quantiles.loc[idx, 'min_se_sigf'] = min_sum_errors < alpha
    
    if save:
        if save_dir is None:
            file_name = f'null_hypothesis_testing.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'null_hypothesis_testing.{file_type}')
        if file_type == 'xlsx':
            quantiles.to_excel(file_name)
        elif file_type == 'csv':
            quantiles.to_csv(file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")
    
    return quantiles


# In[ ]:


def simulation_hist(sim_null, ct_null,
                    sim_proper, ct_proper, ct, bins=20,
                    file_type='pdf', save=False, save_dir=None, per_fig_height=4,
                    sharex='all', alignment='paired', show=True):  # new parameter for alignment option
    n_rows = sim_null.shape[0]

    # Validate the sharex option
    if sharex not in ['row', 'col', 'all', None, 'none']:  # include None as a valid option
        raise ValueError("Invalid value for sharex. Allowed values are: 'row', 'col', 'all', or None.")
    if sharex is None:
        sharex = 'none'
    # Adjust the subplot creation based on the alignment option
    if alignment == 'paired':
        fig, axs = plt.subplots(n_rows, 2, figsize=(10, n_rows * per_fig_height), sharex=sharex)
        axs = axs.ravel()  # Flatten the axes array
    elif alignment == 'separated':
        fig, axs = plt.subplots(2, n_rows, figsize=(10, 2 * per_fig_height), sharex=sharex)  # 2 rows for 'null' and 'proper', n_rows columns
        axs = axs.ravel()  # Flatten the axes array
    else:
        raise ValueError("Invalid value for alignment. Allowed values are: 'paired', 'separated'.")

    entropies_null = ct_null['Entropy'] 
    entropies_proper = ct_proper['Entropy']

    y_entropy = ct['Entropy'].tolist()[-1]
    for idx, ((index_null, row_null), (index_proper, row_proper)) in enumerate(zip(sim_null.iterrows(), sim_proper.iterrows())):
        if alignment == 'paired':
            ax_null = axs[2 * idx]
            ax_proper = axs[2 * idx + 1]
        else:  # 'separated'
            ax_null = axs[idx]  # top row
            ax_proper = axs[idx + n_rows]  # bottom row

        # 'null' data
        sns.histplot(row_null, bins=bins, kde=False, ax=ax_null, stat='density', legend=False)
        entropy_null = entropies_null.iloc[idx]
        ax_null.axvline(x=entropy_null, color='lime', linestyle='dashed', linewidth=2, label=f'Entropy = {entropy_null:.2f}')
        ax_null.axvline(x=y_entropy, color='orange', linestyle='dashed', linewidth=2, label=f'Y-Entropy = {y_entropy:.2f}')

        ax_null.legend()
        ax_null.set_title(f'{index_null} Null')

        # 'proper' data
        sns.histplot(row_proper, bins=bins, kde=False, ax=ax_proper, stat='density', legend=False)
        entropy_proper = entropies_proper.iloc[idx]
        ax_proper.axvline(x=entropy_proper, color='lime', linestyle='dashed', linewidth=3, label=f'Entropy = {entropy_proper:.2f}')
        ax_proper.axvline(x=y_entropy, color='orange', linestyle='dashed', linewidth=3, label=f'Y-Entropy = {y_entropy:.2f}')
        ax_proper.legend()
        ax_proper.set_title(f'{index_proper} Proper')

    # Make x-tick labels visible for all subplots
    for ax in axs:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)  # Ensure x-ticks are visible

    plt.tight_layout()

    # Save the figure
    if save:
        if save_dir is None:
            file_name = f'simulation_sharex_{sharex}_alignment_{alignment}_hist.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'simulation_sharex_{sharex}_alignment_{alignment}_hist.{file_type}')
        plt.savefig(file_name, format=file_type, bbox_inches='tight')
        # print(f"Plot saved as {file_name}")

    if show:
        plt.show()
    else:
        plt.close()


# In[ ]:



import os
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from fastcluster import linkage_vector, linkage

def binary_bipartite_network_clusterplot(df, response_col, method='ward', metric='euclidean',
                                         show_all_labels=False, return_clusters=False, t=None, ct_type='null',
                                         n_clusters_row=None, n_clusters_col=None, save_dir=None, figsize=(10, 15), 
                                         show_plot=False, save=False, title=None, file_type='pdf', flip=False):
    if df.shape[1] <= 1:
        print(f"Warning: df has {df.shape[1]} columns. Cannot compute bipartite network")
        print(f"Exiting...")
        return None
    if flip:
        df_ = df.drop(response_col, axis=1).T
    else:
        df_ = df.drop(response_col, axis=1)

   
    try:
        categories = df[response_col].drop_duplicates().sort_values(ascending=False).tolist()
    except Exception as e:
        print(f"Warning: response_col {response_col} gave error {e}.")
        assert False, "Exiting..."
        
    lut = dict(zip(categories, sns.color_palette("hls", len(categories))))

    df_ = df_.astype(float)

    
    row_linkage = linkage_vector(df_, method=method, metric=metric)
    col_linkage = linkage_vector(df_.T, method=method, metric=metric)


    # Cluster assignments
        
    try:
        if n_clusters_col:
            print(f"Clustering columns into {n_clusters_col} clusters")
            col_clusters = fcluster(col_linkage, n_clusters_col, criterion='maxclust')
            df['row_clusters'] = col_clusters
    except Exception as e:
        print(f"Col Warning: {e}")

    try:
        if n_clusters_row:
            print(f"Clustering rows into {n_clusters_row} clusters")
            row_clusters = fcluster(row_linkage, n_clusters_row, criterion='maxclust')

            none_list = [None]
            none_list.extend(row_clusters.tolist())
            col_cluster_df = pd.DataFrame({"col_cluster": none_list}).reset_index(drop=True)
            col_cluster_df.index = list(df.columns[:-1])
            col_cluster_df = col_cluster_df.T

            df = pd.concat([df, col_cluster_df], axis=0)

    except Exception as e:
        print(f"Row Warning: {e}")
    
    print(f"Done clustering")
    # Convert the response column to colors
    col_colors = df[response_col].map(lut)

    # Generate color palettes for row and column clusters
    row_colors = pd.Series(row_clusters, index=df_.index).map(lambda x: "C{}".format(x-1))
    
    try:
        g = sns.clustermap(df_, method=method, metric=metric, cmap='flare', figsize=figsize,  row_colors=row_colors,
                       col_colors=col_colors, row_linkage=row_linkage, col_linkage=col_linkage,
                       tree_kws=dict(linewidths=2.3))
        print(f"Done Drawing clustermap")
    except Exception as e:
        print(f"Error: {e}")
        print("Could not draw clustermap")
        return(df_)

    if show_all_labels:
        # Get the default tick size
        default_size = g.ax_heatmap.yaxis.get_major_ticks()[0].tick2line.get_markersize()

        # Set minor ticks for y-axis in the middle of the heatmap rows
        g.ax_heatmap.set_yticks([x + 0.5 for x in range(len(g.data2d.index))], minor=True)
        g.ax_heatmap.set_yticklabels(g.data2d.index, minor=True, va='center')

        # Alter y-ticks: use the default tick size and 2 times that
        for i, tick in enumerate(g.ax_heatmap.yaxis.get_minor_ticks()):
            tick_length = default_size if i % 2 == 0 else 2*default_size
            tick.tick1line.set_markersize(tick_length)
            tick.tick1line.set_visible(True)  # make minor tick visible

        # Hide major y-ticks and their labels
        g.ax_heatmap.yaxis.set_major_formatter(plt.NullFormatter())
        g.ax_heatmap.yaxis.set_major_locator(plt.NullLocator())

        # Adjust x-tick labels rotation
        g.ax_heatmap.set_xticks([x + 0.5 for x in range(len(g.data2d.columns))], minor=True)
        g.ax_heatmap.set_xticklabels(g.data2d.columns, rotation=90, minor=True)
        g.ax_heatmap.xaxis.set_major_formatter(plt.NullFormatter())
        g.ax_heatmap.xaxis.set_major_locator(plt.NullLocator())

    if title is None:
        title = f'Bipartite network with for categories {df.drop(response_col, axis=1).columns.tolist()}'
    g.ax_heatmap.text(0.5, -0.1, title, ha="center", va="center", transform=g.ax_heatmap.transAxes, fontsize=12)

    handles = [mpatches.Patch(color=lut[category], label=category) for category in categories]
    g.ax_heatmap.legend(handles=handles, bbox_to_anchor=(1, 1), loc='upper right', ncol=1, bbox_transform=plt.gcf().transFigure, borderaxespad=0.9)
    
    
    if show_plot:
        plt.show()

    if save:
        if save_dir is None:
            file_name = f'clustered_heatmap_{ct_type}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'clustered_heatmap_{ct_type}.{file_type}')
            

        g.savefig(file_name, dpi=300, bbox_inches='tight', format=file_type)
        print("========================================================================\n")
        print(f"Saved clustermap to {file_name}")
        
    if return_clusters:
        file_name = os.path.join(save_dir, f'selected_bipartite_cluster_ids.csv')
        df.to_csv(file_name)
        return df



# In[ ]:


from matplotlib.lines import Line2D

def plot_combined(sim_null, sim_proper, ct_null, ct_proper, ct, nrows, ncols, file_type='pdf', save=False, save_dir=None, per_fig_height=4, show=True):
    if len(sim_null) != len(sim_proper) or len(ct_null) != len(ct_proper):
        raise ValueError("Input dataframes should have the same number of rows")

    num_plots = len(sim_null)
    num_figures = -(-num_plots // (nrows * ncols))  # Ceiling division to get number of figures needed
    plot_count = 0
    y_entropy = ct.loc[:, 'Entropy'].tolist()[-1]
    for fig_num in range(num_figures):
        fig, axs = plt.subplots(nrows, ncols, figsize=(12 * ncols, per_fig_height * nrows))
        axs = axs.flatten()  # Flattening to easily manage the case of more than one row and column

        for ax_index in range(nrows * ncols):
            if plot_count < num_plots:
                ax = axs[ax_index]

  
                sns.histplot(sim_null.iloc[plot_count, :], kde=False, bins=bins, color="blue", stat="density", alpha=0.5, ax=ax)
                sns.histplot(sim_proper.iloc[plot_count, :], kde=False, bins=bins, color="red", stat="density", alpha=0.5, ax=ax)

                entropy_null = ct_null['Entropy'].iloc[plot_count]
                
                ax.axvline(x=entropy_null, color='lime', linestyle='--', label=f"Entropy {entropy_null:.2f}")
                ax.axvline(x=y_entropy, color='orange', linestyle='--', label=f"Y-Entropy {y_entropy:.2f}")
                ax.set_title(f'Category {sim_proper.index[plot_count][1]} of {sim_proper.index[plot_count][0]}')

                # Create custom legend elements
                custom_lines = [Line2D([0], [0], color="blue", lw=4),
                                Line2D([0], [0], color="red", lw=4),
                                Line2D([0], [0], color="lime", linestyle='--', lw=3),
                                Line2D([0], [0], color="orange", linestyle='--', lw=3)]

                # Generate legend
                ax.legend(custom_lines, ['Null', 'Proper', f"Entropy {entropy_null:.2f}", f"Y-Entropy {y_entropy:.2f}"])
                
                plot_count += 1
            else:
                axs[ax_index].remove()  # Remove the extra axes if the number of plots is less than nrows * ncols

        plt.tight_layout()

        # Save the figure if required
    if save:
        if save_dir is None:
            file_name = f'simulation_hist_{fig_num+1}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'simulation_hist.{file_type}')
        plt.savefig(file_name, format=file_type, bbox_inches='tight')
        # print(f"Plot saved as {file_name}")
    if show:
        plt.show()
    else:
        plt.close()


# In[ ]:


def binary_bipartite_network(ct_ci, ct, df, pred_col, response_col, 
                            ct_type, only_sigf=True, sigf_col='alpha_sigf',
                            save=False, save_dir=None, file_type='xlsx'):
    df_ = df.copy(deep=True)
    if isinstance(pred_col, str):
        pred_col = [pred_col]
    if isinstance(response_col, str):
        response_col = [response_col]

    pred_col_df = df_[pred_col].copy(deep=True)

    pred_col_df = pd.DataFrame(pred_col_df)
    pred_col_df.columns = pred_col
    binary_bipartite = pd.get_dummies(pred_col_df, columns= pred_col, prefix= '', prefix_sep='')

    # List of columns to drop
    if only_sigf:
        if isinstance(sigf_col, str):
            sigf_col = [sigf_col]
        for indx, col in enumerate(sigf_col):
            columns_to_drop1 = ct_ci[ct_ci[col] == True].index.get_level_values(-1).astype(str)
            if indx == 0:
                columns_to_drop = columns_to_drop1
            columns_to_drop = columns_to_drop.intersection(columns_to_drop1)
            print(f"Columns to Hold in {col}: {columns_to_drop1}")
    print(f"Final Columns (Intersection) to Hold: {columns_to_drop}")
    print("===========================================================")
    binary_bipartite = binary_bipartite[columns_to_drop]
       
    binary_bipartite.columns = [f'{pred_col[0]}, {col}' for col in binary_bipartite.columns]
    binary_bipartite = pd.concat([df_[response_col], binary_bipartite], axis=1)

    if save:
        if save_dir is None:
            file_name = f'bipartite_network_{sigf_col}_{ct_type}.{file_type}'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'bipartite_network_{sigf_col}_{ct_type}.{file_type}')
        if file_type == 'xlsx':
            binary_bipartite.to_excel(file_name)
        elif file_type == 'csv':
            binary_bipartite.to_csv(file_name)
        else:
            raise ValueError("file_type must be either 'xlsx' or 'csv'")

    return(binary_bipartite)


# In[ ]:


def make_cat_dataset(df, cols, bins, save=False, save_path=None):
    file_path, ext = os.path.splitext(save_path)
    file_path = file_path + f'_cat{ext}'
    if os.path.exists(file_path):
        print("========================================================================")
        print(f"\nCategorical data set already exists in {file_path}. Skipping...")

        return pd.read_csv(file_path, index_col=0)
    
    df_copy = df.copy(deep=True)
    for col in tqdm(cols):

        if df_copy[col].dtype == 'int' or df_copy[col].dtype == 'float':
            pred_cat = pd.cut(x=df_copy[col], bins=bins, labels=list(range(bins)))
        else:
            pred_cat = df_copy[col]
        df_copy[col] = pred_cat
        df_copy[col] = df_copy[col].astype('category')

    if save:
        if save_path is None:
            raise ValueError("save_dir cannot be None when save is True")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_copy.to_csv(file_path)
        
    return df_copy

import pandas as pd
import numpy as np
from tqdm import tqdm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm  # for notebook environments, use tqdm.notebook.tqdm
import os

def calculate_distribution_similarity(dist1, dist2):
    """
    Calculate similarity between two distributions.
    This could be KL divergence, but here we use a simpler method for demonstration.
    """
    dist1 = dist1 / dist1.sum()
    dist2 = dist2 / dist2.sum()
    # Simple L1 distance (Manhattan distance)
    return np.sum(np.abs(dist1 - dist2))

def make_cat_dataset_with_quantile_check(df, cols, quantiles_to_check, response_col, save=False, save_path=None):
    if save_path:
        file_path, ext = os.path.splitext(save_path)
        file_path += f'_cat_quantile_{quantiles_to_check}{ext}'
        if os.path.exists(file_path):
            print("========================================================================")
            print(f"Categorical dataset already exists at {file_path}. Skipping...")
            return pd.read_csv(file_path, index_col=0)
    
    df_copy = df.copy()
    overall_dist = df_copy[response_col].value_counts(normalize=True)

    for col in tqdm(cols, desc="Processing columns"):
        
        best_distance = []

        if df_copy[col].dtype in ['int', 'float']:
            for q in quantiles_to_check:
                unique_values = df_copy[col].unique()
                quantile_value = np.quantile(unique_values, q)
                print(f"col: {col}, Quantile value: {quantile_value} & length of unique values: {len(unique_values)}, range values: {[np.min(unique_values), np.max(unique_values)]}")

                
   

                temp_bin = pd.qcut(df_copy[col], q=[0, quantile_value, 1], duplicates='drop')
                if len(unique_values) <= 4 or len(temp_bin.unique()) < 2:
                    best_distance.append(np.inf)
                    continue
                
                temp_dist = df_copy.groupby(temp_bin)[response_col].value_counts(normalize=True).unstack(fill_value=0).reset_index(drop=True)
                           # Ensure temp_dist has two rows, one for each category
                # if temp_dist.shape[0] == 1:
                    # If only one category is present, manually add the second category with zero percentages
                    # missing_category = 1 - temp_dist.index[0]  # Assuming categories are 0 and 1
                    # temp_dist.loc[missing_category] = 0
                    # temp_dist = temp_dist.sort_index() 
                print("temp_dist below now")
                print(temp_dist)
                # Calculate distribution similarity for both 'lower' and 'upper' bins, sum them as a simple metric
                lower_dist_similarity = calculate_distribution_similarity(temp_dist.loc[0, :], overall_dist)
                upper_dist_similarity = calculate_distribution_similarity(temp_dist.loc[1, :], overall_dist)
                total_similarity = lower_dist_similarity + upper_dist_similarity

                best_distance.append(total_similarity)

            # Use the best quantile to create the final categorical column
            best_quantile = quantiles_to_check[np.argmin(best_distance)]
            df_copy[col] = pd.qcut(df_copy[col], q=[0, best_quantile, 1], duplicates='drop')

    if save and save_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df_copy.to_csv(file_path)
        
    print(f"Processed {len(cols)} columns. Best quantile selected based on distribution similarity.")

    return df_copy

# In[ ]:


import scipy.cluster.hierarchy as sch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


def rgb2hex(rgb):
    """
    Convert RGB to hexadecimal
    """
    try:
        r, g, b = tuple(rgb)
        return "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
    except:
        print(f'failed rgb {rgb}')

def make_plot_dendogram(df, response_color_cols,  cluster_col, cluster_color_col,
                         link, leaf_label_col, save, save_dir,
                           remove_ticks=True, show_plot=False):
    df_ = df.copy()
    clustered_leaf_labels = df_[cluster_color_col].values
    if isinstance(response_color_cols, str):
        response_color_cols = [response_color_cols]

    if len(response_color_cols) == 3:
        unclustered_leaf_labels = list(map(rgb2hex, df_[response_color_cols].values))
    elif len(response_color_cols) == 1:
        unclustered_leaf_labels = df_[response_color_cols].values.flatten()
    else:
        print(f'\nInvalid response_color_cols: {response_color_cols}. This should either contain 3 columns for rgb colors or 1 column for categorical hex colors\n')
        print(f'Exiting...')
        sys.exit(1)
        
    plt.clf()
    fig = plt.figure(figsize=(25, 10))
    print(f'=========================================================')
    print(f'Head of data')
    print(df_.head(3))
    print(f'=========================================================')
    print(f'response colors columns: {response_color_cols}')
    print(f'cluster column: {cluster_col}')
    print(f'cluster color column: {cluster_color_col}')
    print(f'leaf_label_col: {leaf_label_col}')
    print(f'data shape: {df_.shape}')
    print(f'link shape: {link.shape}')
    print(f'data index: {df_.index}')
    print(f'clustered_leaf_labels: {clustered_leaf_labels[:10]}')
    print(f'unclustered_leaf_labels: {unclustered_leaf_labels[:10]}')
    print(f'=========================================================')

    plt.rcParams.update({"font.size": 14, "axes.labelweight": "bold", "font.weight": "bold"})
    if leaf_label_col is None:
        dn = sch.dendrogram(link, color_threshold=0.5*link[-1,2] )
    else:
        leaf_label = df_[leaf_label_col].values
        dn = sch.dendrogram(link, labels=leaf_label, color_threshold=0.5*link[-1,2] )
    

    plt.title("Dendrogram")
    # plt.xlabel("Leaf Labels")
    plt.ylabel("Distance")
    

    print(f"dendogram leave order: {dn['leaves'][:50]}")
    print(f'=========================================================')

    ax = plt.gca()
    x_points = ax.get_xticks()
    y_points = np.zeros(len(x_points))
    xlbls = ax.get_xmajorticklabels()

    leaf_order = dn['leaves']
    
    color_idx = []
    clustered_colors_list = clustered_leaf_labels[leaf_order]
    unclustered_colors_list = unclustered_leaf_labels[leaf_order]
    
    for idx, lbl in enumerate(xlbls):
        lbl.set_fontsize(18)
        lbl.set_fontweight('bold')
        # color_idx.append(lbl.get_text())
        color_idx.append(idx)

    ax_bbox = ax.get_position()
    # cax1 = fig.add_axes([ax_bbox.x0, -0.14 + -0.065, ax_bbox.width, 0.05])
    cax1 = fig.add_axes([ax_bbox.x0, -0.165, ax_bbox.width, 0.05])
    
    # create custom colorbar
    cm1 = LinearSegmentedColormap.from_list('custom_colormap', clustered_colors_list, N=len(clustered_colors_list))
    sc1 = plt.scatter(x_points, y_points, c=color_idx, cmap=cm1)
    clustered_cb = plt.colorbar(sc1, cax=cax1,  orientation='horizontal')

    # set the size of nex axis [x0,y0,width,height]
    # cax2 = fig.add_axes([ax_bbox.x0, -0.01, ax_bbox.width, 0.05])
    cax2 = fig.add_axes([ax_bbox.x0, -0.14, ax_bbox.width, 0.05])
    cm2 = LinearSegmentedColormap.from_list('custom_colormap', unclustered_colors_list, N=len(unclustered_colors_list))
    sc2 = plt.scatter(x_points, y_points, c=color_idx, cmap=cm2)
    unclustered_cb = plt.colorbar(sc2, cax=cax2,  orientation='horizontal', pad = 0.3)

    df_cluster_unique = df_[[cluster_col, cluster_color_col]].drop_duplicates()
    cluster_colors = df_cluster_unique[cluster_color_col].values
    cluster_label = df_cluster_unique[cluster_col].values
    
    cax3 = fig.add_axes([ax_bbox.width+0.15, ax_bbox.y0, 0.05, ax_bbox.height-0.05])
    cmap = mpl.colors.ListedColormap(cluster_colors)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, df_cluster_unique.shape[0]), cmap.N)
    sc3 = plt.scatter(cluster_label, np.zeros(len(cluster_label)), c = cluster_label, cmap=cmap, norm=norm)
    clustered_vert = plt.colorbar(sc3, cax=cax3,  orientation='vertical', spacing='proportional', ticks=range(len(cluster_label)))
    clustered_vert.ax.set_title('Cluster Id', loc='center', pad=10, fontsize=16, fontweight='bold')
    
    if len(response_color_cols) == 1:
        response_color_cols = response_color_cols[0]
        df_cluster_unique = df_[[response_color_cols, response_color_cols.split(' ')[0]]].drop_duplicates()
        cluster_colors = df_cluster_unique[response_color_cols].values
        cluster_label_ = df_cluster_unique[response_color_cols.split(' ')[0]].values
        cluster_label = np.array([x for x in range(len(cluster_label_))])

        cax4 = fig.add_axes([ax_bbox.width+0.15, -0.01, 0.05, 0.05])
        cmap = mpl.colors.ListedColormap(cluster_colors)
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5, df_cluster_unique.shape[0]), cmap.N)
        sc4 = plt.scatter(cluster_label, np.zeros(len(cluster_label)), c = cluster_label, cmap=cmap, norm=norm)
        unclustered_vert = plt.colorbar(sc4, cax=cax4,  orientation='vertical', ticks=range(len(cluster_label)))
        unclustered_vert.ax.set_title(response_color_cols.split(' ')[0], loc='center', pad=10, fontsize=16, fontweight='bold')
        unclustered_vert.set_ticklabels(cluster_label_) 

    if remove_ticks:
        ax.set_xticklabels([])
    clustered_cb.ax.tick_params(size=0)
    unclustered_cb.ax.tick_params(size=0)
    clustered_cb.ax.set_xticklabels([])
    unclustered_cb.ax.set_xticklabels([])

    if save:
        if save_dir is None:
            file_name = f'dendogram.pdf'
        else:
            os.makedirs(save_dir, exist_ok=True)
            file_name = os.path.join(save_dir, f'dendogram.pdf')
        plt.savefig(file_name, format='pdf', bbox_inches='tight')
        print(f"Plot saved as {file_name}")
    if show_plot:
        plt.show()

import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

def representative_rgb(df, cols, method="mean", weights=None, qtl=None):
    df_ = df.copy()
    df_ = df_[cols]
    if method == "mean":
        return tuple(df_.mean().astype(int))
    
    elif method == "quantile":
        if qtl is None:
            raise ValueError("A quantile value is required for the quantile method.\n\nExample: Pass qtl=0.5 for the median.")
        return tuple(df_.quantile(qtl))
    
    elif method == "mode":
        mode_val = df_.mode().iloc[0]
        return tuple(mode_val.astype(int))
    
    elif method == "dominant":
        kmeans = KMeans(n_clusters=1).fit(df_)
        dominant_color = kmeans.cluster_centers_[0]
        return tuple(dominant_color.astype(int))
    
    elif method == "weighted":
        if weights is None:
            raise ValueError("Weights are required for the weighted average method.")
        weighted_avg = (df_.T * weights).T.sum() / sum(weights)
        return tuple(weighted_avg.astype(int))
    
    elif method == "pca":
        if (df_.var() == 0).any():
            print("Warning: One or more columns have zero variance. PCA will not work with these columns.\n\nReturning the mean instead.")
            return tuple(df_.mean().astype(int))
        pca = PCA(n_components=1)
        transformed_data = pca.fit_transform(df_)
        representative_score = np.mean(transformed_data, axis=0)
        pca_color = pca.inverse_transform(representative_score)
        return tuple(pca_color.astype(int))

    
    elif method == "harmonic":
        harmonic_mean = df_.apply(stats.hmean, axis=0) if not df_.isin([0]).any().any() else df_.mean()
        return tuple(harmonic_mean.astype(int))
    
    else:
        raise ValueError(f"Unknown method: {method}")


import matplotlib.colors as mcolors
def generate_distinct_colors(n):
    """ Generate n visually distinct colors. """
    colors = [mcolors.hsv_to_rgb((i/n, 1, 1)) for i in range(n)]
    colors = [mcolors.to_hex(color) for color in colors]
    return colors

def ColorsSequence3DScatterPerCluster(df, cols, method, weights=None, qtl=None):
    df_ = df.copy()
    clusters = df_['Clusters'].drop_duplicates()
    if method == 'random':
        clustered_colors_list = generate_distinct_colors(clusters.shape[0])
        df_['ClusterColor'] = df_['Clusters'].map(dict(zip(clusters, clustered_colors_list)))
        return df_
    else:
        for ii in range(clusters.shape[0]):
            colors = df_.query('Clusters == {}'.format(ii))
            color_rep = representative_rgb(colors, cols, method=method, weights=weights, qtl=qtl)[:3]
            df_.loc[df_['Clusters'] == ii, 'ClusterColor'] = rgb2hex(color_rep)

        return df_


def mapClusters(df, link, n_clusters = 2, height = None):
    df_ = df.copy()
    if height == None:
        clusters = sch.cut_tree(Z=link, n_clusters=n_clusters)
    else:
        clusters = sch.cut_tree(Z=link, height = height)
    df_['Clusters'] = clusters
    return df_


def cluster_center(df, center_cols, n_cluster,  method, leaf_label_col=None, response_color_cols=None,
                    weights=None, qtl=None, drop_duplicates=False, add_noise=False,
                     save=False, save_dir=None, plot_dendogram = True):
    df_ = df.copy()
    if add_noise:
        noise_scale = 1e-5  
        np.random.seed(0)  
        for col in center_cols:
            noise = np.random.uniform(-noise_scale, noise_scale, df.shape[0])
            df_[col] = df_[col].astype('float') + noise
            
    df_center_unique = df_[center_cols].drop_duplicates()
    print('Clustering on ...')
    print(df_center_unique.head(3)) 
    df_center_unique_ = df_center_unique.copy(deep=True)           
    link = linkage(df_center_unique_, method='ward', metric='euclidean')
    df_center_unique = ColorsSequence3DScatterPerCluster(mapClusters(df_center_unique, link, n_clusters=n_cluster), center_cols,  method=method, weights=weights, qtl=qtl)

    if drop_duplicates:
        df__ = pd.merge(df_, df_center_unique, how='right', on=center_cols).drop_duplicates(subset=center_cols)
        print("========================================================")
        print(f'Dropped {df_.shape[0] - df__.shape[0]} rows. They are')
        print(df_[~df_.index.isin(df__.index)])
    else:
        df__ = pd.merge(df_, df_center_unique, how='right', on=center_cols)
    df_ = pd.merge(df_, df_center_unique, how='left', on=center_cols)
    # print(df__)
    # print(df__.shape)

    if plot_dendogram:
        if response_color_cols is None:
            response_color_cols = center_cols
        if leaf_label_col is None:
            print(f"Removing dendogram ticks since leaf_label_col is None")
            remove_ticks = True
        else:
            print(f"Using {leaf_label_col} as leaf labels since it is not None")
            remove_ticks = False
            df__.index = df__[leaf_label_col]
            
        make_plot_dendogram(df=df__, response_color_cols=response_color_cols, cluster_col='Clusters', cluster_color_col='ClusterColor', 
                            save=save, save_dir=save_dir, link=link, leaf_label_col=leaf_label_col, remove_ticks=remove_ticks)


# In[ ]:


from itertools import combinations
from sklearn.cluster import KMeans

# def combine_cols(df, cols, new_col_name, sep, categorical, bin=None):
    
#     df_ = df.copy()
#     res_df, res_cat_df = pd.DataFrame(), pd.DataFrame()
#     if categorical:
#         res = df_[cols].astype(str).agg(sep.join, axis=1).astype('category')
#         res_cat = res.cat.rename_categories(range(res.nunique()))

#         res_df[new_col_name] = res
#         res_cat_df[new_col_name] = res_cat
#     else:
#         res = df_[cols].values
#         kmeans = KMeans(n_clusters=bin, random_state=0).fit(res)
#         res_df[new_col_name] = list(map(tuple, res))
#         res_cat_df[new_col_name] = kmeans.labels_
            
    
#     return res_df.reset_index(drop=True), res_cat_df.reset_index(drop=True)

# def make_combined_df(df, combine, cols=None, sep=', ', categorical=True, bin=None, save=False, save_path=None, verbose=True):
#     df_ = df.copy()

#     if cols == None:
#         cols = df_.columns.tolist()
#     if verbose:
#         print(f'Combining all possible {combine} columns from {cols[:7]} ...')

    
#     if (categorical==False) and ((bin == None) or (bin < 2)):
#         assert False, "bin must be specified and > 2 if categorical is False"

#     df_list, df_cat_list = [df.drop(cols, axis=1)], [df.drop(cols, axis=1)]

#     cols_comb = list(combinations(cols, combine))
#     col_str_list = []
#     for idx, col in tqdm(enumerate(cols_comb), desc='Combining Columns', total=len(cols_comb)):
#         if combine > 1:
#             col = list(col)
#             col_str = sep.join(col)
#             col_str_list.append(col_str)
#         res, res_cat = combine_cols(df_, col, col_str, sep=sep, categorical=categorical, bin=bin)
#         if idx < 2 and verbose:
#             print(f'Combining {col} ...')
#             print(res.head(3))
#             print(res_cat.head(3))
#         df_list.append(res)
#         df_cat_list.append(res_cat)
    
#     df_combined = pd.concat(df_list, axis=1)
#     df_combined_cat = pd.concat(df_cat_list, axis=1)

#     if save:
#         if save_path is None:
#                 raise ValueError("save_dir cannot be None when save is True")
#         os.makedirs(save_path, exist_ok=True)
#         file_path, ext = os.path.splitext(save_path)
#         df_combined.to_csv(file_path + f'_combined.{ext}')
#         df_combined_cat.to_csv(file_path + f'_combined_cat.{ext}')


#     return df_combined, df_combined_cat, col_str_list


# In[ ]:


#========================================


import shutil, gc
from concurrent.futures import ProcessPoolExecutor, as_completed

def combine_cols(df, cols, new_col_name, sep, categorical, bin=None):
    df_ = df.copy()
    res_df, res_cat_df = pd.DataFrame(), pd.DataFrame()
    if not isinstance(cols, list):
        cols = list(cols)

    if categorical:
        res = df_[cols].astype(str).agg(sep.join, axis=1).astype('category')
        res_cat = res.cat.rename_categories(range(res.nunique()))
        res_df[new_col_name] = res
        res_cat_df[new_col_name] = res_cat
    else:
        res = df_[cols].values
        kmeans = KMeans(n_clusters=bin, random_state=0).fit(res)
        res_df[new_col_name] = list(map(tuple, res))
        res_cat_df[new_col_name] = kmeans.labels_
    return res_df.reset_index(drop=True), res_cat_df.reset_index(drop=True)

def process_column_combination(df, col, sep, categorical, bin):
    col_str = sep.join(list(col)) if len(col) > 1 else col[0]
    res, res_cat = combine_cols(df, col, col_str, sep=sep, categorical=categorical, bin=bin)
    return res, res_cat, col_str

def make_combined_df(df, num_combine, cols=None, sep=', ', categorical=True, bin=None, save=False, save_path=None, verbose=True):
 
    df_ = df.copy().reset_index(drop=True)
    if cols is None:
        cols = df_.columns.tolist()
    if verbose:
        print(f'Combining all possible {num_combine} columns from {cols[:7]} ...')
    if not categorical and (bin is None or bin < 2):
        raise ValueError("bin must be specified and > 2 if categorical is False")

    df_list, df_cat_list, col_str_list = [df.drop(cols, axis=1)], [df.drop(cols, axis=1)], []
    cols_comb = list(combinations(cols, num_combine))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_column_combination, df_, col, sep, categorical, bin) for col in cols_comb]
        for idx, future in tqdm(enumerate(as_completed(futures)), total=len(cols_comb), desc='Combining Columns'):
            res, res_cat, col_str = future.result()
            # if idx < 2 and verbose:
            # print(f'Combining {cols_comb[idx]} ...')
            if (res.shape[0] != df_.shape[0]) or (res_cat.shape[0] != df_.shape[0]):
                
                print(f'\n====================={col_str}======================================')
                print(f"res has shape {res.shape}")
                print(f"res_cat has shape {res_cat.shape}")

            df_list.append(res.reset_index(drop=True))
            df_cat_list.append(res_cat.reset_index(drop=True))
            col_str_list.append(col_str)

    df_combined = pd.concat(df_list, axis=1)
    df_combined_cat = pd.concat(df_cat_list, axis=1)

    if save:
        if save_path is None:
            raise ValueError("save_dir cannot be None when save is True")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file_path, ext = os.path.splitext(save_path)
        df_combined.to_csv(file_path + f'_transformed_{num_combine}_features{ext}')
        df_combined_cat.to_csv(file_path + f'_transformed_cat_{num_combine}_features{ext}')

    return df_combined, df_combined_cat, col_str_list


def process_pred_col(pred_col, df, response_col, n_samples, alpha, sigf_col,
                    save, save_dir, file_type, only_sigf, use_checkpoint):
    # This function processes each predictor column.
    # Add the necessary imports and helper function calls (like contigency_table, ensemble_ct, etc.) here.
    save_dir_ = os.path.join(save_dir, pred_col)
    df_ = df.copy(deep=True)
    ct_type = 'null'
    filepath = glob(f'{save_dir_}/*bipartite_network*', recursive=True)

    if len(filepath) == 0:
        pass
    elif len(filepath) == 1:
        filepath = filepath[0]

        if use_checkpoint:
            
            print(f"Loading {pred_col} via checkpoint")
            return pd.read_excel(filepath, index_col=0, engine='openpyxl').reset_index(drop=True)

    else:
        assert len(filepath) == 0, f"More than one bipartite network file found for {pred_col}"
    
    os.makedirs(save_dir_, exist_ok=True)
    ct_df = contigency_table(df_, pred_cols=[pred_col],
                    response_col=response_col, save=save, save_dir=save_dir_, file_type='xlsx')

    ct_type = 'null'
    
    sim_df_null = ensemble_ct( ct_df[0], n_samples = n_samples, ct_type=ct_type, save=save, save_dir=save_dir_)
    ct_ci_df_null = ensemble_ct_CI(sim_df_null, ct_df[0], alpha=alpha,ct_type=ct_type, file_type=file_type,
                            save=True, save_dir=save_dir_)


    ct_type = 'proper'

    sim_df_proper = ensemble_ct( ct_df[0], n_samples = n_samples, ct_type=ct_type, save=save, save_dir=save_dir_)
    ct_ci_df_proper = ensemble_ct_CI(sim_df_null, ct_df[0], alpha=alpha,ct_type=ct_type, file_type=file_type,
                            save=True, save_dir=save_dir_)
    #####################################################
    #####################################################

    ct_type = 'null'

    ct_ci_df_null = add_min_sum_errors(sim_df_null, sim_df_proper, ct_ci_df_null, alpha=alpha, n_samples = n_samples, save=save, save_dir=save_dir_, file_type='xlsx')
    bb_net_null = binary_bipartite_network(ct_ci_df_null, ct_df, df_, pred_col=pred_col,
                                    response_col=response_col, ct_type=ct_type, only_sigf=only_sigf, sigf_col=sigf_col,
                                        file_type=file_type, save=save, save_dir=save_dir_)              
    try:
        plot_combined(sim_df_null, sim_df_proper, ct_ci_df_null, ct_ci_df_proper, ct_df[0], nrows=3, ncols=2, save=save, save_dir=save_dir_, show=False)
    except Exception as e:
        print(f"Failed to plot combined for {pred_col} due to {e}")
    try:
        simulation_hist(sim_df_null, ct_ci_df_null,
                    sim_df_proper, ct_ci_df_proper, ct_df[0],
                    file_type='pdf', save=save, save_dir=save_dir_, per_fig_height=4,
                    sharex='row', alignment='paired', show=False)
    except Exception as e:
        print(f"Failed to plot simulation_hist for {pred_col} due to {e}")
    
    plt.close('all')
    return bb_net_null



def run_experiment(df, pred_cols, response_col, n_samples, alpha, sigf_col, do_summary=True,
                    save=False, save_dir=None, generate_files_only=False, leaf_label_col=None, 
                    n_clusters_row=2, n_clusters_col=2, use_checkpoint=True,
                    file_type='xlsx', only_sigf=False, show_all_labels=False, max_workers=8):
    
    df_ = df.copy(deep=True)

    print("\n\n")
    print(f'Significance: {sigf_col}')
    print(f'alpha: {alpha}')

    print('===========================================================')
    if (save_dir is None) and (save_dir == '') and (save == False):
        save_dir = 'saved_results'
    if os.path.exists(save_dir) and (use_checkpoint == False):
        shutil.rmtree(save_dir)
    bb_net_null_list = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pred_col, pred_col, df_, response_col, n_samples, alpha, sigf_col,
                                                    save, save_dir, file_type, only_sigf, use_checkpoint
                                                    ): pred_col for pred_col in pred_cols}
        for indx, future in tqdm(enumerate(as_completed(futures)), total=len(pred_cols)):
            bb_net_null = future.result()
            if indx != 0:
                bb_net_null = bb_net_null.drop(columns=response_col)
            bb_net_null_list.append(bb_net_null.reset_index(drop=True))
            gc.collect()
    if generate_files_only:
                    
        print(f"Completed generating files only")
        return None

    bb_net_null_df = pd.concat(bb_net_null_list, axis=1)
    print('========================Bipartite Network===================================')
    print(bb_net_null_df)
    print('===========================================================')

    bb_net_null_df.index = df_[leaf_label_col].values
    print(f"Bipartite network shape: ")
    print(bb_net_null_df.head(3))
    print(f"Bipartite network shape: {bb_net_null_df.shape}")

    bb_net_clustermap = binary_bipartite_network_clusterplot(bb_net_null_df, response_col=response_col, method='ward', metric='euclidean', 
                                                        show_all_labels=show_all_labels, return_clusters=True, t=None, 
                                                        n_clusters_row=n_clusters_row, n_clusters_col=n_clusters_col,
                                                        figsize=(30, 30), save=True, save_dir=save_dir, file_type='pdf', title='', flip=True)
    
    if do_summary:
        gather_null_hyp = sorted(glob(f'{save_dir}/**/*null_hypothesis_testing*', recursive=True))
        null_hyp = pd.concat([pd.read_excel(file, index_col=0, engine='openpyxl') for file in gather_null_hyp], axis=0)
        null_hyp.to_excel(f'{save_dir}/null_hypothesis_testing_summary.xlsx')

        bb_net_null_df.T.to_excel(f'{save_dir}/bipartite_network_summary.xlsx')




from sklearn.feature_selection import VarianceThreshold

def remove_near_constant(data, pred_cols, threshold=0.01):
    df_ = data.copy(deep=True)
    data_ = data.copy(deep=True)
    data_ = data_[pred_cols]
    selector = VarianceThreshold(threshold)
    selector.fit(data_)
    cols_to_keep = data_.columns[selector.get_support()]
    cols_to_drop = [col for col in pred_cols if col not in cols_to_keep]

    return data.drop(columns=cols_to_drop), cols_to_drop



# ##### Read Data

# In[ ]:

def parse_args():
    parser = argparse.ArgumentParser(description='Run significance analysis on pistachio color data')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Config file path')
    parser.add_argument('--save', action='store_true', help='Save results')
    parser.add_argument('--combine', action='store_true', help='Combine columns')
    parser.add_argument('--num_combine', type=int, default=2, help='Number of columns to combine')
    parser.add_argument('--categorical', action='store_true', help='Use categorical data')
    parser.add_argument('--data_filename', type=str, help='Data file path basename')
    parser.add_argument('--save_dir', type=str, default='save_folder', help='Save directory')
    parser.add_argument('--use_checkpoint', action='store_true', help='Use checkpoint')
    args = parser.parse_args()
    args.combine = True if args.num_combine > 1 else False
    return args

# In[ ]:

def analyze(args):

    save = args.save

    combine, num_combine, categorical = args.combine, args.num_combine, args.categorical

    data_filename = args.data_filename


    def load_config(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    config = load_config(args.config)
# ... and so on for other configurations
    
    print(f"\n\n\n===========================================================\n")
    print(f"Running analysis with the following parameters:")
    args_dict = vars(args)

    # Print the comment and the args_dict
    print("\n# Script Args")
    print(yaml.dump(args_dict))

    # Print another comment and the args.__dict__
    print("\n\n# Config.yaml Args")
    print(yaml.dump(config))
    
    print("===========================================================")
    print(f"Setting recursion limit to {config['recursion_limit']}")
    sys.setrecursionlimit(config['recursion_limit'])
# In[ ]:

    print("\n===========================================================\n")

    file_dir = config['file_dir']

    filepath = os.path.join(file_dir, data_filename)

    print(f"Analyzing data in {filepath}")


# In[ ]:


    data_df = pd.read_csv(filepath)
    print("Read in dataframe:")
    print(data_df.head(3))




# In[ ]:

    not_pred_cols = config['not_pred_cols']
    bins = config['bins']
    num_combine = args.num_combine

    quantile_check = config['quantile_check']
    response_col = config['response_col']
    leaf_label_col = config['leaf_label_col']
    response_color_cols = config['response_color_cols']
    weights = config['weights']
    n_cluster = config['n_cluster']
    qtl = config['qtl']
    add_noise = config['add_noise']
    drop_duplicates = config['drop_duplicates']
    plot_dendogram = config['plot_dendogram']
    cluster_center_method = config['cluster_center_method']
    do_clustering = config['do_clustering']   
    quantiles_to_check = config['quantiles_to_check']   
    remove_low_variance = config['remove_low_variance']

    if remove_low_variance:
        pred_cols = sorted(list(set(data_df.columns).difference(set(not_pred_cols))))
        print(f"Some predictor columns: ")
        print(pred_cols[:6])
        print(f"Original number of predictor columns: {len(pred_cols)}")        

        data_df, dropped_cols = remove_near_constant(data_df, pred_cols, threshold=0.001)
        print("===========================================================")
        print(f"Removed near constant columns. New dataframe shape: {data_df.shape}")
        print(f"Removed columns: {dropped_cols}")
        print("===========================================================")
        


    pred_cols = sorted(list(set(data_df.columns).difference(set(not_pred_cols))))
    
    print("===========================================================")
    print(f"Some predictor columns: ")
    print(pred_cols[:6])
    print(f"Number of predictor columns: {len(pred_cols)}")        

    if categorical:
        if quantile_check:
            data_df_cat = make_cat_dataset_with_quantile_check(df=data_df, cols = pred_cols, quantiles_to_check=quantiles_to_check, response_col=response_col, save=save, save_path=filepath)
        else:
            data_df_cat = make_cat_dataset(df=data_df, cols = pred_cols, bins=bins, save=save, save_path=filepath)
        print("\n===========================================================\n")
        print("Categorical dataframe:")
        print(data_df_cat.head(3))



# In[ ]:


    if combine and (num_combine > 1):

        file_path, ext = os.path.splitext(filepath)
        if os.path.exists(file_path + f'_transformed_{num_combine}_features{ext}') and os.path.exists(file_path + f'_transformed_cat_{num_combine}_features{ext}'):
            print("========================================================================")
            print(f"\nLoading {file_path + f'_transformed_{num_combine}_features{ext}'}")
            transformed_data = pd.read_csv(file_path + f'_transformed_{num_combine}_features{ext}', index_col=0)
            transformed_data_recoded = pd.read_csv(file_path + f'_transformed_cat_{num_combine}_features{ext}', index_col=0)
            pred_cols = sorted(list(set(transformed_data_recoded.columns).difference(set(not_pred_cols))))
            print(f"\nLoaded {file_path + f'_transformed_{num_combine}_features{ext}'}")
        else:
            print(f"\nMaking {num_combine} combinations column sets from {len(pred_cols)} with categorical={categorical}")
            data_df_to_combine = data_df_cat if categorical else data_df
            transformed_data, transformed_data_recoded, pred_cols = make_combined_df(data_df_to_combine, num_combine=num_combine, cols=pred_cols, sep=', ', categorical=categorical, bin=3, save=save, save_path=filepath, verbose=True)
        
        print("========================================================================")    
        print(f"Transformed data:")
        print(transformed_data.head(3))
        print("========================================================================")
        print(f"Transformed data recoded:")
        print(transformed_data_recoded.head(3))
        print("========================================================================")
        print(f"Transformed data recoded shape: {transformed_data_recoded.shape}")
        transformed_data_df = transformed_data_recoded
    else:
        
        transformed_data_df = data_df_cat if categorical else data_df
        


    filter_response = config['filter_response']
    valid_response_list = config['valid_response_list']
    response_filter_str = '_'.join(valid_response_list) if (filter_response and (valid_response_list is not None)) else 'all_response_subtypes'

    if filter_response and valid_response_list is not None:
        transformed_data_df = transformed_data_df[transformed_data_df[response_col].isin(valid_response_list)]
    elif filter_response and valid_response_list is None:
        print("\n===========================================================\n")
        raise ValueError("response_filter cannot be None when filter_response is True")
    elif (not filter_response) and (valid_response_list is not None):
        print("\n===========================================================\n")
        raise ValueError("filter_response cannot be False when response_filter is not None")
    else:
        pass
    transformed_data_df = transformed_data_df.reset_index(drop=True)
    print('===========================================================')
    print('Distribution of Response Variable')
    print('===========================================================')
    print(transformed_data_df[response_col].value_counts().to_frame())

# In[ ]:

    if args.save_dir == '':
       args.save_dir = os.path.dirname(filepath) 
    if combine and (num_combine > 1):
        if categorical:
            save_dir = os.path.join(args.save_dir, 'combine_cat', str(num_combine) + f'_features_combinations_using_{bins}_bins' , response_filter_str)
        else:
            save_dir = os.path.join(args.save_dir, 'combine_raw', str(num_combine) + '_features_combinations_using_raw_variables', response_filter_str)
    else:
        if categorical:
            save_dir = os.path.join(args.save_dir, f'1_feature_using_categorical_using_{bins}_bins', response_filter_str)
        else:
            save_dir = os.path.join(args.save_dir, '1_feature_using_raw_variable', response_filter_str)
    
    print("\n===========================================================\n")
    print(f"Setting save_dir to {save_dir}")


# In[ ]:

    if do_clustering:
        transformed_data_df_ = transformed_data_df.copy(deep=True)
        if (not response_color_cols) or (response_color_cols is None) or (response_color_cols == ''):
            print("\n===========================================================\n")
            print(f"\nGenerating response color cols using {response_col}\n")
            response_color_list = generate_distinct_colors(transformed_data_df_[[response_col]].drop_duplicates().shape[0])
            groups = transformed_data_df_[response_col].drop_duplicates().values.tolist()
            response_color_cols = f"{response_col} Groups"
            transformed_data_df_[response_color_cols] = transformed_data_df_[response_col].map(dict(zip(groups, response_color_list )))
            print(transformed_data_df_[[response_color_cols, response_col]].drop_duplicates().head(3))
            # assert False, "STOP HERE"
        cluster_center(df=transformed_data_df_, center_cols=pred_cols, n_cluster=n_cluster,  method=cluster_center_method, 
                leaf_label_col=leaf_label_col, response_color_cols=response_color_cols, weights=weights, qtl=qtl, save=save, save_dir=save_dir,
                add_noise=add_noise, drop_duplicates=drop_duplicates, plot_dendogram = plot_dendogram)
    
        if not categorical:
            print("\n===========================================================\n")
            print('Categorical is set to False. Script stops here')
            sys.exit()

# In[ ]:
    print("\n===========================================================\n")
    print(f"Transformed dataframe:")
    print(transformed_data_df.head(5))
    print(f"Transformed dataframe shape: {transformed_data_df.shape}")


# In[ ]:

    
    use_checkpoint = args.use_checkpoint

    show_all_labels = config['show_all_labels']
    only_sigf = config['only_sigf']
    generate_files_only = config['generate_files_only']
    alpha = config['alpha']
    do_summary = config['do_summary']
    n_clusters_row=config['n_clusters_row']
    n_clusters_col=config['n_clusters_col']

    sigf_col = config['sigf_col']
    sigf_col_str = '_'.join(sigf_col)
    n_samples = config['n_samples']
    file_type= config['spreadsheet_file_type']
    
    save_dir=os.path.join(save_dir, 'analysis', sigf_col_str)
    
    max_workers = config['max_workers']


    run_experiment(df=transformed_data_df, pred_cols=pred_cols, response_col=response_col, n_samples=n_samples,
                    alpha=alpha, sigf_col=sigf_col, do_summary=do_summary, save=save, save_dir=save_dir, leaf_label_col=leaf_label_col,
                    generate_files_only=generate_files_only, use_checkpoint=use_checkpoint, file_type=file_type, n_clusters_row=n_clusters_row, n_clusters_col=n_clusters_col,
                    only_sigf=only_sigf, show_all_labels=show_all_labels, max_workers=max_workers)
    
  
    print("\n===========================================================\n")
    print(f"Analysis complete. Results saved in {save_dir}")
    save_run_config_path = os.path.join(save_dir, 'run_config.yaml')
    with open(save_run_config_path, 'w', encoding='utf-8') as file:
        file.write("# Script Args\n")
        yaml.dump(args_dict, file)
        file.write("\n\n# Config.yaml Args\n")
        yaml.dump(config, file)

from datetime import datetime

def main():
    print("=============================================================\n")
    print(f'Start time is: {datetime.now().strftime("%H:%M:%S")}')
    start_time = datetime.now()

    args = parse_args()

    analyze(args)
    elapsed_time = datetime.now() - start_time
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds += elapsed_time.microseconds / 1e6  # Add microseconds as a decimal to seconds

    print("\n=============================================================\n")
    print("End time is: ", datetime.now().strftime("%H:%M:%S"))
    print()
    print("The script ran for {}hr {}min {:.6f}sec".format(hours, minutes, seconds))
    print("\n=============================================================\n")

if __name__ == '__main__':
    main()