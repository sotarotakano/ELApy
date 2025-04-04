#!/usr/bin/env python
__author__ = 'Sotaro Takano and Kenta Suzuki'
__version__ = '0.0.1'

'''Functions for visualizing energy landscape by surface or contour plots'''

import os
import math
import sys
import copy
from os.path import join
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from ELApy.ELAutility import Bi, cEnergy, CDigitsInteger, CIntegerDigits
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from pygam import GAM, s, te


def _run_MDS(ocmatrix,NMDS=True):
    '''
    Running MDS or NMDS for the occurence matrix.
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.

    NMDS: bool
        IF True, running NMDS.
    
    Returns
    -----------
    X_mds: list
        The computed MDS scores in the all samples. 
        The function always returns 2-dimensions scores.
    '''
    
    if NMDS:
        # Compute pairwise distance matrix
        D = pairwise_distances(ocmatrix)
        # Apply non-metric MDS
        nmds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
        X_nmds = nmds.fit_transform(D)
        return X_nmds
    else:
        mds = MDS(n_components=2, random_state=1)
        X_mds = mds.fit_transform(ocmatrix)
        return X_mds


def _run_PCA(ocmatrix):
    '''
    Running PCA for the occurence matrix.
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.

    Returns
    -----------
    X_mds: feature
        The PC scores for the all samples.
    '''
    pca = PCA()
    pca.fit(ocmatrix)
    PCA(copy=True, n_components=None, whiten=False)
    feature = pca.transform(ocmatrix)
    return feature


def twodimension_summary(ocmatrix,hest,Jest,stbs_new,method="NMDS",ShowPlot=True):
    '''
    Summarizing species occurence data. The community compostiiton array
    in each sample is first prosessed to 2-dimensionalization. Also, to which stable
    states in community compositions are converged is calculated.
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.
    
    hest : numpy.array
        The implicit parameters for species occurence.
    
    Jest : numpy.array
        The species-species interactions parameters.
    
    stbs_new: pandas.DataFrame
        stable states summary after ELpruning.
    
    method : str
        Users can define the type of low-dimensionalization from
        "NMDS"(default), "MDS", or "PCA".
    
    ShowPlot: bool
        If true, the result of low-dimensionalization is visualized
        by 2-dimension plot. 

    Returns
    -----------
    oc_summary: pandas.DataFrame
        The summary of occurence data with 2-dimensionalized scores,
        representing relative similarity between community compositions.
        in each sample. Samples are categorized by their belonging basins.
    '''
    if method == "NMDS":
        print("Running NMDS...")
        dimtypes=["NMDS1","NMDS2"]
        X_tfm =  _run_MDS(ocmatrix,NMDS=True)
    elif method == "MDS":
        print("Running MDS...")
        dimtypes=["MDS1","MDS2"]
        X_tfm =  _run_MDS(ocmatrix,NMDS=False)
    elif method == "PCA":
        print("Running PCA...")
        dimtypes = ["PC1","PC2"]
        X_tfm =  _run_PCA(ocmatrix)

    if ShowPlot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # plot
        ax.scatter(X_tfm[:, 0], X_tfm[:, 1])
        ax.set_title(method)
        plt.xlabel(dimtypes[0])
        plt.ylabel(dimtypes[1])
        plt.show()

    oc_summary = pd.DataFrame(0.0,index=ocmatrix.index,
                            columns=["Energy","sstate","sstate_energy","dim1","dim2"])
    oc_summary["sstate"] = oc_summary["sstate"].astype(int)
    for n,i in enumerate(oc_summary.index):
        sstate_info = Bi(hest,Jest,ocmatrix.loc[i])
        oc_summary.loc[i,"Energy"] = cEnergy(ocmatrix.loc[i],hest,Jest)
        oc_summary.loc[i,"sstate"] = int(sstate_info[-1])
        oc_summary.loc[i,"sstate_energy"] = float(sstate_info[-2])
        oc_summary.loc[i,"dim1"] = X_tfm[n,0]
        oc_summary.loc[i,"dim2"] = X_tfm[n,1] 
    
    # adding the info of belonging stable point for each point as RGB array
    all_sstates = list(set(oc_summary["sstate"]))
    color_dict = {int(x):n for n,x in enumerate(all_sstates)}
    for i in oc_summary.index:
        if not math.isnan(stbs_new.loc[float(oc_summary.loc[i,"sstate"]),"alternative"]):
            oc_summary.loc[i,"colors"] = color_dict[int(stbs_new.loc[float(oc_summary.loc[i,"sstate"]),"alternative"])]
        else:
            oc_summary.loc[i,"colors"] = color_dict[int(oc_summary.loc[i,"sstate"])]

    return oc_summary



def GAM_fitting(oc_summary,fraction=0.25,noise_sd=0.05,n_replicates=16):
    '''
    Connecting points smoothly by GAM function for drawing smooth surface
    or contour plot.

    Parameters
    -----------
    oc_summary : pandas.DataFrame
        The summary of the analyzed samples. This is generated by 
        twodimension_summary function.
    
    fraction : float
        The fraction of points used for the fitting (0 - 1).
    
    noise_sd : float
        The magnitude of noise for GAM fitting.
    
    n_replicates: int
        The number of replicates used for GAM fitting 
        (should be larger than 8).
   
    Returns
    -----------
    pred_Energy : numpy.array
        Predicted energy by GAM fitting.

    mds_mesh: list(numpy.array)
        Coordinates of points in two-dimension surface.
    '''

    oc_replicated = pd.concat([oc_summary] * n_replicates, ignore_index= False).reset_index()
    oc_replicated = oc_replicated.sort_values("index").reset_index().drop(["level_0"],axis=1)
    oc_replicated["replicate_id"] = np.tile(np.arange(n_replicates),int(len(oc_replicated)/n_replicates))

    # add noise
    oc_idxs = {x for x in oc_replicated["index"]}
    for idx in oc_idxs:
        oc_replicated.loc[oc_replicated["index"] == idx, "Energy"] += np.random.normal(loc=0, scale=noise_sd, size=n_replicates)
        oc_replicated.loc[oc_replicated["index"] == idx, "dim1"] += np.random.normal(loc=0, scale=noise_sd, size=n_replicates)
        oc_replicated.loc[oc_replicated["index"] == idx, "dim2"] += np.random.normal(loc=0, scale=noise_sd, size=n_replicates)

    mod = GAM(
        s(0,n_splines=n_replicates) +    
        s(1, n_splines=n_replicates) +    
        te(0, 1,n_splines=n_replicates), distribution='normal', max_iter = 2000, tol=1e-12,
    )

    mod.fit(oc_replicated[["dim1", "dim2"]], oc_replicated['Energy'])

    mds1_seq = np.linspace(oc_replicated["dim1"].min(), oc_replicated["dim1"].max(), int(oc_summary.shape[0]*fraction))
    mds2_seq = np.linspace(oc_replicated["dim2"].min(), oc_replicated["dim2"].max(), int(oc_summary.shape[0]*fraction))

    mds_mesh = np.meshgrid(mds1_seq, mds2_seq)
    pred_Energy = np.zeros_like(mds_mesh[0])
    
    print("Processing " + str(mds1_seq.size) + " points...")
    for i in range(mds1_seq.size):   
        if i%100 == 0:
            print(str(i) + " points finished.")
        for j in range(mds2_seq.size):
            pred_Energy[j, i] = mod.predict(np.array([mds1_seq[i], mds2_seq[j]]).reshape(1,2))

    return oc_summary, pred_Energy, mds_mesh




def draw_contour_plot(oc_summary, pred_Energy, mds_mesh, ocmatrix = [], target_points=[], 
                      neighbor_range=1, max_energy = 10, min_energy = -10, figname = "contour_plot.pdf"):
    '''
    Drawing a contour plot based on the GAM fitting results.

    Parameters
    -----------
    oc_summary : pandas.DataFrame
        The summary of the analyzed samples. This is generated by 
        twodimension_summary function.

    pred_Energy : numpy.array
        Predicted energy by GAM fitting.

    mds_mesh: list(numpy.array)
        Coordinates of points in two-dimension surface.
    
    ocmatrix : pandas.DataFrame
        (optional) If there are points same as or neighbor to 
        the user-targeted community compositions, this function
        can searching those points from ocmatrix and then 
        plot them in 2d surface. 
    
    neighbor_range : int 
        The parameter 
    
    n_replicates: int
        The number of replicates used for GAM fitting 
        (should be larger than 8).
   
    '''
    
    pred_Energy_plot = copy.copy(pred_Energy)
    pred_Energy_plot[pred_Energy > max_energy] = max_energy
    pred_Energy_plot[pred_Energy < min_energy] = min_energy
    
    # plot contour
    plt.contourf(mds_mesh[0], mds_mesh[1], pred_Energy_plot, 20, cmap='bwr')
    plt.colorbar()
    
    # plot scatter: the color of each plot corresponds to the stable point to which each point eventually ends up
    plt.scatter(oc_summary["dim1"], oc_summary["dim2"], c=oc_summary["colors"],
                cmap="viridis",s = 1)

    if len(target_points) > 0:
        cmap = plt.get_cmap('nipy_spectral') 
        num_colors = len(target_points)
        colors = [cmap(i / num_colors) for i in range(num_colors)]
        for n,tp in enumerate(target_points):
            tp_label = np.array(CIntegerDigits(int(tp),ocmatrix.shape[1]))
            stable_point = []
            for i in ocmatrix.index:
                if sum(abs(np.array(ocmatrix.loc[i]) - tp_label)) <= neighbor_range:
                    stable_point.append(i)

            ss_df= oc_summary.loc[stable_point]
            plt.scatter(ss_df["dim1"], ss_df["dim2"], c = colors[n], s = 20)
        
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.title('Energy landscape')
    plt.savefig(figname)

    plt.show()



def draw_surface_plot(oc_summary, pred_Energy, mds_mesh, elev = 50, azim = -150, 
                      ocmatrix = [], target_points=[], neighbor_range=1, 
                      max_energy = 10, min_energy = -10, figname = "surface_plot.pdf"):
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d", facecolor="w")
    ax.view_init(elev, azim)

    theCM = mpl.colormaps.get_cmap('coolwarm')
    theCM._init()
    alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
    theCM._lut[:-3,-1] = alphas
    
    # plot scatter: the color of each plot corresponds to the stable point to which each point eventually ends up
    ax.scatter(oc_summary["dim1"], oc_summary["dim2"], oc_summary["Energy"], s = 1,
               c=oc_summary["colors"], zorder = 2)

    if len(target_points) > 0:
        cmap = plt.get_cmap('nipy_spectral') 
        num_colors = len(target_points)
        colors = [cmap(i / num_colors) for i in range(num_colors)]
        for n,tp in enumerate(target_points):
            tp_label = np.array(CIntegerDigits(int(tp),ocmatrix.shape[1]))
            stable_point = []
            for i in ocmatrix.index:
                if sum(abs(np.array(ocmatrix.loc[i]) - tp_label)) <= neighbor_range:
                    stable_point.append(i)

            ss_df= oc_summary.loc[stable_point]
            ax.scatter(ss_df["dim1"], ss_df["dim2"], ss_df["Energy"], c = colors[n], s = 5)

    # plot surface
    pred_Energy_plot = copy.copy(pred_Energy)
    pred_Energy_plot[pred_Energy > max_energy] = max_energy
    pred_Energy_plot[pred_Energy < min_energy] = min_energy
    ax.plot_surface(mds_mesh[0], mds_mesh[1], pred_Energy_plot, 
                    edgecolor="black",linewidth=0.1,cmap=theCM,alpha=0.4,zorder=1)
    plt.savefig(figname)
    plt.show()