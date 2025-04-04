import os
import math
import sys
from os.path import join
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import ast
from sklearn.decomposition import PCA
import itertools as it
from scipy.sparse import *

#from ELAmain import Bi, cEnergy, CDigitsInteger, CIntegerDigits
from ELApy.ELAutility import Bi, cEnergy, CDigitsInteger, CIntegerDigits

## Function for the data visualization

# Function 
def PCplot(ocmatrix, hest, jest, export=False):
  pca = PCA()
  pca.fit(ocmatrix)
  state_PC = \
    pd.concat(
        [ocmatrix.apply(
            lambda x: Bi([x, cEnergy(x, hest, jest)], hest, jest),
            axis='columns').apply(lambda x: str(x[0])).rename('stablestates'),
         pd.DataFrame(
             pca.transform(ocmatrix),
             index=ocmatrix.index,
             columns=['PC' + str(n) for n in range(1, len(ocmatrix.columns) + 1)])
         ], axis='columns', sort=False)
  sns.scatterplot(data=state_PC, x='PC1', y='PC2', hue='stablestates')
  plt.show()
  if export:
    return state_PC
  else:
    return None


# calc_ccc: Subfunction for GraphObj

# ww: float
# ssenergies: list
# tippingenergies: list of lists

def calc_ccc(ww, ssenergies, tippingenergies):
    sta = list(pd.Series(ssenergies)[pd.Series(ssenergies) <= ww].index)
    tip = list(pd.DataFrame(tippingenergies).reset_index().melt(
      id_vars='index', var_name='columns', value_name='energy').query(
          'energy <= @ww').apply(
            lambda x: (x['index'], x['columns']), axis='columns').values)
    rips = [*[(i, i) for i in sta], *tip]
    tempmat = csr_matrix(([[1] * len(rips)][0],
                          ([rip[0] for rip in rips], [rip[1] for rip in rips]))).toarray()
    ccc_pre1 = csgraph.connected_components(tempmat + tempmat.T)[1]
    ccc_pre2 = pd.concat([pd.Series(ccc_pre1), pd.Series(range(len(ccc_pre1)))],
                         axis='columns', sort=False).groupby(0)[1].apply(
                           lambda x: x.tolist()).tolist()
    ccc = [nodes for nodes in ccc_pre2
           if (len(nodes) >= 2) | ((len(nodes) == 1) & (nodes[0] in sta))]
    return ccc


# Assort: Subfunction of GraphObj

# parent: list
# daughter: list

def Assort(parent, daughter):
    orderkey = pd.DataFrame(
      [[[set(x).issubset(_parent) for _parent in parent].index(True),
        -len(x), x[0]] for x in daughter],
        columns=['setloc', 'length', 'firstN']).sort_values(
            by=['setloc', 'length', 'firstN']).index
    return [daughter[i] for i in list(orderkey)]


# calc_cee: Subfunction of GraphObj

def calc_cee(hee):
    cee = []
    for daughter in hee:
        if len(cee) == 0:
            cee.append(daughter)
        else:
            parent = cee[-1]
            cee.append(Assort(parent, daughter))
    return cee


# calc_xpos1: Subfunction for calcxpos

def calc_xpos1(xpos_prev, _rcee):
    if len(_rcee) == len(xpos_prev[0]):
        output = xpos_prev
    else:
        xx = xpos_prev
        yy = _rcee
        pp = pd.DataFrame(yy).reset_index().melt(
            id_vars='index', var_name='columns', value_name='_yy').dropna().set_index(
                ['_yy']).loc[list(it.chain.from_iterable(xx[0])), 'index'].values
        pp2 = pd.concat(
        [pd.Series(pp), pd.Series(xx[1])], axis='columns', sort=False).apply(
            lambda x: (x[0], x[1]), axis='columns')
        pp3 = [[(i, j) for (i, j) in pp2 if i == _pp] for _pp in np.unique(pp)]
        output = ([[_yy[0]] for _yy in yy],
                [sum([j for (i, j) in _pp3]) / len(_pp3) for _pp3 in pp3])
    return output


# calc_xpos: Subfunction for calc_xpos

# rcee: list

# xpositions: list

def calc_xpos(rcee):
    xpositions = []
    for _rcee in rcee:
        if len(xpositions) == 0:
            xpositions.append(([[__rcee[0]] for __rcee in _rcee], 
                               list(range(len(_rcee)))))
        else:
            xpos_prev = xpositions[-1]
            xpositions.append(calc_xpos1(xpos_prev, _rcee))
    return xpositions



# GraphObj: Create the object for disconnectivity graph

def GraphObj(ssidxs,ssenergies,tippingpoints,tippingenergies):
    energyall = [energy for energy in list(
       set(ssenergies) | set(it.chain.from_iterable(tippingenergies)))
       if energy != np.inf]
    rre = pd.concat(
      [pd.concat([pd.Series(ssenergies),pd.Series(ssidxs)],
                 axis='columns', sort=False),
                 pd.concat([pd.Series(list(it.chain.from_iterable(tippingenergies))),
                            pd.Series(list(it.chain.from_iterable(tippingpoints)))],
                            axis='columns', sort=False)], 
                            axis='index', sort=False).rename(columns={0: 'energy', 1: 'point'}).\
                                drop_duplicates(subset=['energy', 'point'])
    eee = pd.Series(energyall, index=energyall).apply(
       lambda x: calc_ccc(x, ssenergies, tippingenergies)).sort_index(ascending=False)
    ese = eee[pd.Series(eee).astype('str').drop_duplicates(keep='last').index]
    eps = ese.tolist()
    ori = eps[0][0]
    hee = [[*_eps0,*[[_eps1] for _eps1 in set(ori) - set(it.chain.from_iterable(_eps0))]] \
        for _eps0 in eps]
    cee = calc_cee(hee)
    rcee = pd.Series(cee).sort_index(ascending=False).tolist()
    xpositions = pd.DataFrame(calc_xpos(rcee))[1].sort_index(ascending=False).tolist()
    nodes2xposi = pd.concat(
       [pd.Series(list(it.chain.from_iterable(cee))),
        pd.Series(list(it.chain.from_iterable(xpositions)))],
        axis='columns').rename(columns={0: 'cee', 1: 'xpositions'}).astype(
            str).drop_duplicates(subset=['cee', 'xpositions'])
    ff_pre = ese.reset_index().rename(
       columns={'index': 'energy'}).set_index('energy').apply(
          lambda x: pd.DataFrame(list(x), index=range(len(x))).T.rename(
            columns={0: 'cee'}).assign(energy=x.name), axis='columns')
    ff = pd.concat([ff_pre.iloc[i].assign(
       cee=ff_pre.iloc[i]['cee'].astype('str')) for i in range(
          len(ff_pre))]).set_index('cee').assign(
            node2xposi=nodes2xposi.set_index('cee')['xpositions'])
    graphinfo_pre = ff.groupby(['cee']).apply(
       lambda x: x[['energy', 'node2xposi']].sort_values(
          by='energy', ascending=True).iloc[0, :].astype('float')).reset_index()
    graphinfo_pre2 = graphinfo_pre.assign(cee=graphinfo_pre['cee'].apply(
       lambda x: ast.literal_eval(x)))
    graphinfo = graphinfo_pre2.assign(
       len_cee=graphinfo_pre2['cee'].apply(lambda x: len(x)),
       sum_cee=graphinfo_pre2['cee'].apply(lambda x: sum(x))).sort_values(
          by=['len_cee', 'sum_cee'], ascending=True)
    
    return graphinfo.set_index('energy').assign(
              point=rre.set_index('energy')['point']).\
                reset_index().drop(columns=['len_cee', 'sum_cee']).reindex(
                      columns=['cee', 'energy', 'node2xposi', 'point'])


# DisconnectivityGraph: Drawing Disconnectivity graph

# grobj: pd.DataFrame
# s: int
# DG_sample: str
# scale_adj: tuple
# origin_adj: tuple


def DisconnectivityGraph(grobj, nspecies, DG_sample='DG_sample',filename="DisconnectivityGraph.pdf",
                         scale_adj = (0.55, 0.70), origin_adj = (0.75, 0.90)):
    
    _range = [grobj['energy'].min(), grobj['energy'].max()]
    jun = grobj[grobj['cee'].apply(lambda x: len(x)) == 1].sort_values(
       by='node2xposi', ascending=True)
    jen = grobj[grobj['cee'].apply(lambda x: len(x)) > 1].sort_values(
       by='node2xposi', ascending=True)
    jun["point"] = [float(x) for x in jun["point"]]
    jen["point"] = [float(x) for x in jen["point"]]
  
    # Draw figure
    fig, ax = plt.subplots(figsize=(12, 9))
    # scatter with annot
    hgn0 = jun.plot.scatter(ax=ax, x='node2xposi', y='energy', c='point',
                             colormap='brg', s=100)
    for k, v in jun.iterrows():
       hgn0.annotate('C' + str(int(v.iloc[3])), xy=(v.iloc[2],v.iloc[1]), 
                     xytext=(v.iloc[2] + 0.05, v.iloc[1] + 0.15),size=12, weight='bold')
    hgn1 = jen.plot.scatter(ax=ax, x='node2xposi', y='energy', c='black', s=100)
    
    for k, v in jen.iterrows():
       hgn1.annotate('C' + str(int(v.iloc[3])), xy=(v.iloc[2],v.iloc[1]), 
                     xytext=(v.iloc[2] + 0.05, v.iloc[1] + 0.15),size=12, weight='bold')
    # Pie
    ax_sub = pd.Series(index=grobj.index)
    (xmin, xmax) = (grobj['node2xposi'].min(), grobj['node2xposi'].max())
    (ymin, ymax) = (grobj['energy'].min(), grobj['energy'].max())
    for i in ax_sub.index:
        x = (grobj.loc[i, 'node2xposi'] - xmin + origin_adj[0]) / (xmax - xmin) * scale_adj[0]
        y = (grobj.loc[i, 'energy'] - ymin + origin_adj[1]) / (ymax - ymin) * scale_adj[1]
        ax_sub[i] = fig.add_axes([x, y, 0.05, 0.05])
        pd.Series(1, index=range(nspecies)).plot(
            ax=ax_sub[i], kind='pie', labels=None, label='',
            colors=CIntegerDigits(int(grobj.loc[i, 'point']), nspecies).replace(
               1, 'blue').replace(0, 'white').tolist())
    # Line
    if len(grobj) != 1:
        for i in grobj.index[:-1]:
            aa = grobj.loc[i, :]
            bb = grobj[grobj['cee'].apply(
                lambda x: set(aa['cee']).issubset(set(x)))].drop(
                    index=aa.name).iloc[0, :]
            (x1, y1, x2, y2, x3, y3) = (aa.iloc[2], aa.iloc[1], aa.iloc[2], 
                                        bb.iloc[1], bb.iloc[2], bb.iloc[1])
            ax.plot([x1, x2], [y1, y2], color='black')
            ax.plot([x2, x3], [y2, y3], color='black')
    plt.suptitle(DG_sample, fontsize=18)
    plt.savefig(filename)
    plt.show()
    return None


# PCplot2: 3.6. 

# ocmatrix: pd.DataFrame
# hest: pd.Series
# jest: pd.DataFrame
# ssrep: pd.Series
# export: Boolean

def PCplot2(ocmatrix,hest,Jest,savefig = True,ShowFigure=True,filename="PCplot2.pdf"):

    pca = PCA()
    pca.fit(ocmatrix)
    
    oc_summary_PCA = pd.DataFrame(0.0,index=ocmatrix.index,
                                  columns=["Energy","PC1","PC2","sstate","sstate_energy"])
    oc_summary_PCA["sstate"] = oc_summary_PCA["sstate"].astype(str)
    PCscore = pd.DataFrame(pca.transform(ocmatrix),index=ocmatrix.index,
                           columns=['PC' + str(n+1) for n in range(len(ocmatrix.columns))])
    
    for n,i in enumerate(oc_summary_PCA.index):
        sstate_info = Bi(hest,Jest,ocmatrix.loc[i])
        oc_summary_PCA.loc[i,"index"] = CDigitsInteger(ocmatrix.loc[i])
        oc_summary_PCA.loc[i,"Energy"] = cEnergy(ocmatrix.loc[i],hest,Jest)
        oc_summary_PCA.loc[i,"sstate"] = str(int(sstate_info[-1]))
        oc_summary_PCA.loc[i,"sstate_energy"] = float(sstate_info[-2])
        # assign PCA information
        oc_summary_PCA.loc[i,"PC1"] = PCscore.loc[i,"PC1"]
        oc_summary_PCA.loc[i,"PC2"] = PCscore.loc[i,"PC2"]
    
    if ShowFigure:
        sns.scatterplot(data=oc_summary_PCA, x='PC1', y='PC2', 
                        hue='sstate',legend=True)
    if savefig:
        plt.savefig(filename)

    return oc_summary_PCA



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
  
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


# IntrGraph: Drawing species-species interaction graph

# ssidx: the list of indices of stable states
# jest: species interaction matrix
# specieslabel(optional): the user-defined species labels. The length should be same as jest matrix
# j_thresh: the threshold for visualizing interactions
#  (the interactions more than j_thresh or less than -j_thresh are visualized)


def IntrGraph(ssidxs,jest,specieslabel=[],j_thresh=0.25,savefig=True,filename="IntrGraph.pdf"):
    stbs_digits = pd.DataFrame([CIntegerDigits(int(x),jest.shape[0]) 
                                for x in ssidxs],index=ssidxs)
    fmq = list(stbs_digits.loc[:,stbs_digits.max() > 0].columns.values)
    jest_df = pd.DataFrame(jest)
    if specieslabel:
        jest_df.index = specieslabel
        jest_df.columns.values = specieslabel

    jest_df = jest_df.iloc[fmq,fmq]
    
    plink = []
    qlink = []
    for i in jest_df.index:
        for j in jest_df.columns.values:
            if jest_df.loc[i,j] > j_thresh:
                plink.append([i,j])
            elif jest_df.loc[i,j] < -j_thresh:
                qlink.append([i,j])
    pca = PCA()
    pca.fit(jest_df)
    
    
    pcp = pd.DataFrame(pca.transform(jest_df),index=jest_df.index,
      columns=['PC' + str(n+1) for n in range(jest_df.shape[1])]).iloc[:,:2]
    # Lighter and Thickness map
    lmap = jest_df.reset_index().melt(
      id_vars='index', var_name='column', value_name='value').set_index(
          ['index', 'column']).applymap(
              lambda x: min(max(0, x), 1) if x >= 0 else max(-1, x))
    tmap = jest_df.reset_index().melt(
      id_vars='index', var_name='column', value_name='value').set_index(
          ['index', 'column']).applymap(
              lambda x: min(max(0, x), 1) if x >= 0 else max(-1, x))
    # plot
    sns.set_style('white')
    fig, ax = plt.subplots(figsize=(9, 6))
    for l in plink:
        (x1, y1) = (pcp.loc[l[0], 'PC1'], pcp.loc[l[0], 'PC2'])
        (x2, y2) = (pcp.loc[l[1], 'PC1'], pcp.loc[l[1], 'PC2'])
        ax.plot([x1, x2], [y1, y2],
                color=lighten_color('darkblue', 1 - lmap.loc[(l[0], l[1]), 'value']),
                lw=2.00 * tmap.loc[(l[0], l[1]), 'value'])
    for l in qlink:
        (x1, y1) = (pcp.loc[l[0], 'PC1'], pcp.loc[l[0], 'PC2'])
        (x2, y2) = (pcp.loc[l[1], 'PC1'], pcp.loc[l[1], 'PC2'])
        qax = ax.plot([x1, x2], [y1, y2],
                color=lighten_color('darkred', 1 - abs(lmap.loc[(l[0], l[1]), 'value'])),
                lw=2.00 * abs(tmap.loc[(l[0], l[1]), 'value']))
    
    pcp.plot.scatter(ax=ax, x='PC1', y='PC2', s=100)
    for k, v in pcp.reset_index().iterrows():
        ax.annotate(v[0], xy=(v[1], v[2]), 
                    xytext=(v[1] + 0.05, v[2] + 0.15), size=12, weight='bold')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    if savefig:
        plt.savefig(filename)
    
    plt.show()