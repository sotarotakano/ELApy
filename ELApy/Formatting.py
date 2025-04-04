import sys
import copy
import numpy as np
import pandas as pd
from scipy.sparse import *

# other common modules
import warnings

#warnings.simplefilter('ignore')



def Formatting(
    baseabtable,
    basemetadata=[],
    normalizeq=True,
    parameters=(0.05, 0.05, 0.95),
    SortSpecies = True,
    ShowMessage = True
    ):
    
    '''Formatting the species composition matrix for the ELA.
    
    1. If the target species is present at the higher rate than the thresold,
       that species is regarded as "present (1)" in that environment.
    
    2. Then, computing the prevalence of each species within a given dataset 
       after the binary conversion. The species those which are found in 
       more than "minocth" but less than "maxorth" are left. 
       Finally, prevalent but non-universal species list are selected.
    
    Parameters
    -----------
    basetable : pandas.DataFrame
        community composition matrix 
        index :  samples or envrionments
        column:  species (taxons)
        
        
    basemetadata : pandas.DataFrame
        Explicit environmental factors for species
        index:  Types of environmental factor (pH, temparature, 
                the age of hosts etc...)
        column: species (taxon)
        
    parameters : tuple or list (length 3)
        [0] (ath): The threshold for species ratio for converting to the binrary
        [1] (minorth): Lower threshold for the fraction of samples 
                       where the target species was present
        [2] (maxorth): Upper threshold for the fraction of samples 
                       where the target species was present
        
    normalizeseq : bool
        If True, normalizing the data so that the total abundance of species is
        1 in each sample.
    
    SortSpecies : bool
        sorting the species list in ascending order. 
        This is specifically important to prevent that
        converted binrary indices become over system maximum size of int (or long) type. 
    
    Showmessage : bool
        whether summary is shown or not.
    
    Returns
    -----------
    ocmatrix : pandas.DataFrame
        Species occurence information (0/1) across environments based on a given threshold. 
    
    abmatrix : pandas.DataFrame
        Species abundance matrix (not converted to the binary(0/1)).
    
    envmatrix: pandas.DataFrame
        The values of environmental factors across environments.
    
    '''
    
    
    ath = parameters[0] # threshold of species ratio for converting 
    
    # minorth: Lower threshold for the fraction of samples where the target species was present
    # maxorth: Upper threshold for the fraction of samples where the target species was present
    minocth = parameters[1] 
    maxocth = parameters[2] 

    # Selecting the species listed on both the community composition data and environment data
    taxonlabel = baseabtable.columns # species
    absamplelabel = baseabtable.index # samples
    if len(basemetadata) > 0:
        mdsamplelabel = basemetadata.index
        sharedsamplelabel = sorted(list(set(absamplelabel) & set(mdsamplelabel)))
        abnum = baseabtable.loc[sharedsamplelabel, :] 
        fctmatrix = basemetadata.loc[sharedsamplelabel, :]
    else:
        abnum = copy.copy(baseabtable)
        fctmatrix  = []
    # (Optional) Normalizing the data so that the total abundance of species is 1 in each sample 
    if normalizeq:
        abnum = abnum.apply(lambda x: x / x.sum(), axis=1)
        
    # Converting to the binary matrix
    aboc = pd.DataFrame(0,index=abnum.index,columns=abnum.columns.values,dtype=int)
    for i in abnum.index:
        for j in abnum.columns.values:
            if abnum.loc[i,j] >= ath:
                aboc.loc[i,j] = 1

    # Computing the prevalence of each species within a given dataset after the binary conversion
    # Then, picking out those which are found in more than "minocth" but less than maxorth.
    occrit = aboc.mean()[(aboc.mean() > minocth) & (aboc.mean() < maxocth)].index 
    ocmatrix = aboc[occrit] # Picking out the specified species listed above
    
    if ocmatrix.shape[1] < 1:
        warnings.warn("no species satisfied minimum and maximum occurence rate.",UserWarning,stacklevel=1)
        return [],[],[]
    # sorting the species list in ascending order. This is specifically important to prevent that
    # converted binrary indices become over system maximum size of int (or long) type. 
    if SortSpecies:
        ocmatrix = ocmatrix[ocmatrix.sum(axis=0).sort_values().index]
    
    abmatrix = abnum[ocmatrix.columns.values] # Picking out the specified species listed above

    try:
        maxidx = max([int("".join([str(x) for x in ocmatrix.loc[i]]),base=2) for i in ocmatrix.index])
    except:
        print("".join([str(x) for x in ocmatrix.loc[i]]))
    if maxidx > sys.maxsize:
        message = "The processed dataset contains too many species " + \
        "(the binary species occurence vector will not correctly be converted to the index.)"
        warnings.warn(message,UserWarning,stacklevel=1)
    
    samplelabel = list(abnum.index) # sample labels
    taxonlabel = list(occrit) # taxon labels (species labels)
    
    if ShowMessage:
        print('Processed ' + str(len(samplelabel)) + ' samples.')
        print('Relative abundance threshold = ' + str(ath))
        print('Occurrence threshold (lower) = ' + str(minocth))
        print('Occurrence threshold (upper) = ' + str(maxocth))
        print('Selected ' + str(len(taxonlabel)) + ' out of ' + str(len(baseabtable.columns)) + ' species.')

    return ocmatrix, abmatrix, fctmatrix
