import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"


from multiprocessing import Pool
import numpy as np
import pandas as pd
# other common modules
import warnings
import sys

# import C++ modules
from ELApy.build.StochOpt import *


def simpleSA_multi(args):
    '''
    The interface function for realizing multiprocessing of simpleSA from Python. 
    '''
    result = simpleSA_cpp(args[0],args[1],args[2],args[3],args[4],args[5],args[6])
    return result

def fullSA_multi(args):
    '''
    The interface function for realizing multiprocessing of fullSA from Python. 
    '''
    result = fullSA_cpp(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7])
    return result


def test_fitting(ocmatrix,sa_results,serials,itv,envmatrix=[]):
    '''
    The function for computing fitting errors in StochOpt function
    using packed results in StochOpt function.
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.

    sa_results: list
        The list of numpy arrays. Each numpy array contains the estimated parameters
        (h, J, (and g)) in StochOpt function.
    
    serials: int
        The used numbers for serials (the number of runs in the StochOpt function.)

    itv: int
        The used parameter in StochOpt function. This is the sampling interval of the
        fitted parameters in the StochOpt function.

    envmatrix (optional): pandas.DataFrame
        The matrix of species-environment interactions 

    Returns
    -----------
    res_serials: pandas.DataFrame
        The result of computing fitting errors in StochOpt function.
    '''
    nspecies = ocmatrix.shape[1]
    timepoints = [(x+1)*itv for x in range(int(sa_results[0].shape[0]/nspecies))]
    res_serials = pd.DataFrame(0.0,index=range(serials),columns=timepoints)
    
    for i,SAparams in enumerate(sa_results):
        rowidx = 0
        res = []
        while rowidx < SAparams.shape[0]:
            params_t = SAparams[rowidx:rowidx+nspecies]
            if len(envmatrix) > 0:
                h_t = params_t[:,0]
                g_t = params_t[:,1:envmatrix.shape[1]+1]
                J_t =  params_t[:,envmatrix.shape[1]+1:-2]
                res.append(float(validateSA(ocmatrix,h_t,J_t,gest=g_t,env=envmatrix)))
            else:
                h_t = params_t[:,0]
                J_t =  params_t[:,1:-2]
                res.append(float(validateSA(ocmatrix,h_t,J_t)))
            rowidx += nspecies
        res_serials.loc[i,:] = res
    return res_serials


def validateSA(ocmatrix,hest,Jest,gest = [], env=[]):
    '''
    The function for computing fitting errors in StochOpt function.
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.

    hest: numpy.array
        The estimated parameters of implicit envtionmental parameters for species.
    
    Jest: numpy.array
        The estimated parameters of species-species interactions.

    gest: numpy.array
        The estimated parameters of explicit environment-species interactions.

    env (optional): pandas.DataFrame
        The matrix of environment factors values

    Returns
    -----------
    res: float
        The computed error for the fitted parameters.
    '''
    ocmat_np = np.array(ocmatrix)
    env_np = np.array(env)
    if len(gest) == 0:
        pp = np.abs(1 - ocmat_np - 1/(1+np.exp(((-hest[:,np.newaxis]-(ocmat_np@Jest).T).T))))
    else:
        if len(env) > 0:
            pp = np.abs(1- ocmat_np - 1/(1+np.exp(-((env_np@gest.T).T + hest[:,np.newaxis]).T-(ocmat_np@Jest))))
        else:
            warnings.warn("ERROR!! Environmental factor matrix is missing. Please specify.")
            return
    res = -np.mean(np.log(pp[pp > 0]).flatten())
    return res


def Findbp(ocmatrix, envmatrix = [], ssize = 0.5, we = [0.001], 
           totalitr = 4000, lmd = [0.001,0.1,0.2,0.25,0.4], serials = 16, itv = 100,
           threads = 16, runadamW = True, Sparse = True, fastfitting=False):
    '''
    The function for finding best parameter sets for StochOpt function.
    In default, the parameters regarding sparse matrix (lmd), adamW (we), and total
    iterations (totalitr) are the target parameters and this function will search
    the best parameter sets (returing the smallest errors). 
    
    Parameters
    -----------
    ocmatrix : pandas.DataFrame
        The occurence matrix of the given species list in the given sample sets.
        This is the binary matrix (0/1) obatined by Formatting function.

    envmatrix (optional): pandas.DataFrame
        The matrix of environment factors values
    
    ssize: float
        The used size of ocmatrix as training data sets.
    
    we: list
        The hyper parameter for adamW specified by a list.
    
    lmd: list
        The parameter for sparse matirx, should be a list object.

    totalitr: int
        The total (maximum) iterations for the parameter fitting.
    
    serials: int
        The numbers of serials for fitting parameters.

    itv: int
        The sampling intervals of the fitted parameters in StochOpt:
    
    threads: int
        The used logical CPU cores for the computation.
    
    runadamW: bool
        If true, the function uses adamW algorithm for the fitting.
    
    Sparse: bool
        If true, the function makes the matrix soarse. 
    
    fastfitting: bool
        If true, the function will search the paramter sets 
        with minimal iterations that can return less than 
        1.03*min_res errors.
   

    Returns
    -----------
    best_params: lists
        The top 5 best parameter sets (lmd, we, iteration) with res values
    
    all_results: dict
        The fitting error values across serials and interations in each
        we and lmd parameter sets.    
    '''

    min_points = {}
    runfullSA = False
    randidxs = [np.random.randint(ocmatrix.shape[0]) 
                for x in range(int(ocmatrix.shape[0]*ssize))]
    octest = ocmatrix.iloc[randidxs,:]
    if len(envmatrix) > 0:
        envtest = envmatrix.iloc[randidxs,:]
        runfullSA = True

    if not runadamW:
        we = [0]

    if not Sparse:
        lmd = [0]

    allresults = dict()
    
    for l in lmd:
        for w in we:
            print("Try: lambda=%f, we=%f, runadamW=%s, Sparse=%s"%(l,w,str(runadamW),str(Sparse)))
            if runfullSA:
                sa_results = fullSA(octest, envtest, we = w, totalitr = totalitr, lmd = l, 
                                                 serials = serials, itv = itv,threads = threads, 
                                                 runadamW = runadamW, Sparse = Sparse, getall=True)
            else:   
                sa_results = SimpleSA(octest, we = w, totalitr = totalitr, lmd = l, 
                                                   serials = serials, itv = itv,threads = threads, 
                                                   runadamW = runadamW, Sparse = Sparse, getall=True)
            
            current_results = test_fitting(ocmatrix,sa_results,serials,itv,envmatrix=envmatrix)
            if fastfitting:
                # returning parameters with minimum iterations allowing less than +3% errors than the minimum
                mean_results = current_results.mean(axis=0)
                min_point =  mean_results.sort_values().head(1)
                min_err = float(min_point.iloc[0])
                mean_results = mean_results.loc[mean_results < min_err*1.03].head(5)
                for m in mean_results.index:
                    min_points.update({(l,w,int(m)):float(mean_results.loc[m])})
            else:
                min_point = current_results.mean(axis=0).sort_values().head(5)
                for m in min_point.index:
                    min_points.update({(l,w,int(m)):float(min_point.loc[m])})
            allresults.update({(l,w):current_results})
        best_params =  sorted(min_points.items(),key=lambda x:x[1])[:5]
    return best_params, allresults


def SimpleSA(ocmatrix, we = 0.01, totalitr = 1000, lmd = 0.01, serials = 1, itv = 100,
             threads = 1, runadamW = False, Sparse = True, getall=False):
    '''Stochastic approximation of pairwise maximum entropy model
    with only implicit environmetal factor
    
    Parameters
    -----------
    ocmatrix: pandas.DataFrame
        formatted binary community composition data (index = sample, columns = species)
    
    we: list
        The hyper parameter for adamW specified by a list.
    
    totalitr: int
        The total (maximum) iterations for the parameter fitting.
    
    lmd: list
        The parameter for sparse matirx, should be a list object.

    serials: int
        The numbers of serials for fitting parameters.

    itv: int
        The sampling intervals of the fitted parameters in StochOpt:
    
    threads: int
        The used logical CPU cores for the computation.
    
    runadamW: bool
        If true, the function uses adamW algorithm for the fitting.
    
    Sparse: bool
        If true, the function makes the matrix soarse. 
    
    getall: bool
        If true, this function returns all fitted parameters during StochOpt 
        with a given interval.

    
    Returns
    -----------
    h: numpy.array
        Unobserved environemntal factors (array)
    J : numpy array
        Species-species interaction strength paramters (matrix)
    upd: 
        the hisory of parameter update during the iteration
    '''

    ocmatrix = np.array(ocmatrix)
    if serials < 1:
        serials = 1
        message = "Serials should be > 1. Here, automatically set to 1"
        warnings.warn(message,UserWarning,stacklevel=1)
        params = simpleSA_cpp(ocmatrix, we, totalitr, lmd, itv ,runadamW, Sparse)
    elif serials == 1:
        params = simpleSA_cpp(ocmatrix, we, totalitr, lmd, itv ,runadamW, Sparse)
        if getall:
            return params
        params_end = params[-ocmatrix.shape[1]:]
        h = params_end[:,0]
        #J = params[:,1:]
        J =  params_end[:,1:-2]
        upd = params[:,-2:]
    else:
        with Pool(processes=threads) as pool:
            args = ([ocmatrix, we, totalitr, lmd, itv, runadamW, Sparse] for _ in range(serials))
            packed_results = pool.map(simpleSA_multi,args)
        if getall:
            return packed_results
        # calculate average h and J
        packed_results_end = [x[-ocmatrix.shape[1]:] for x in packed_results]
        Js = [x[:,1:-2] for x in packed_results_end]
        hs = [x[:,0] for x in packed_results_end]
        upd = [x[:,-2] for x in packed_results]
        for n in range(len(packed_results_end)):
            if n == 0:
                J = Js[n]
                h = hs[n]
            else:
                J += Js[n]
                h += hs[n]

        J = J/len(Js)
        h = h/len(hs)

    return h, J, upd


def fullSA(ocmatrix, envmatrix, we = 0.01, totalitr = 1000, lmd = 0.01, serials = 1, itv = 100,
             threads = 1, runadamW = False, Sparse = True, getall=False):
    
    '''Stochastic approximation of pairwise maximum entropy model
    with explicit environmetal factor
    
    Parameters
    -----------
    ocmatrix: pandas.DataFrame
        Formatted binary community composition data (index = sample, columns = species)
    
    envmatrix: pandas.DataFrame
        Formatted environmental factor matrix data

    we: list
        The hyper parameter for adamW specified by a list.
    
    totalitr: int
        The total (maximum) iterations for the parameter fitting.
    
    lmd: list
        The parameter for sparse matirx, should be a list object.

    serials: int
        The numbers of serials for fitting parameters.

    itv: int
        The sampling intervals of the fitted parameters in StochOpt:
    
    threads: int
        The used logical CPU cores for the computation.
    
    runadamW: bool
        If true, the function uses adamW algorithm for the fitting.
    
    Sparse: bool
        If true, the function makes the matrix soarse. 

    getall: bool
        If true, this function returns all fitted parameters during StochOpt 
        with a given interval.
    
    Returns
    -----------
    h: numpy.array
        Unobserved environemntal factors (array)
    g: numpy.array
        Explicit environmental facotrs (array)
    J : numpy array
        Species-species interaction strength paramters (matrix)
    upd: 
        the hisory of parameter update during the iteration
    '''
        
    ocmatrix = np.array(ocmatrix)
    envmatrix = np.array(envmatrix)

    if serials < 1:
        serials = 1
        message = "Serials should be > 1. Here, automatically set to 1"
        warnings.warn(message,UserWarning,stacklevel=1)
        params = fullSA_cpp(ocmatrix, envmatrix, we, totalitr, lmd, itv ,runadamW, Sparse)
    elif serials == 1:
        params = fullSA_cpp(ocmatrix, envmatrix, we, totalitr, lmd, itv ,runadamW, Sparse)
        if getall:
            return params
        params_end = params[-ocmatrix.shape[1]:]
        h = params_end[:,0]
        g = params_end[:,1:envmatrix.shape[1]+1]
        J =  params_end[:,envmatrix.shape[1]+1:-2]
        upd = params[:,-2:]
    else:
        with Pool(processes=threads) as pool:
            args = ([ocmatrix, envmatrix, we, totalitr, lmd, itv, runadamW, Sparse] for _ in range(serials))
            packed_results = pool.map(fullSA_multi,args)
        if getall:
            return packed_results
        # calculate average h and J
        packed_results_end = [x[-ocmatrix.shape[1]:] for x in packed_results]
        Js = [x[:,envmatrix.shape[1]+1:-2] for x in packed_results_end]
        gs = [x[:,1:envmatrix.shape[1]+1] for x in packed_results_end]
        hs = [x[:,0] for x in packed_results_end]
        upd = [x[:,-2] for x in packed_results_end]
        for n in range(len(packed_results)):
            if n == 0:
                J = Js[n]
                h = hs[n]
                g = gs[n]
            else:
                J += Js[n]
                h += hs[n]
                g += gs[n]

        J = J/len(Js)
        h = h/len(hs)
        g = g/len(gs)

    return h, g, J, upd