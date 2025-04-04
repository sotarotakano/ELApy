
'''
The ELA module for running energy landscape anaysis.

All modules basically work with C++ ELA functions.
'''

import os
import copy
import numpy as np
import pandas as pd
import math
import time
from os.path import join
import datetime
from multiprocessing import Pool
import itertools as it
from ELApy.ELAutility import Formatting, reformatting_result, EnvRescale
from ELApy.StochOptmain import Findbp, SimpleSA, fullSA
from ELApy.surfaceplot import twodimension_summary
from ELApy.visualization import GraphObj, PCplot2


__author__ = 'Sotaro Takano and Kenta Suzuki'
__version__ = '0.0.1'
__credits__ = ["Sotaro Takano", "Kenta Suzuki"]
__license__ = "MIT"

__maintainer__ = "Sotaro Takano"
__email__ = "sotaro.takano@riken.jp"
__status__ = "Beta"

os.environ["OPENBLAS_NUM_THREADS"] = "1"

#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"

# other common modules
import warnings; 
#warnings.simplefilter('ignore')

# import C++ modules
from ELApy.build.ELA import *



def _SteepestDescent_multi(args):
    '''
    The interface function for realizing multiprocessing of SteepestDescent from Python. 
    '''
    J = args[0] 
    h = args[1]
    params = np.vstack([J,h])
    result = SteepestDescent_cpp_ind(params)
    return result

def _FindTipps_multi(args):
    '''
    The interface function for realizing multiprocessing of FindTippingpoint from Python. 
    '''
    params = args[0] 
    tmax = args[1]
    result = FindingTippingpoint_cpp_ind(params,tmax)
    return result


class ELA:
    """
    the main object of ELA

    The ELA class stores all parameters, results, and graphobjects, 
    which are used for running ELA analysis and post-analyses.

    The parameters are predicted in ELApy.params class.

    "savedir" should be specified becaues all the analyzed data is
    stored in this directory.
    
    Parameters
    ----------
    abdata : pandas.DataFrame
        community composition matrix 
        index :  samples or envrionments
        column:  species (taxons)
        
        
    envdata : pandas.DataFrame
        Explicit environmental factors for species
        index:  Types of environmental factor (pH, temparature, 
                the age of hosts etc...)
        column: species (taxon)
    
    Rescale_env : bool
        If True, the environmental matix is normalized by min and max values.
   
    savedir : str, optional
        a directory path where all the analzed results will be saved.
    
    threads: int
        The used logical CPU cores for the computation.
    
    serials : int (default 16)
        Total runs in the paramter fitting process
        (the larger numbers will give more precise predicted parameters
         but takes longer time)
    
        
    Attributes
    ----------
    ocmatrix : pandas.DataFrame
        Species occurence information (0/1) across environments based on a given threshold. 
    
    abmatrix : pandas.DataFrame
        Species abundance matrix (not converted to the binary(0/1)).
    
    envmatrix: pandas.DataFrame
        The values of environmental factors across environments.
    
    refenv: pandas.Series
        The parameters in a reference environment
    
    eid: str (e.g., an column value of envmatrix)
        The name of target environmental factor
    
    params_range: (iterable) list, tuble, np.array, etc...
        The values of the targeted environmental factor.
        The given data set is used as a parameter range for ELA.

    h : numpy.array
        unobserved environmental fator

    J : numpy.array
        interspecies interaction matrix
    
    g : numpy.array
            explicit environmental factor

    stable_states : pandas.DataFrame
        the list of basins identified by Steepest Descent method, 
        accompanied by the relevant attribute parameters.
            
    tipping_points : pandas.DataFrame
        the list of tipping points in the all combinations of possible
        basins. identified by Steepest Descent method, 
        accompanied by the relevant attribute parameters.
    
    stable_states_pruned : pandas.DataFrame
        The summary of the identified stable states after pruning.
        If a focal basin is regarded as "transchangable" to the other 
        deeper basin, then the information of the alternative stable
        state is appended.

    tipping_points_pruned : pandas.DataFrame
        The summary of the tipping points. The tipping points information
        between unstale basins pairs are removed.    

    gradELA_summary : pandas.DataFrame (with explicit environmental parameters)
        The summary of the identified basins when the selected environmental factor
        is chnaged in the user-defined range. All possible basins' information are
        summarized with their energy and alternative states (i.e., if it exists, that
        basin is not a stable one). 
    
    threads: int
        The used logical CPU cores for the computation.
    

    Examples
    --------

    >>> # Here is the simplest example (autorun mode).
    >>> # From rawdata (the case only species abundance data exists),
    >>> # a following command generates an energy landscape and 
    >>> # returns stable states and tipping points information.
    >>> ela = ELA.ELA(abdata,autorun=True)  
    >>> twodim_summary = ela.get_2d_summary()
    >>> twodim_summary, pred_Energy, mds_mesh = GAM_fitting(twodim_summary,fraction=1)
    >>> draw_contour_plot(twodim_summary, pred_Energy, mds_mesh, max_energy = 3, min_energy = -20)
    >>> # With explicit environmental parameters


    Reference
    -----------
    Suzuki, K., Nakaoka, S., Fukuda, S., & Masuya, H. (2021).
    "Energy landscape analysis elucidates the multistability of ecological communities 
    across environmental gradients." Ecological Monographs.
    '''

    """

    # Functions
    def __init__(self, abdata = [], envdata=[], threads = 1, serials = 16, 
                 savedir : str ='', autorun = False, eid : str = ""):

        # define instance variables
        if len(savedir) > 0:
            self.savedir = savedir
        else:
            self.savedir = os.getcwd()
            # self.savedir = os.getcwd() + '/' + datetime.date.today().strftime("%Y-%m-%d")
        
        # set other core parameters and datasets
        if len(abdata) < 1:
            warnings.warn("No species abundance data is specified," + \
                          "You can set manually by ela.abdata = abmatrix(pd.DataFrame)")
        self.abdata = abdata
        
        if len(envdata) < 1:
            warnings.warn("No environmental factor data is specified, " + \
                           "the program will run normalSA, otherwise " + \
                           "you can set manually by ela.envdata = envmatrix <- (pandas.DataFrame)")
        
        self.envdata = envdata
        self.serials = serials
        self.threads = threads

        if autorun:
            if len(envdata) < 1:
                self.Formatting_rawdata(SortSpecies=False) # formatting rawdata
                _ = self.find_bestparams()
                self.run_simpleSA(bestparams=True)

                # Then running ELA using the fitted parameter sets
                self.normalELA()
                # Pruning stable states
                self.ELPruning(pss = 0.1)
            else:
                self.Formatting_rawdata(SortSpecies=False,Rescale_env=True) # formatting rawdata
                self.refenv = self.envmatrix.median()

                _ = self.find_bestparams()
                self.run_fullSA(bestparams=True)
                
                if len(eid) < 1:
                    print("No eid(target environment) is specified. If you want to specify the target eid, " + \
                          "please set it by 'eid = '. ")
                    print("Automatically select the first column as eid here...")
                    self.eid = self.envmatrix.columns.values[0]
                else:
                    self.eid = eid

                # Then running ELA using the fitted parameter sets
                self.gradELA()



    def Formatting_rawdata(self, normalizeq = False, parameters = (0.05, 0.05, 0.95),
                           Rescale_env = True, SortSpecies = True, ShowMessage = True):
        
        '''Formatting abundance and environmental factor matrices.
        
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
        
        Rescale_env : bool
            If True, the environmental matix is normalized by min and max values.
        
        SortSpecies : bool
            sorting the species list in ascending order. 
            This is specifically important to prevent that
            converted binrary indices become over system maximum size of int (or long) type. 
        
        Showmessage : bool
            whether summary is shown or not.
            
        '''
        self.ocmatrix, self.abmatrix, self.envmatrix \
            = Formatting(self.abdata, self.envdata, normalizeq=normalizeq,
                         parameters=parameters,
                         SortSpecies=SortSpecies, ShowMessage=ShowMessage)
        if len(self.envmatrix) > 0:
            if Rescale_env:
                self.envmatrix, self.envcoeffs = EnvRescale(self.envmatrix)


    def find_bestparams(self, ssize = 0.5, we = [0.001], 
                        totalitr = 4000, lmd = [0.001,0.1,0.2,0.25,0.4], serials = 16, itv = 100,
                        runadamW = True, Sparse = True, fastfitting=False):
        best_params, allresults = Findbp(self.ocmatrix, self.envmatrix, ssize, we, totalitr, lmd, serials,
                                         itv, self.threads, runadamW, Sparse, fastfitting)
        self.best_params = best_params[0][0]
        return allresults
    
    

    def run_simpleSA(self, we = 0.001, lmd = 0.4,totalitr = 4000, bestparams = True):
        if bestparams:
            if "best_params" not in dir(self):
                raise AttributeError("The best hyper parameters does not exist for this ELA object." + \
                                "Please run ELA.find_bestparams() first.")
            else:
                lmd = self.best_params[0]
                we = self.best_params[1]
                totalitr = self.best_params[2]

        if lmd > 1e-06:
            Sparse = True
        if we > 1e-06:
            runadamW = True 

        self.h, self.J, self.upd = SimpleSA(self.ocmatrix, we, totalitr, lmd, 
                                            self.serials, 100, self.threads, 
                                            runadamW = runadamW, 
                                            Sparse = Sparse, getall=False)


    def run_fullSA(self, we = 0.001, lmd = 0.4, totalitr = 4000, bestparams = True):
        if bestparams:
            if "best_params" not in dir(self):
                raise AttributeError("The best hyper parameters does not exist for this ELA object." + \
                                "Please run ELA.find_bestparams() first.")
            else:
                lmd = self.best_params[0]
                we = self.best_params[1]
                totalitr = self.best_params[2]


        if lmd > 1e-06:
            Sparse = True
        if we > 1e-06:
            runadamW = True 

        if len(self.envmatrix) > 0:
            self.h, self.g, self.J, self.upd = fullSA(self.ocmatrix, self.envmatrix, 
                                                      we, totalitr, lmd, self.serials, 
                                                      100, self.threads, 
                                                      runadamW = runadamW, 
                                                      Sparse = Sparse, getall=False)
        else:
            warnings.warn("No environmental matrix is specified... running normalELA")
            self.h, self.J, self.upd = SimpleSA(self.ocmatrix, we, totalitr, lmd, 
                                            self.serials, 100, self.threads, 
                                            runadamW = runadamW, 
                                            Sparse = Sparse, getall=False)


    def set_envparams(self,refenv,eid,params_range):
        '''This function should be called when running gradELA'''
        self.refenv = refenv
        self.eid = eid
        self.params_range = params_range


    def normalELA(self, n_xinit=10000, tmax=10000, SkipTippingPoint=False, ShowTime=True):
    
        '''
        Main function of energy landscape analysis (ELA) using the extended pairwise
        maximum entropy model.
        === Basin search ===
        1. Computing the energy values in the neighborhood of point of interest.
        (i.e., The presence/absence status in only one selected species changed 
        from the original composition (ystat) and computing the energy.)
        
        2. Then, selecting the point posessing minimal energy from the candidate list.
        If the energy is smaller in the candidate point than the current point,
        the focal candidate point is expected to locate in closer to an basin 
        in the landscape and set as the next searching point. 
        If not, the current point is a basin and the search is terminated.
        
        3. After identifying basins, searching a tipping point, a point with a minimal energy barrier, 
        between each pair of basins. If the energy barrier is below the threshold, more unstable basin
        (with larger energy) will be pruned by the function "ELpruning".
        
        Parameters
        -----------
        n_xinit: int
            the number of initial points for searching basins in the landscape
            
        tmax: float
            the original energy parameter in a given community
        
        SkipTippingPoint: bool
            skip the process of the tipping point analysis and the subsequent pruning
            of stable states
        
        ShowTime: bool
            If True, the execution time in each process is displayed.             
        '''
        
        starttime = time.time()
        # Searching candidates for stable states in the energy landscape
        # defined by a given set of h and J parameters.

        if "h" not in dir(self) or "J" not in dir(self):
            raise AttributeError("h and J is not set for this ELA object. Please run ELA.run_simpleSA() or " + \
                            "ELA.runfullSA() first.")

        Nspecies = self.J.shape[1]   # The number of species
        species_idx = [int(x) for x in np.arange(Nspecies)] # create species indices

        if self.threads == 1:
            stable_states = pd.DataFrame(SSestimate_cpp(np.array(self.h),np.array(self.J),n_xinit))
        else:
            # Empirically, multiprocessing for the SteepestDescent process does not
            # efficiently work. Therefore, the single thread usually shows the best performance.
            # However, with large species numbers and huge iterations, 
            # multi-threads are sometimes favored (judged by our empirical threshold).

            if Nspecies*n_xinit > 500000: 
                #print("Multiprocessing is applied to SteepestDescent for the current trial...")
                ss_array = np.zeros(n_xinit*(Nspecies+2)).reshape(n_xinit,Nspecies+2)

                with Pool(processes=min(8,self.threads)) as pool:
                    args = ([self.J,self.h] for i in range(n_xinit))
                    results = pool.map(_SteepestDescent_multi,args)
                    
                for i,result in enumerate(results):
                    ss_array[i] = result

                ss_unique = np.unique(ss_array[:,-1],return_index=True)[1]
                stable_states = pd.DataFrame(ss_array[ss_unique])
            else:
                #print("Multiprocessing is not optimal for the current dataset...")
                stable_states = pd.DataFrame(SSestimate_cpp(np.array(self.h),np.array(self.J),n_xinit))

        stable_states.columns = species_idx + ["energy","ssid"]
        stable_states["ssid"] = stable_states["ssid"].astype(int)
        stable_states = stable_states.set_index("ssid")
        

        totaltime = time.time() - starttime
        if ShowTime:
            print("[ELA]Basins searching finsihed...")
            print("[ELA]Total time:%f.3"%totaltime)
            starttime = time.time()
        
        
        if SkipTippingPoint:
            return stable_states, []
        else:
            # comb: create a set of possible combinations of 2 stable states
            n_combs = math.comb(len(stable_states),2)
            comb1 = np.zeros(n_combs*Nspecies).reshape(n_combs,Nspecies)
            comb2 = np.zeros(n_combs*Nspecies).reshape(n_combs,Nspecies)
            for i,x in enumerate(it.combinations(stable_states.index,2)):
                comb1[i] = stable_states.loc[x[0],species_idx]
                comb2[i] = stable_states.loc[x[1],species_idx]

            # Running tipping point search by calling the external C++ function.
            if self.threads == 1:
                tipping_points = pd.DataFrame(TPestimate_cpp(comb1,comb2,
                                            np.array(self.h),np.array(self.J), tmax))
            else:
                if n_combs > 6: 
                # In tipping point search process, multiprocessing is strongly encouraged
                # espcially in the case there are more than 7 possible stable states. 
                # We also set the number of optimal threads (empirical), depending on the 
                # number of possible stable states. 
                    #print("Multiprocessing is applied to FindTippingPoint for the current trial...")
                    tipp_array = np.zeros(n_combs*(Nspecies+3)).reshape(n_combs,Nspecies+3)

                    args = []
                    for i in range(len(comb1)):
                        s1 = comb1[i]
                        s2 = comb2[i]
                        args.append([np.vstack([self.J,self.h,s1,s2]),tmax])

                    with Pool(processes=min(int(len(args)/8),self.threads)) as pool:
                        results = pool.map(_FindTipps_multi,args)
                    
                    for i,result in enumerate(results):
                        tipp_array[i] = result
                    tipping_points = pd.DataFrame(tipp_array)
                else:
                    tipping_points = pd.DataFrame(TPestimate_cpp(comb1,comb2,
                                                np.array(self.h),np.array(self.J), tmax))

            tipping_points.columns = species_idx + ["deltaE","ss1","ss2"]
            tipping_points[["ss1","ss2"]] = tipping_points[["ss1","ss2"]].astype(int)

            if ShowTime:
                totaltime = time.time() - starttime
                print("[ELA]Tipping points searching finsihed...")
                print("[ELA]Total time:%f.3"%totaltime)

            self.stable_states = stable_states
            self.tipping_points = tipping_points



    def gradELA(self, n_xinit=10000, tmax=10000, pss = 0.1):

        '''
        Main function of energy landscape analysis (ELA) using the extended pairwise
        maximum entropy model with explicit environmental data.
        
        The core of this function is same as ELA, but environment -> species interactions
        are computed with h and g.
            
        
        Parameters
        -----------
        pss : float
            The coefficient used to determine the transitionable tipping points. 
            This energy is multiplied by this coefficient to determine 
            if a transition between two stable states possibly occur.


        Common Parameters to ELA
        -----------
        n_xinit: int
            the number of initial points for searching basins in the landscape
            
        tmax: float
            the original energy parameter in a given community
        
        '''

        if "g" not in dir(self):
            raise AttributeError("species-environment interaction matrix 'g' does not exist for this ELA object." + \
                            "Please run ELA.run_fullSA() first.")

        g_np = np.array(self.g)
        h_np = np.array(self.h)

        for p in self.params_range:
            self.refenv.loc[self.eid] = p
            e_params = np.array(self.refenv)
            
            self.h = e_params @ g_np.T + h_np
            self.normalELA(n_xinit=n_xinit, tmax=tmax, SkipTippingPoint=False, ShowTime=False)
            self.ELPruning(pss)
            self.stable_states_pruned[self.refenv.index] = e_params

            if "stbs_summary" in locals():
                stbs_summary = pd.concat([stbs_summary,self.stable_states_pruned],axis=0)
            else:
                stbs_summary = self.stable_states_pruned

            #print("GradELA generated {} energy landscapes for {} at reference environment:\n{}".format(str(p), eid, str(refenv)))
        stbs_summary = stbs_summary.reset_index()
        self.gradELA_summary = stbs_summary
        self.h = h_np



    def ELPruning(self, pss=0.1):
        '''
        Function for pruning the identified stable states by comparing the 
        tipping points in each pair of stable states. If the energy in the 
        lowest tipping point is less than the user-defined threshokd (pss),
        the shallower stable states in a focal pair was removed (regarded as
        unstable steady state), then the deeped stable state will remain.
        Before running this fucntion, stable states and tipping points should 
        be identified by running normalELA.

        Parameters
        -----------
        pss: float
            The threshold for judging tipping points. In each pairs of stable states, 
            if the energy difference between deeper stable states and tipping points are
            less than pss*pmax(maximum energy barrier in the landscape), shallower tipping
            point will be removed and only deeper stable state remains.
        '''

        if "stable_states" not in dir(self):
            raise AttributeError("stable states summary does not exist ..." +  \
                            "Please run ELA.normalELA() first.")


        if len(self.stable_states) < 2:
            warnings.warn("The length of stable states matrix is less than 2. Pruning is skipped...")
            return
        
        stbs_exist = set(self.stable_states.index)
        stbs_new = copy.copy(self.stable_states)
        stbs_new["alternative"] = np.nan
        
        # Formatting tipping points dataframe
        tipps_new = copy.copy(self.tipping_points)
        tipps_new[["deltaST1","deltaST2"]] = np.nan
        for i in self.tipping_points.index:
            ss1 = self.tipping_points.loc[i,"ss1"]
            ss2 = self.tipping_points.loc[i,"ss2"]
            deltaE = self.tipping_points.loc[i,"deltaE"]
            tipps_new.loc[i,"deltaST1"] = deltaE - self.stable_states.loc[ss1,"energy"]
            tipps_new.loc[i,"deltaST2"] = deltaE - self.stable_states.loc[ss2,"energy"]
            tipps_new.loc[i,["deeper","shallower"]] = self.stable_states.loc[[ss1,ss2]].sort_values("energy").index
            tipps_new.loc[i,"mindeltaST"] = min(tipps_new.loc[i,"deltaST1"], tipps_new.loc[i,"deltaST2"])
        
        # maximum energy differences between basin and the corresponding tipping point
        paxmax = max(tipps_new[["deltaST1","deltaST2"]].max(axis=1))
        
        # Eliminating (and merging) unstable basins 
        # (i.e., the energy barrier is below the threshold)
        while tipps_new['mindeltaST'].min() < pss * paxmax:
            paxmax = max(tipps_new[["deltaST1","deltaST2"]].max(axis=1))
            if tipps_new['mindeltaST'].min() >= pss * paxmax:
                break
            else:
                unstable = tipps_new.sort_values(by='mindeltaST', ascending=True).iloc[:1]
                us_id = int(unstable.iloc[0]["shallower"])
                
                # delete the unstable basin
                stbs_new.loc[us_id,"alternative"] = int(unstable.iloc[0]["deeper"])
                stbs_exist -= {us_id}
                
                # delete tipping point data of the unstable basin
                tipps_new = tipps_new.loc[(tipps_new["ss1"] != us_id)&(tipps_new["ss2"] != us_id)]
            
            # If only one stable state remains, stop the pruning 
            if len(stbs_exist) == 1: 
                break
        
        tipps_new[["ss1","ss2"]] = tipps_new[["ss1","ss2"]].astype(int)
        self.stable_states_pruned = stbs_new
        self.tipping_points_pruned = tipps_new
    
    # Visualization
    def get_2d_summary(self,method="NMDS",ShowPlot=True):
        if "stable_states_pruned" not in dir(self):
            raise AttributeError("pruned stable states summary does not exist ..." + \
                            "Please run ELA.ELpruning() first.")
        _2d_summary = twodimension_summary(self.ocmatrix,self.h,self.J,
                                               self.stable_states_pruned,method="NMDS",ShowPlot=True)
        return _2d_summary 
    

    def get_GraphObj(self):
        if "stable_states_pruned" not in dir(self):
            raise AttributeError("pruned stable states summary does not exist ..." + \
                            "Please run ELA.ELpruning() first.")
        ssids,ssEs,tipps,tippEs = reformatting_result(self.stable_states_pruned,self.stable_states,
                                                      self.tipping_points,self.ocmatrix,self.h,self.J)
        graphobj = GraphObj(ssids,ssEs,tipps,tippEs)
        return graphobj
    
    def draw_PCplot(self, figname = "PCplot.pdf"):
        PCplot_df = PCplot2(self.ocmatrix,self.h,self.J,savefig = True,ShowFigure=True,
                            filename=join(self.savedir,figname))
        return PCplot_df
    
    def get_gradELA_diagram(self):
        if "gradELA_summary" not in dir(self):
            raise AttributeError("gradient ELA summary does not exist ..." +  \
                            "Please run ELA.gradELA() first.")
        
        stbs_plot_df = copy.copy(self.gradELA_summary)
        stbs_plot_df["ssid"] = [str(int(x)) for x in self.gradELA_summary["ssid"]]
        stbs_plot_df["state"] = ["unstable" for x in range(len(self.gradELA_summary))]
        for i in self.gradELA_summary.index:
            if math.isnan(self.gradELA_summary.loc[i,"alternative"]):
                stbs_plot_df.loc[i,"state"] = "stable"
        stbs_plot_df = stbs_plot_df.drop(["alternative"], axis=1).sort_values("state")

        return stbs_plot_df