from ELApy import ELA
from os.path import join
import pandas as pd
from ELApy.ELAutility import *
from ELApy.visualization import *
from ELApy.surfaceplot import *

maindir = "testdata"

abdata = pd.read_csv(join(maindir,'abundance_table.csv'),index_col = 0)
metadata = pd.read_csv(join(maindir,'sample_metadata.csv'),index_col = 0)

# Simple example
ela = ELA.ELA(abdata, threads = 1)   # create ELA object and load rawdata
ela.Formatting_rawdata(SortSpecies=False) # formatting rawdata

ela.run_simpleSA(bestparams=False,serials=16)

ela.normalELA()
ela.ELPruning()

twodim_summary = ela.get_2d_summary(ShowPlot=False)
NMDS_df, pred_Energy, mds_mesh = GAM_fitting(twodim_summary,fraction=0.5)
draw_contour_plot(NMDS_df, pred_Energy, mds_mesh, max_energy = 3, min_energy = -20)