import sys
sys.path.insert(0,"utils")
from myconfig import *
from myfunctions import *
##
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import torch
import time
import copy
import glob
import json
import string
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
##
from operator import itemgetter
from scipy.optimize import curve_fit, fsolve, minimize
from scipy.integrate import quad
from scipy.stats import norm
# from google.colab import files
from datetime import datetime
from textwrap import fill


Ntrain=100
Ntest=10
template_types=['simul','flat']
signal_types=['rect-lowstat','rect-higstat','gaus-lowstat','gaus-higstat']

def main():
    ##### -------------------------------------------- GENERATE SAMPLES + STATS -------------------
    for template_type in template_types:
        c=myconfig()
        c.debug=False
        ## Configure template
        [c.SIGNALS_SHAPES_DENSITIES,
        c.X_AXIS_SIGNAL_LOW_HIGH_INDICES,
        c.Y_AXIS_SIGNAL_LOW_HIGH_INDICES]=CreateSignalShapesDensities(c)
        c.SHOULD_LOAD_TEMPLATE=True if template_type=='simul' else False
        c.get_template()
        ## Generate Ntrain bkg-only matrices for training
        print(">>> Generating %s bkg-only matrices for training with template-type \'%s\'"%(Ntrain,template_type))
        c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET=Ntrain
        c.NUM_OF_DIFFERENT_LHC_DATASETS=1
        c.LHC_DATASETS_PROPERTIES=[{
            c.LHC_DATASET_ID: 0,
            c.LHC_DATASET_TYPE: c.BACKGROUND_ONLY_DATASET,
            c.LHC_DATASET_COMMENT: ''}]
        c.LHC_DATASETS,bla,bla1,bla2=CreateLHCDatasets(c)
        savedata=np.squeeze(c.LHC_DATASETS)
        fname="gendata_template-%s_train.npy"%template_type
        np.save(fname,savedata)
        print("    Saved in \'%s\', shape = "%fname,savedata.shape)
        ## Loop on signal types
        for signal_type in signal_types:
            ## Configure signal
            if 'rect' in signal_type:
                shape=c.RECTANGLE_SHAPE
                stds=(25,25)
            else:
                shape=c.GAUSSIAN_SHAPE
                stds=(10,10)
            if 'lowstat' in signal_type:
                means=(125, 125)
                statlabel='low statistics'
            else:
                means=(65, 35)
                statlabel='high statistics'
            c.SIGNAL_DENSITY_SHAPES_PROPERTIES=[{
                c.BASE_SHAPE:shape,
                c.MEAN_VALUES:means,
                c.STD_VALUES:stds}]
            ## Generate Ntest pairs of bkg-only/bkg+sig matrices for validation/testing
            print("  >> Generating %s pairs of bkg-only/bkg+sig matrices for validation/testing with template-type \'%s\' and signal-type \'%s\'"%(Ntest,template_type,signal_type))
            c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET=Ntest
            c.NUM_OF_DIFFERENT_LHC_DATASETS=2
            c.LHC_DATASETS_PROPERTIES=[
                {   c.LHC_DATASET_ID:0,
                    c.LHC_DATASET_TYPE: c.BACKGROUND_ONLY_DATASET,
                    c.LHC_DATASET_COMMENT: ''},
                {   c.LHC_DATASET_ID:1,
                    c.LHC_DATASET_TYPE: c.BACKGROUND_SIGNAL_DATASET,
                    c.SIGNAL_DENSITY_SHAPE_INDEX:0,
                    c.NUM_OF_SIGMA_OF_SIGNAL: 3,
                    c.POISSON_TIMING: c.BEFORE_SIGNAL_ADDITION,
                    c.LHC_DATASET_COMMENT: statlabel,
                    c.SIGNAL_GEN_METHOD: c.LIK_METHOD}]
            c.LHC_DATASETS,bla,bla1,bla2=CreateLHCDatasets(c,usesamebkg=True)
            savedata=np.squeeze(c.LHC_DATASETS)
            # ## DEBUG
            # print(savedata[0][0][0][0],savedata[1][0][0][0])
            # print(np.sum(savedata[0]),np.sum(savedata[1]))
            fname="gendata_template-%s_signal-%s_test.npy"%(template_type,signal_type)
            np.save(fname,savedata)
            print("    Saved in \'%s\', shape = "%fname,savedata.shape)
    
if __name__=="__main__":
    main()

