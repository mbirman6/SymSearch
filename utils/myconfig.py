import os
from shutil import copyfile
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

class myconfig():
    def __init__(self):
        self.debug=True
        ###### ------------ CONFIG ---------------- ##########
        SEED_VALUE = 42
        np.random.seed(SEED_VALUE) # For debug - random processes in code will yield the same results each run
        self.PLOT_P_VALUE_LOG_SCALE = True
        self.PLOT_FILE_FORMAT = '.pdf'
        DEFAULT_FIGURE_SIZE = [6.0, 4.0]
        self.DIM_OF_MATRIX = 28
        self.X_AXIS=np.copy([30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.,155.,160.,165.,170.])
        self.Y_AXIS=np.copy([10.,15.,20.,25.,30.,35.,40.,45.,50.,55.,60.,65.,70.,75.,80.,85.,90.,95.,100.,105.,110.,115.,120.,125.,130.,135.,140.,145.,150.])

        ## ------ Definition and generation of signal shapes
        self.NUM_OF_STDS_FOR_GAUSSIAN_SIGMA=2.5 # Num of std's away from guassian's mean to count sqrt(B) (in order to get 1 sigma)
        self.BASE_SHAPE='Base Shape'
        self.GAUSSIAN_SHAPE='Gaussian Shape'
        self.RECTANGLE_SHAPE='Rectangle Shape'
        self.MEAN_VALUES='Mean Values'
        self.STD_VALUES='STD Values'
        ## Define signal parameters (see down in the create template from template) - for gaussian, for rectangle signal
        self.NUM_OF_DIFFERENT_SIGNAL_DENSITY_SHAPES = 1
        self.SIGNAL_DENSITY_SHAPES_PROPERTIES = [{} for i in range(self.NUM_OF_DIFFERENT_SIGNAL_DENSITY_SHAPES)]
        # SIGNAL_DENSITY_SHAPES_PROPERTIES[0] = {
            #                                    BASE_SHAPE: RECTANGLE_SHAPE,  # Low statistics region
            #                                    MEAN_VALUES: (125, 125), # GeV # (125000, 125000) MeV
            #                                    STD_VALUES: (25, 25)} # GeV # (25000, 25000) MeV
        self.SIGNAL_DENSITY_SHAPES_PROPERTIES[0] = {
                                               self.BASE_SHAPE: self.GAUSSIAN_SHAPE,   # Low statistics region
                                               self.MEAN_VALUES: (125, 125), # GeV # (125000, 125000) MeV
                                               self.STD_VALUES: (10, 10)} # GeV # (10000, 10000) MeV 
        # SIGNAL_DENSITY_SHAPES_PROPERTIES[2] = {
        #                                        BASE_SHAPE: RECTANGLE_SHAPE,  # High statistics region
        #                                        MEAN_VALUES: (65, 35), # GeV # (65000, 35000) MeV
        #                                        STD_VALUES: (25, 25)} # GeV # (25000, 25000) MeV
        # SIGNAL_DENSITY_SHAPES_PROPERTIES[3] = {
        #                                        BASE_SHAPE: GAUSSIAN_SHAPE,  # High statistics region
        #                                        MEAN_VALUES: (65, 35), # GeV # (65000, 35000) MeV
        #                                        STD_VALUES: (10, 10)} # GeV # (10000, 10000) MeV                              # Creates all the signal shapes densities
        self.SIGNALS_SHAPES_DENSITIES=0
        self.X_AXIS_SIGNAL_LOW_HIGH_INDICES=0
        self.Y_AXIS_SIGNAL_LOW_HIGH_INDICES=0

        ## ------ Background template settings
        self.SHOULD_LOAD_TEMPLATE = True #False # True
        self.LOAD_TEMPLATE_FILE_NAME = 'em_reco_none_Mcoll_Lep0Pt_28_28_no_signal.npz'

        ## ------ Definition and generation of LHC datasets
        # Define the total number of matrices in each sample (regular_1, regular_2, regular_3, with_signal_gaussian, with_signal_rectangle)
        self.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET = 100
        self.LHC_DATASET_ID = 'LHC Dataset ID'
        self.LHC_DATASET_TYPE = 'LHC Dataset Type'
        self.BACKGROUND_ONLY_DATASET = 'Background Only '
        self.BACKGROUND_SIGNAL_DATASET = 'Background and Signal'
        self.SIGNAL_DENSITY_SHAPE_INDEX = 'Signal Density Index'
        self.NUM_OF_SIGMA_OF_SIGNAL = 'Number of Sigmas'
        self.POISSON_TIMING = 'Poisson Timing'
        self.BEFORE_SIGNAL_ADDITION = 'Before Signal addition'
        self.LHC_DATASET_COMMENT = 'LHC Dataset Comment'
        self.SIGNAL_GEN_METHOD = 'Signal generation method'
        self.SUM_METHOD = 'Sum method'
        self.LIK_METHOD = 'Likelihood method'
        # # Define the properties of the LHC measurements datasets
        # NUM_OF_DIFFERENT_LHC_DATASETS = 4
        # LHC_DATASETS_PROPERTIES = [{} for i in range(NUM_OF_DIFFERENT_LHC_DATASETS)]
        # LHC_DATASETS_PROPERTIES[0] = {LHC_DATASET_ID: 0,
        #                               LHC_DATASET_TYPE: BACKGROUND_ONLY_DATASET,
        #                               LHC_DATASET_COMMENT: ''}
        # LHC_DATASETS_PROPERTIES[1] = {LHC_DATASET_ID: 1,
        #                               LHC_DATASET_TYPE: BACKGROUND_ONLY_DATASET,
        #                               LHC_DATASET_COMMENT: ''}
        # LHC_DATASETS_PROPERTIES[2] = {LHC_DATASET_ID: 2,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[3] = {LHC_DATASET_ID: 3,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[4] = {LHC_DATASET_ID: 4,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 1,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[5] = {LHC_DATASET_ID: 5,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 1,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[6] = {LHC_DATASET_ID: 6,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 2,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[7] = {LHC_DATASET_ID: 7,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 2,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[8] = {LHC_DATASET_ID: 8,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 3,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[9] = {LHC_DATASET_ID: 9,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 3,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: SUM_METHOD}
        # LHC_DATASETS_PROPERTIES[10] = {LHC_DATASET_ID: 10,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[11] = {LHC_DATASET_ID: 11,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[12] = {LHC_DATASET_ID: 12,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 1,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[13] = {LHC_DATASET_ID: 13,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 1,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'low statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[14] = {LHC_DATASET_ID: 14,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 2,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[15] = {LHC_DATASET_ID: 15,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 2,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[16] = {LHC_DATASET_ID: 16,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 3,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        # LHC_DATASETS_PROPERTIES[3] = {LHC_DATASET_ID: 3,
        #                               LHC_DATASET_TYPE: BACKGROUND_SIGNAL_DATASET,
        #                               SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               NUM_OF_SIGMA_OF_SIGNAL: 5,
        #                               POISSON_TIMING: BEFORE_SIGNAL_ADDITION,
        #                               LHC_DATASET_COMMENT: 'high statistics',
        #                               SIGNAL_GEN_METHOD: LIK_METHOD}
        self.NUM_OF_DIFFERENT_LHC_DATASETS = 2
        self.LHC_DATASETS_PROPERTIES = [{} for i in range(self.NUM_OF_DIFFERENT_LHC_DATASETS)]
        self.LHC_DATASETS_PROPERTIES[0] = {self.LHC_DATASET_ID: 0,
                                      self.LHC_DATASET_TYPE: self.BACKGROUND_ONLY_DATASET,
                                      self.LHC_DATASET_COMMENT: ''}
        self.LHC_DATASETS_PROPERTIES[1] = {self.LHC_DATASET_ID: 1,
                                      self.LHC_DATASET_TYPE: self.BACKGROUND_SIGNAL_DATASET,
                                      self.SIGNAL_DENSITY_SHAPE_INDEX: 0,
                                      self.NUM_OF_SIGMA_OF_SIGNAL: 3,
                                      self.POISSON_TIMING: self.BEFORE_SIGNAL_ADDITION,
                                      self.LHC_DATASET_COMMENT: 'low statistics',
                                      self.SIGNAL_GEN_METHOD: self.LIK_METHOD}
        # self.LHC_DATASETS_PROPERTIES[2] = {self.LHC_DATASET_ID: 2,
        #                               self.LHC_DATASET_TYPE: self.BACKGROUND_SIGNAL_DATASET,
        #                               self.SIGNAL_DENSITY_SHAPE_INDEX: 0,
        #                               self.NUM_OF_SIGMA_OF_SIGNAL: 3,
        #                               self.POISSON_TIMING: self.BEFORE_SIGNAL_ADDITION,
        #                               self.LHC_DATASET_COMMENT: 'low statistics',
        #                               self.SIGNAL_GEN_METHOD: self.SUM_METHOD}

        ## ------ General Search - calc test-statistic dists
        ## GetDistributionsOfMatrix
        self.NUM_OF_DISTRIBUTION_METHODS = 3
        self.ORIG_DIST_INDEX = 0
        self.POS_APPROX_INDEX = 1 # POIS(X)
        self.POIS_GAUSS_APPROX_INDEX = 2 # POIS(GAUSS(X, SQRT(X)))
        ##
        self.NUM_OF_GAUSSIAN_POPT = 3
        self.POPT_GAUSSIAN_MEAN = 0
        self.POPT_GAUSSIAN_STD = 1
        self.POPT_GAUSSIAN_AMPLITUDE = 2
        ##
        self.BACKGROUND_DATASET_INDEX = 0 # The index of the dataset that is used as the "background only" in the analysis
        self.ALL_NON_BACKGROUND_DATASETS_INDICES = [curr_index for curr_index in range(self.NUM_OF_DIFFERENT_LHC_DATASETS) if curr_index != self.BACKGROUND_DATASET_INDEX]
        self.ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS = [(self.BACKGROUND_DATASET_INDEX, curr_non_background_index) for curr_non_background_index in self.ALL_NON_BACKGROUND_DATASETS_INDICES]
        self.NUMBER_OF_ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS = len(self.ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS)
        self.HISTOGRAM_BIN_DIVISION = 50
        self.NUM_OF_DIFFERENT_STAT_HYPOTH_TEST_VALUES = 3 # number of different statistics
        self.CDF_INDEX = 0 # CDF
        self.P_VALUE_INDEX = 1 # P_VALUE_TO_THE_RIGHT_INDEX
        self.Z_INDEX = 2  # Z = \PHI^{-1}(CDF)
        self.NUM_OF_DISTRIBUTION_STATISTICS = 2
        self.DISTRIBUTION_MEAN_INDEX = 0
        self.DISTRIBUTION_STD_INDEX = 1
        # Define the tests (TODO: maybe create an array of test functions)
        self.NSigma_test =  lambda  minuend, subtrahend : np.divide(minuend - subtrahend, np.sqrt(subtrahend)) #lambda  minuend, subtrahend : np.divide(minuend - subtrahend, np.sqrt(minuend + subtrahend)) # lambda  minuend, subtrahend : np.divide(minuend - subtrahend, np.sqrt(subtrahend)) #lambda  minuend, subtrahend : np.divide(minuend - subtrahend, np.sqrt(minuend + subtrahend))
        #Diff_norm_by_pois_param_test = lambda  minuend, subtrahend : np.divide(minuend - subtrahend, np.sqrt(POISSON_PARAMETER))
        self.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON = self.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET #NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET # NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET
        self.SHOULD_SAVE_RESULTS = True
        self.SAVE_RESULTS_FILE_NAME = 'Poisson_100_background_Diff_test_likelihood_signals' # 'Poisson_100_background_Diff_test_likelihood_signals' #'Poisson_100_background_Diff_test_new' #'Poisson_100_background_NSigma_test_new' # 'em_reco_none_no_signal_NSigma_test' 'em_reco_none_no_signal_NSigma_test_likelihood_signals'
        self.MIDDLE_BACKUP_EVERY = 2000 # iterations
        # print(self.SAVE_RESULTS_FILE_NAME)

    def get_template(self):
        GDRIVE_TEMPLATE_PATH="fromOphir/Templates"
        # Checks if template should be loaded or just generate a flat one (i.e constant Poisson parameter)
        if (self.SHOULD_LOAD_TEMPLATE == True):
            # If a template should be loaded
            # Sets the amount of entries the template will be shifted by
            self.SHIFT_TEMPLATE_BY_NUM_OF_ENTRIES = 25
            # Gets the full path name of the template
            template_full_path = GDRIVE_TEMPLATE_PATH + '/' + self.LOAD_TEMPLATE_FILE_NAME
            # Copies it to local disk
            if self.debug:
                print("myconfig: Copying background template %s"%self.LOAD_TEMPLATE_FILE_NAME)
            copyfile(template_full_path,"./%s"%self.LOAD_TEMPLATE_FILE_NAME)
            # Loads the file that contains the template
            template_file = np.load(self.LOAD_TEMPLATE_FILE_NAME) #dict(np.load(LOAD_RESULTS_FILE_NAME))
            # Gets the template
            self.BACKGROUND_ONLY_TEMPLATE_noshift = np.copy(template_file['entries'])
            # Shifts the template
            self.BACKGROUND_ONLY_TEMPLATE = self.BACKGROUND_ONLY_TEMPLATE_noshift + self.SHIFT_TEMPLATE_BY_NUM_OF_ENTRIES
            # Happens in config
            if self.debug:
                template_file = np.load(self.LOAD_TEMPLATE_FILE_NAME)
                template_min = np.min(template_file['entries'])
                template_max = np.max(template_file['entries'])
                print("\n>>>> ------ Background template settings")
                # Plots information about the template
                print('The samples included in it are: ', template_file['samples'])
                print('The data SHAPE for each property of this measurement is: ' + ' entries: ', template_file['entries'].shape, 
                        ', errors: ', template_file['errors'].shape, ', edges: ', template_file['edges'].shape)
                print('The data TYPE for each property of this measurement is: ' + ' entries: ', template_file['entries'].dtype, 
                        ', errors: ', template_file['errors'].dtype, ', edges: ', template_file['edges'].dtype)
                print('\nHere is some statistical information about the sample:')
                print('------------------------------------------------------')
                print('Total entries: ' + str(np.sum(template_file['entries'])) + ' +\- ' + str(np.sqrt(np.sum(np.square(template_file['errors'])))))
                print('Min entry: ', template_min)
                print('Max entry: ', template_max)
                print('Median entry: ', np.median(template_file['entries']))
                print('Mean entry +\- Std: ', np.mean(template_file['entries']), '+\-', np.std(template_file['entries']))
                print('Indices of bins with zero entries: ', np.argwhere(template_file['entries'] == 0))
        elif (self.SHOULD_LOAD_TEMPLATE == False):
            # If a template should be generated
            # Sets the constant poisson parameter for all the bins
            self.POISSON_PARAMETER = 100
            # Sets the template
            self.BACKGROUND_ONLY_TEMPLATE = self.POISSON_PARAMETER * np.ones((self.DIM_OF_MATRIX, self.DIM_OF_MATRIX)) 

        
