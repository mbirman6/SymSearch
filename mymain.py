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


dogen=True
# dogen=False
if dogen:
    dogenplots=True
    dogenplots=False
        

def main():
    c=myconfig() # configuration variables
    ##### ------------------------------------------------------------- GENERATE SAMPLES + STATS ----------------------------------------------------------------------------
    ## ------ Definition and generation of signal shapes
    print("\n>>>> ------ Definition and generation of signal shapes")
    [c.SIGNALS_SHAPES_DENSITIES,
     c.X_AXIS_SIGNAL_LOW_HIGH_INDICES,
     c.Y_AXIS_SIGNAL_LOW_HIGH_INDICES]=CreateSignalShapesDensities(c)
    # if dogenplots:
    #     for i,curr_signal_shape_template in enumerate(c.SIGNALS_SHAPES_DENSITIES):
    #         X_axis_signal_low_index,X_axis_signal_high_index=c.X_AXIS_SIGNAL_LOW_HIGH_INDICES[i]
    #         Y_axis_signal_low_index,Y_axis_signal_high_index=c.Y_AXIS_SIGNAL_LOW_HIGH_INDICES[i]
    #         PlotMatrix(c,curr_signal_shape_template, strTitle = GetSignalStringByIndex(c,i),x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #         print('The total AREA of the current signal is: ', np.sum(curr_signal_shape_template))
    #         print('The total AREA of the current signal is (WITHIN RESTRICTED INDICES): ', 
    #                 np.sum(curr_signal_shape_template [X_axis_signal_low_index:X_axis_signal_high_index,
    #                                                     Y_axis_signal_low_index:Y_axis_signal_high_index]))
    #         print()

    ## ------ Background template settings
    # Happens in config
    template_file = np.load(c.LOAD_TEMPLATE_FILE_NAME)
    print("\n>>>> ------ Background template settings")
    # Plots information about the template
    print('The samples included in it are: ', template_file['samples'])
    print('The data SHAPE for each property of this measurement is: ' + ' entries: ', template_file['entries'].shape, 
            ', errors: ', template_file['errors'].shape, ', edges: ', template_file['edges'].shape)
    print('The data TYPE for each property of this measurement is: ' + ' entries: ', template_file['entries'].dtype, 
            ', errors: ', template_file['errors'].dtype, ', edges: ', template_file['edges'].dtype)
    template_min = np.min(template_file['entries'])
    template_max = np.max(template_file['entries'])
    print('\nHere is some statistical information about the sample:')
    print('------------------------------------------------------')
    print('Total entries: ' + str(np.sum(template_file['entries'])) + ' +\- ' + str(np.sqrt(np.sum(np.square(template_file['errors'])))))
    print('Min entry: ', template_min)
    print('Max entry: ', template_max)
    print('Median entry: ', np.median(template_file['entries']))
    print('Mean entry +\- Std: ', np.mean(template_file['entries']), '+\-', np.std(template_file['entries']))
    print('Indices of bins with zero entries: ', np.argwhere(template_file['entries'] == 0))
    # if dogenplots:
    #     if (c.SHOULD_LOAD_TEMPLATE == True):
    #         # Plots the template before a shift with a log scale
    #         PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE_noshift, strTitle = 'Loaded Template before shift',x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #         PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE_noshift, strTitle = 'Loaded Template before shift (log scale)', log_scale = True,x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #         # Plots the template after the shift
    #         PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE, strTitle = 'Loaded Template after a shift of ' + str(c.SHIFT_TEMPLATE_BY_NUM_OF_ENTRIES),x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #         PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE, strTitle = 'Loaded Template after a shift of ' + str(c.SHIFT_TEMPLATE_BY_NUM_OF_ENTRIES) + ' (log scale)', log_scale = True,x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #     elif (c.SHOULD_LOAD_TEMPLATE == False):
    #         # Plots the template
    #         PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE, strTitle = 'Generated template with a constant poisson parameter of ' + str(c.POISSON_PARAMETER),x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    # Plots all the indices of the bins with zero entries
    print('\n========================================\n')
    print('Indices of bins with zero entries: ', np.argwhere(c.BACKGROUND_ONLY_TEMPLATE == 0))
    
    ## ------ Definition and generation of LHC datasets
    print("\n>>>> ------ Definition and generation of LHC datasets")
    c.LHC_DATASETS, mu_magnitudes_per_dataset, mu_sanity_check, qzero_sanity_check = CreateLHCDatasets(c)
    print('DONE.')
    # if dogenplots:
    #     for curr_dataset_index in range(len(c.LHC_DATASETS)):
    #         print('This is data set ', curr_dataset_index)
    #         PlotMatrix(c,c.LHC_DATASETS[curr_dataset_index, 0, :, :], 'The signal and background of the measurement number 0',x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    #         PlotMatrix(c,c.LHC_DATASETS[curr_dataset_index, c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET - 1, :, :], 'The signal and background of the measurement number ' + str(c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET - 1),x_axis=c.X_AXIS,y_axis=c.Y_AXIS)
    print(c.LHC_DATASETS.shape)
    np.save("test_gen_data",c.LHC_DATASETS)
    test=np.load("test_gen_data.npy")
    print(test.shape)

    # ## ------ Plots some statistics of the generated datasets
    # if dogenplots:
    #     print("\n>>>> ------ Plots some statistics of the generated datasets")
    #     get_gendata_stats(c,mu_magnitudes_per_dataset,qzero_sanity_check)
    #     ## More detailed statistics about the generated datsets
    #     get_gendata_stats_mudef(c,mu_magnitudes_per_dataset)
    #     # get_gendata_stats_q0def_vs_q0dec(c,qzero_sanity_check) ## NOT WORK
    #     get_gendata_stats_mudef_vs_mudec(c,mu_magnitudes_per_dataset,mu_sanity_check)
    #     get_gendata_stats_mudefsum_vs_mudeflik(c,mu_magnitudes_per_dataset)
    #     # PlotAllSignalsNBackgroundNDatasets(c,save_plots = True)
    # print('FINISHED.')
    # ##### ------------------------------------------------------------- END: GENERATE SAMPLES + STATS ----------------------------------------------------------------------------

    # ##### -------------------------------------------------------------   PERFORM THE SEARCHES   ----------------------------------------------------------------------------
    # ## ------ General Search - calc test-statistic dists
    # ## Getting the test-statistics (e.g Diff, NSigma) distributions
    # ## Here we calculate the test-statistic distributions for each matrix in the LHC dataset that we consider as the "background only". We consider all the distributions (original - to which we don't have access in real life, and approximated - to which we want to compare the original). Later, we calculate the NSigma scores of the matrices from the other LHC datasets and use this score to calculate their related p-values.
    # print("\n>>>> ------ General Search - calc test-statistic dists")
    # c.ORIGINAL_MATRIX_DISTRIBUTION = c.LHC_DATASETS[c.BACKGROUND_DATASET_INDEX, :, :, :]
    # ORIGINAL_LIKE_MATRIX_DISTRIBUTION, _, _, _ = CreateLHCDatasets(c,c.BACKGROUND_DATASET_INDEX)
    # c.ORIGINAL_LIKE_MATRIX_DISTRIBUTION = ORIGINAL_LIKE_MATRIX_DISTRIBUTION[0, :, :, :]

    # ## Option 1: Calculating the test-statistic distributions
    # # Measures the test start time
    # test_start_time = time.process_time()
    # # Initializes the array that saves the histograms raw data
    # # (10K matrices) * (3 distributions) *  (10K NSigma results OR num_of_distribution_statistics)
    # all_histogram_raw_data = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                   c.NUM_OF_DISTRIBUTION_METHODS,
    #                                   c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET), np.nan)
    # all_histogram_statistics = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                     c.NUM_OF_DISTRIBUTION_METHODS,
    #                                     c.NUM_OF_DISTRIBUTION_STATISTICS),
    #                                    np.nan)
    # print('all_histogram_raw_data.shape', all_histogram_raw_data.shape)
    # print('all_histogram_statistics.shape', all_histogram_statistics.shape)
    # # Initializes the arrays that saves the bins and entries for each histogram
    # # (10K matrices) * (3 distributions) *  (num_of_bins_edges OR num_of_bins)
    # all_histogram_bins = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                               c.NUM_OF_DISTRIBUTION_METHODS,
    #                               c.HISTOGRAM_BIN_DIVISION), np.nan)
    # all_histograms_bins_centers = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                        c.NUM_OF_DISTRIBUTION_METHODS,
    #                                        c.HISTOGRAM_BIN_DIVISION - 1), np.nan)
    # all_histogram_entries = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                  c.NUM_OF_DISTRIBUTION_METHODS,
    #                                  c.HISTOGRAM_BIN_DIVISION - 1), np.nan)
    # print('all_histogram_bins.shape', all_histogram_bins.shape)
    # print('all_histograms_bins_centers.shape', all_histograms_bins_centers.shape)
    # print('all_histogram_entries.shape', all_histogram_entries.shape)
    # # Initializes the array that saves the parameters of the Gaussian fit for each matrix distribution approximation
    # # (10K different background_matrices) * (3 different backgrouund_distributions) * (3 gaussian_fit_parameters)
    # all_gaussian_fits_popts = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                    c.NUM_OF_DISTRIBUTION_METHODS,
    #                                    c.NUM_OF_GAUSSIAN_POPT),
    #                                   np.nan)
    # all_gaussian_fits_residuals = np.full((c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON,
    #                                        c.NUM_OF_DISTRIBUTION_METHODS,
    #                                        c.HISTOGRAM_BIN_DIVISION - 1), np.nan)
    # print('all_gaussian_fits_popts.shape', all_gaussian_fits_popts.shape)
    # print('all_gaussian_fits_residuals.shape', all_gaussian_fits_residuals.shape)
    # # Initializes the array that saves the p-values
    # # (5 background-non_backround LHC datasets pairs) * (3 different backgrouund_distributions) * (3 CDF/P-VALUE/Z) * (10K different background_matrices)
    # all_stat_hypoth_test_values = np.full((c.NUMBER_OF_ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS,
    #                                        c.NUM_OF_DISTRIBUTION_METHODS,
    #                                        c.NUM_OF_DIFFERENT_STAT_HYPOTH_TEST_VALUES,
    #                                        c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON), np.nan)
    # print('all_stat_hypoth_test_values.shape', all_stat_hypoth_test_values.shape)
    # # Calculates the global NSigma test score between the background and non-background matrices (same matrix index in both LHC datasets)
    # # (5 background-non_backround pairs) * (10K different background_matrices)
    # local_NSigma_non_background_test_results = np.array([c.NSigma_test(c.LHC_DATASETS[non_background_index, :, :, :],
    #                                                                  c.LHC_DATASETS[background_index, :, :, :]) \
    #                                                      for (background_index, non_background_index) in c.ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS])
    # print('local_NSigma_non_background_test_results.shape', local_NSigma_non_background_test_results.shape)
    # global_NSigma_non_background_test_results = np.mean(local_NSigma_non_background_test_results, axis = (2, 3))
    # print('global_NSigma_non_background_test_results.shape' , global_NSigma_non_background_test_results.shape)
    # # DEBUG: Checks that the NSigma quick calculation indeed performs what I think
    # #sampled_background_matrix = np.copy(LHC_measurements_datasets[BACKGROUND_DATASET_INDEX, 5, :, :])
    # #sampled_non_background_matrix = np.copy(LHC_measurements_datasets[3, 5, :, :])
    # #sampled_local_NSigma_test_result = NSigma_test(sampled_non_background_matrix, sampled_background_matrix)
    # #sampled_global_NSigma_test_results = np.mean(sampled_local_NSigma_test_result)
    # #print('DEBUG NSigma - START')
    # #print(sampled_local_NSigma_test_result - local_NSigma_non_background_test_results[2, 5, :, :])
    # #print(sampled_global_NSigma_test_results - global_NSigma_non_background_test_results[2, 5])
    # #print('DEBUG NSigma - END')
    # # Goes over all the matrices in the "background only" dataset
    # for curr_background_matrix_index in range(c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON): # NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET
    #     print('\nProcessing background matrix number ', curr_background_matrix_index, ' out of ',c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON)
    #     # Gets the current "background only" matrix
    #     curr_background_matrix = np.copy(c.ORIGINAL_MATRIX_DISTRIBUTION[curr_background_matrix_index, :, :])
    #     # Gets the current background matrix original and 2 approximate distributions
    #     curr_background_matrix_distributions = GetDistributionsOfMatrix(c,curr_background_matrix,c.ORIGINAL_LIKE_MATRIX_DISTRIBUTION)
    #     # Perform NSigma test of the distributions against the current background matrix (local score)
    #     curr_local_NSigma_background_test_distributions = c.NSigma_test(curr_background_matrix_distributions, curr_background_matrix)
    #     # Get the global score of the distributions against the current background matrix
    #     curr_global_NSigma_background_test_distributions = np.mean(curr_local_NSigma_background_test_distributions, axis = (2, 3))
    #     # Saves the current background only matrix NSigma distributions
    #     all_histogram_raw_data[curr_background_matrix_index, :, :] = np.copy(curr_global_NSigma_background_test_distributions)
    #     # Saves the current background only matrix NSigma distributions statistics
    #     all_histogram_statistics[curr_background_matrix_index, :, c.DISTRIBUTION_MEAN_INDEX] = np.mean(curr_global_NSigma_background_test_distributions , axis = 1)
    #     all_histogram_statistics[curr_background_matrix_index, :, c.DISTRIBUTION_STD_INDEX] = np.std(curr_global_NSigma_background_test_distributions , axis = 1)
    #     # Gets the histograms of each distribution (optional: plot one example for each distribution)
    #     for curr_distribution_method_index in range(c.NUM_OF_DISTRIBUTION_METHODS):
    #         # Gets the current distribution of the NSigma scores
    #         curr_background_distribution_scores = np.copy(curr_global_NSigma_background_test_distributions[curr_distribution_method_index, :])
    #         # Calculates the bins and bins centers for the current distribution
    #         arrBins_curr_histogram, arrBinsCenters_curr_histogram =    \
    #                                     CreateBinsForHistogram(curr_background_distribution_scores, bin_division = c.HISTOGRAM_BIN_DIVISION)
    #         # Gets the current histogram entries (using numpy and not matplotlib since I don't want to plot it)
    #         arrBinsEntries_curr_histogram, _ = np.histogram(curr_background_distribution_scores.flatten(), bins = arrBins_curr_histogram)
    #         # arrBinsEntries_curr_plt_histogram, _, _ = plt.hist(curr_background_distribution_scores.flatten(), bins = arrBins_curr_histogram)
    #         # plt.show()
    #         # input()
    #         # Saves the histogram bins and entries
    #         all_histogram_bins[curr_background_matrix_index, curr_distribution_method_index, :] = np.copy(arrBins_curr_histogram)
    #         all_histograms_bins_centers[curr_background_matrix_index, curr_distribution_method_index, :] = np.copy(arrBinsCenters_curr_histogram)
    #         all_histogram_entries[curr_background_matrix_index, curr_distribution_method_index, :] = np.copy(arrBinsEntries_curr_histogram)
    #         #print('arrBins_curr_histogram.shape', arrBins_curr_histogram.shape)
    #         #print('arrBinsEntries_curr_histogram.shape', arrBinsEntries_curr_histogram.shape)
    #         # Performs a Gaussian fit for the current histogram
    #         popt_curr_histogram, pcov_curr_histogram, arrYResiduals_curr_histogram = \
    #                                                                                 PerformGaussianFit(arrBinsCenters_curr_histogram, arrBinsEntries_curr_histogram)
    #         # Saves the gaussian fit parameters and residuals
    #         all_gaussian_fits_popts[curr_background_matrix_index, curr_distribution_method_index, :] = np.copy(popt_curr_histogram)
    #         all_gaussian_fits_residuals[curr_background_matrix_index, curr_distribution_method_index, :] = np.copy(arrYResiduals_curr_histogram)
    #         # Calculates the total area of the Gaussian fit
    #         total_integral_of_gaussian_fit, _ = quad(MyGaussian, -np.inf, np.inf,
    #                                                      args = (popt_curr_histogram[c.POPT_GAUSSIAN_MEAN],
    #                                                                  popt_curr_histogram[c.POPT_GAUSSIAN_STD],
    #                                                                  popt_curr_histogram[c.POPT_GAUSSIAN_AMPLITUDE]))
    #         # Goes over all the scores of the background matrices
    #         for curr_background_non_background_pair_index in range(c.NUMBER_OF_ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS):
    #             #(curr_background_dataset_index, curr_non_background_dataset_index) = c.ALL_BACKGROUND_AND_NON_BACKGROUND_PAIRS[curr_background_non_background_pair_index]
    #             # Gets the current scores of the non-background matrices
    #             curr_global_NSigma_non_background_matrix_test_result = np.copy(global_NSigma_non_background_test_results[curr_background_non_background_pair_index, curr_background_matrix_index])
    #             # Calculates the p-value
    #             integral_right_non_background_matrix, _ = quad(MyGaussian, curr_global_NSigma_non_background_matrix_test_result, np.inf,
    #                                                                args = (popt_curr_histogram[c.POPT_GAUSSIAN_MEAN],
    #                                                                            popt_curr_histogram[c.POPT_GAUSSIAN_STD],
    #                                                                            popt_curr_histogram[c.POPT_GAUSSIAN_AMPLITUDE]))
    #             p_value_non_background_matrix = integral_right_non_background_matrix/total_integral_of_gaussian_fit
    #             # Calculates the CDF
    #             # integral_left_non_background_matrix, _ = quad(Gaussian, -np.inf, curr_global_NSigma_non_background_matrix_test_result,
    #             #                                                                                                args = (popt_curr_histogram[c.POPT_GAUSSIAN_MEAN],
    #             #                                                                                                                popt_curr_histogram[c.POPT_GAUSSIAN_STD],
    #             #                                                                                                                popt_curr_histogram[c.POPT_GAUSSIAN_AMPLITUDE]))
    #             CDF_non_background_matrix = 1 - p_value_non_background_matrix # integral_left_non_background_matrix/total_integral_of_gaussian_fit
    #             # Calculates Z
    #             Z_non_background_matrix = norm.ppf(CDF_non_background_matrix)
    #             #print('CDF_non_background_matrix', CDF_non_background_matrix)
    #             #print('Z_non_background_matrix', Z_non_background_matrix)
    #             #print('p_value_non_background_matrix', p_value_non_background_matrix)
    #             # Saves the p-values
    #             all_stat_hypoth_test_values[curr_background_non_background_pair_index, curr_distribution_method_index, c.P_VALUE_INDEX, curr_background_matrix_index] = p_value_non_background_matrix
    #             all_stat_hypoth_test_values[curr_background_non_background_pair_index, curr_distribution_method_index, c.CDF_INDEX, curr_background_matrix_index] = CDF_non_background_matrix
    #             all_stat_hypoth_test_values[curr_background_non_background_pair_index, curr_distribution_method_index, c.Z_INDEX, curr_background_matrix_index] = Z_non_background_matrix
    #     # Checks if the data should be saved
    #     if ((c.SHOULD_SAVE_RESULTS == True) and
    #             (((curr_background_matrix_index + 1) % c.MIDDLE_BACKUP_EVERY == 0) or
    #              ((curr_background_matrix_index + 1) == c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON))):
    #         # Saves the data
    #         print('Saving data...')
    #         day_time_string = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    #         curr_file_title = c.SAVE_RESULTS_FILE_NAME + '_' + str(curr_background_matrix_index + 1) + '_out_of_' + str(c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON) + "_samples_" + day_time_string
    #         curr_data_file_name = GenerateFileName(curr_file_title, '.npz', include_date = False)
    #         curr_signal_densities_file_name = GenerateFileName('Signal_densities_' + day_time_string, '.txt', include_date = False)
    #         curr_lhc_datasets_file_name = GenerateFileName('LHC_datasets_' + day_time_string, '.txt', include_date = False)
    #         curr_zip_file_name = GenerateFileName(curr_file_title, '.zip', include_date = False)
    #         # Saves results
    #         np.savez_compressed(curr_data_file_name,
    #                                                 all_histogram_raw_data = all_histogram_raw_data,
    #                                                 all_histogram_statistics = all_histogram_statistics,
    #                                                 all_histogram_bins = all_histogram_bins,
    #                                                 all_histograms_bins_centers = all_histograms_bins_centers,
    #                                                 all_histogram_entries = all_histogram_entries,
    #                                                 all_gaussian_fits_popts = all_gaussian_fits_popts,
    #                                                 all_gaussian_fits_residuals = all_gaussian_fits_residuals,
    #                                                 all_stat_hypoth_test_values = all_stat_hypoth_test_values,
    #                                                 global_NSigma_non_background_test_results = global_NSigma_non_background_test_results,
    #                                                 BACKGROUND_ONLY_TEMPLATE = c.BACKGROUND_ONLY_TEMPLATE)
    #         print(curr_data_file_name + ' was saved.')
    #         # Saves the current signal densities configurations
    #         with open(curr_signal_densities_file_name, 'w') as fout:
    #             json.dump(c.SIGNAL_DENSITY_SHAPES_PROPERTIES, fout)
    #         print(curr_signal_densities_file_name + ' was saved.')
    #         # Saves the current lhc datasets configurations
    #         with open(curr_lhc_datasets_file_name, 'w') as fout:
    #             json.dump(c.LHC_DATASETS_PROPERTIES, fout)
    #         print(curr_lhc_datasets_file_name + ' was saved.')
    #         # # Compresses the data file to zip
    #         # !zip $curr_zip_file_name $curr_data_file_name $curr_signal_densities_file_name $curr_lhc_datasets_file_name
    #         # # Copies the results to drive
    #         # !cp $curr_zip_file_name $GDRIVE_RESULTS_PATH
    #         # print(curr_zip_file_name + ' was copied to ' + c.GDRIVE_RESULTS_PATH + ' :')
    #         # !ls $GDRIVE_RESULTS_PATH
    #         # Downloads the file
    #         #files.download(curr_data_file_name)
    # # Measures the test end time
    # test_finish_time = time.process_time()
    # test_total_time = test_finish_time - test_start_time
    # test_per_sample_time = test_total_time / c.TOTAL_NUM_OF_BACKGROUND_MATRICES_TO_RUN_ON
    # # Plots the total time it took
    # print('\nTotal MEASURED test time is: ', time.strftime("%H:%M:%S", time.gmtime(test_total_time)), '(HH:MM:SS)')
    # print('MEASURED AVERAGE test time PER SAMPLE is: ', time.strftime("%H:%M:%S", time.gmtime(test_per_sample_time)), '(HH:MM:SS)')
    # test_total_time_for_all_samples = test_per_sample_time * c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET
    # print('\nTotal PREDICTED test time for ALL SAMPLES is: ', time.strftime("%H:%M:%S", time.gmtime(test_total_time_for_all_samples)), '(HH:MM:SS)')
    
    
if __name__=="__main__":
    main()

