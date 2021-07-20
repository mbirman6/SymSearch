import os
from shutil import copyfile
from myconfig import *
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
from scipy.optimize import curve_fit,fsolve,minimize
from scipy.integrate import quad
from scipy.stats import norm
# from google.colab import files
from datetime import datetime
from textwrap import fill
##

## ------ Definition of underlying distribution matrix
def GenerateFileName(str_main_file_name,str_extension,include_date=True):
    # Initializes the list of all the prohibited strings
    prohibited_strings=['\n',',','.','(',')','[',']','{','}','$','\\left','\\right','\\','`','*','>','#','+','!','\'','/','-']
    # Initializes the date string
    day_time_string=""
    # If the date should be added to the string
    if (include_date==True):
        # Gets the date
        day_time_string=" "+datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    # Sets the draft of the filename
    file_name=str_main_file_name+day_time_string
    # Replaces all the prohibited strings with a white space
    for curr_prohibited_string in prohibited_strings:
        if curr_prohibited_string in file_name:
            file_name=file_name.replace(curr_prohibited_string," ")
    # Merges all adjacent white spaces and replaces the white spaces with an underscore
    file_name='_'.join(file_name.split())+str_extension
    return file_name
def PlotMatrix(c,arrMatrix,strTitle,x_axis,y_axis,log_scale=False,save_output=False,main_file_name='PlotMatrix',x_label='X variable',y_label='Y variable'):
    # Reset rc parameters to default
    #mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams["figure.figsize"]=8,6
    plt.rcParams["figure.titlesize"]='x-large'
    plt.rcParams['axes.titlesize']=40 # title font size
    plt.rcParams['axes.labelsize']=24 # axes labels size
    plt.rcParams['xtick.labelsize']=20 # ticks size
    plt.rcParams['ytick.labelsize']=20 # ticks size
    plt.rcParams['legend.fontsize']='xx-large' # Valid font sizes are xx-small,x-small,small,medium,large,x-large,xx-large,smaller,larger.
    # Plots the current final template
    _,ax=plt.subplots(1)
    if (log_scale==False):
        pcm=ax.pcolor(x_axis,y_axis,np.transpose(arrMatrix),cmap= 'viridis')
    elif (log_scale==True):
        min_entry=np.min(arrMatrix[arrMatrix != 0])
        max_entry=np.max(arrMatrix[arrMatrix != 0])
        zero_normalization=min_entry/10
        arrCopy=np.copy(arrMatrix)
        arrCopy[arrCopy==0]=zero_normalization
        pcm=ax.pcolor(x_axis,y_axis,np.transpose(arrCopy),norm=colors.LogNorm(vmin=zero_normalization,vmax=max_entry),cmap= 'viridis')
    plt.colorbar(pcm,ax=ax,extend='max')
    plt.title(fill(strTitle,25))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    curr_plot_file_name=None
    # Checks if the plot should be saved
    if (save_output==True):
        # If the plot should be saved
        # Generates the filename of the current plot
        curr_plot_file_name=GenerateFileName(main_file_name,c.PLOT_FILE_FORMAT,include_date=True)
        # Saves the current plot
        plt.savefig(curr_plot_file_name)
    # Shows the plot
    plt.show()
    return curr_plot_file_name

## ------ Definition and generation of signal shapes
def GetSignalStringByIndex(c,signal_index,b_short_signal_string=False):
    signal_record=c.SIGNAL_DENSITY_SHAPES_PROPERTIES[signal_index]
    signal_shape=signal_record[c.BASE_SHAPE]
    signal_means=signal_record[c.MEAN_VALUES]
    signal_stds=signal_record[c.STD_VALUES]
    signal_x_location=str(signal_means[0])+r' $\pm$ '+str(signal_stds[0])
    signal_y_location=str(signal_means[1])+r' $\pm$ '+str(signal_stds[1])
    if (b_short_signal_string==False):
        # long string
        signal_string='Signal of a '+signal_shape+',located at (x,y)=('+signal_x_location+','+signal_y_location+')'
    else:
        # short string
        signal_string=signal_shape+'d signal'
    return signal_string
def GetBinIndexByLeftEdgeValue(left_limit_value,bins_edges):
    return np.where(bins_edges==left_limit_value)[0][0]
def GetBinEdges(bin_index,bins_edges):
    bin_left_edge=bins_edges[bin_index]
    bin_right_edge=bins_edges[bin_index+1]
    return bin_left_edge,bin_right_edge
def Gaussian_after_variable_change(x):
    return (1/np.sqrt(np.pi))*np.exp(-x**2)  
def Fill1DHistogramByGaussian(gaussian_mean,gaussian_std,axis_edges):
    gaussian_entries=np.zeros(axis_edges.shape[0]-1)
    # Calculates the gaussian contribution in the given axis
    for curr_axis_index in range(gaussian_entries.shape[0]):
        # Gets the current bin edges
        curr_bin_left_edge,curr_bin_right_edge=GetBinEdges(curr_axis_index,axis_edges)
        # Sets the lower and upper integral limits
        lower_integral_limit=(curr_bin_left_edge-gaussian_mean)/(np.sqrt(2)*gaussian_std)
        upper_integral_limit=(curr_bin_right_edge-gaussian_mean)/(np.sqrt(2)*gaussian_std)
        # Calculates the current contribution
        gaussian_entries[curr_axis_index],_=quad(Gaussian_after_variable_change,lower_integral_limit,upper_integral_limit)
    return gaussian_entries
def CreateSignalShapesDensities(c):
    signal_properties_records=c.SIGNAL_DENSITY_SHAPES_PROPERTIES
    matrix_dim=c.DIM_OF_MATRIX
    x_axis=c.X_AXIS
    y_axis=c.Y_AXIS
    # Gets the number of different signal shapes
    num_of_different_signals=len(signal_properties_records)
    # Initializes the array of the signals shapes densities and the array of the signal region indices in each axis
    arr_signals_shapes_densities=[]
    arr_X_axis_signal_low_high_indices=[]
    arr_Y_axis_signal_low_high_indices=[]
    # Creates the signals shapes densities
    for curr_signal_shape_index in range(num_of_different_signals):
        # Gets the current signal shape properties
        curr_signal_shape_properties=signal_properties_records[curr_signal_shape_index]
        if c.debug:
            print('This is a signal Number '+str(curr_signal_shape_index)+' in a '+curr_signal_shape_properties[c.BASE_SHAPE]+'.')
            print('Its MEAN values are: ',curr_signal_shape_properties[c.MEAN_VALUES],'.')
            print('Its STD values are: ',curr_signal_shape_properties[c.STD_VALUES],'.')
    curr_signal_shape_mean_X=curr_signal_shape_properties[c.MEAN_VALUES][0]
    curr_signal_shape_std_X=curr_signal_shape_properties[c.STD_VALUES][0]
    curr_signal_shape_mean_Y=curr_signal_shape_properties[c.MEAN_VALUES][1]
    curr_signal_shape_std_Y=curr_signal_shape_properties[c.STD_VALUES][1]
    # Creates an empty template for the current signal shape density
    curr_signal_shape_template=np.zeros((matrix_dim,matrix_dim))
    # Checks if the current signal shape is a rectangle or a guassian
    if (curr_signal_shape_properties[c.BASE_SHAPE]==c.RECTANGLE_SHAPE):
        # If it is rectangle
        # Get the indices of the signal region
        X_axis_signal_low_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_X-curr_signal_shape_std_X,x_axis)
        X_axis_signal_high_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_X+curr_signal_shape_std_X,x_axis)
        Y_axis_signal_low_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_Y-curr_signal_shape_std_Y,y_axis)
        Y_axis_signal_high_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_Y+curr_signal_shape_std_Y,y_axis)
        # Fills the rectangle such that the area of it is equal to 1
        total_rectangle_area_before_renormalization=(2*curr_signal_shape_std_X)*(2*curr_signal_shape_std_Y)
        curr_signal_shape_template[X_axis_signal_low_index:X_axis_signal_high_index,Y_axis_signal_low_index:Y_axis_signal_high_index]=((x_axis[1]-x_axis[0])*(y_axis[1]-y_axis[0]))/total_rectangle_area_before_renormalization
    elif (curr_signal_shape_properties[c.BASE_SHAPE]==c.GAUSSIAN_SHAPE):
        # If it is a gaussian
        # Get the indices of the signal region
        X_axis_signal_low_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_X-c.NUM_OF_STDS_FOR_GAUSSIAN_SIGMA*curr_signal_shape_std_X,x_axis)
        X_axis_signal_high_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_X+c.NUM_OF_STDS_FOR_GAUSSIAN_SIGMA*curr_signal_shape_std_X,x_axis)
        Y_axis_signal_low_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_Y-c.NUM_OF_STDS_FOR_GAUSSIAN_SIGMA*curr_signal_shape_std_Y,y_axis)
        Y_axis_signal_high_index=GetBinIndexByLeftEdgeValue(curr_signal_shape_mean_Y+c.NUM_OF_STDS_FOR_GAUSSIAN_SIGMA*curr_signal_shape_std_Y,y_axis)
        # Gets the gaussian contributions in each axes
        signal_x_axis_contributions=Fill1DHistogramByGaussian(curr_signal_shape_mean_X,curr_signal_shape_std_X,x_axis)
        signal_y_axis_contributions=Fill1DHistogramByGaussian(curr_signal_shape_mean_Y,curr_signal_shape_std_Y,y_axis)
        # Fills the signal only template
        for curr_X_index in range(signal_x_axis_contributions.shape[0]):
            for curr_Y_index in range(signal_y_axis_contributions.shape[0]): 
                # Sets the value of the current bin
                curr_signal_shape_template[curr_X_index,curr_Y_index]=signal_x_axis_contributions[curr_X_index]*signal_y_axis_contributions[curr_Y_index]
        # Normalizes the total area of the signal to be equal to one
        gaussian_total_sum=np.sum(curr_signal_shape_template)    
        for curr_X_index in range(signal_x_axis_contributions.shape[0]):
            for curr_Y_index in range(signal_y_axis_contributions.shape[0]): 
                # Sets the value of the current bin
                curr_signal_shape_template[curr_X_index,curr_Y_index]=(signal_x_axis_contributions[curr_X_index]*signal_y_axis_contributions[curr_Y_index])/gaussian_total_sum
    # Saves the current signal shape
    arr_signals_shapes_densities.append(curr_signal_shape_template)
    # Saves the current signal low and high indices in each axis
    arr_X_axis_signal_low_high_indices.append([X_axis_signal_low_index,X_axis_signal_high_index])
    arr_Y_axis_signal_low_high_indices.append([Y_axis_signal_low_index,Y_axis_signal_high_index])
    return arr_signals_shapes_densities,arr_X_axis_signal_low_high_indices,arr_Y_axis_signal_low_high_indices

## ------ Definition and generation of LHC datasets
def GetLHCDatasetStringByIndex(c,LHC_dataset_index):
    LHC_dataset_record=c.LHC_DATASETS_PROPERTIES[LHC_dataset_index]
    LHC_dataset_string='Background Only'
    if (LHC_dataset_record[c.LHC_DATASET_TYPE] != c.BACKGROUND_ONLY_DATASET):
        signal_record=c.SIGNAL_DENSITY_SHAPES_PROPERTIES[int(LHC_dataset_record[c.SIGNAL_DENSITY_SHAPE_INDEX])]
        signal_shape=signal_record[c.BASE_SHAPE]
        LHC_dataset_string='Background and a '+signal_shape+' signal of '+str(LHC_dataset_record[c.NUM_OF_SIGMA_OF_SIGNAL])+'$\sigma$' #+ ' '+LHC_dataset_record[c.SIGNAL_GEN_METHOD] #+' (signal index: '+str(LHC_dataset_record[c.SIGNAL_DENSITY_SHAPE_INDEX])+')'
    if (c.LHC_DATASET_COMMENT in LHC_dataset_record):
        if (LHC_dataset_record[c.LHC_DATASET_COMMENT].replace(" ","") != ''):
            LHC_dataset_string=LHC_dataset_string+' ('+LHC_dataset_record[c.LHC_DATASET_COMMENT]+')'
    return LHC_dataset_string
def GetLHCDatasetSignalGenMethodByIndex(c,LHC_dataset_index):
    LHC_dataset_record=c.LHC_DATASETS_PROPERTIES[LHC_dataset_index]
    LHC_dataset_string=''
    if (LHC_dataset_record[c.LHC_DATASET_TYPE] != c.BACKGROUND_ONLY_DATASET):
        LHC_dataset_string=LHC_dataset_record[c.SIGNAL_GEN_METHOD]
    return LHC_dataset_string
def Calcq0MinusZ(mu,B,S,z):
    SdivB=np.divide(S,B)
    ln_part=np.log(np.ones_like(B)+mu*SdivB)
    sum_part=np.multiply(B+mu*S,ln_part)
    sum_result=np.sum(sum_part)
    equation_value=2*(sum_result-mu)-z**2
    return equation_value
def CalcMinusq0(mu,X,R,S):
    SdivR=np.divide(S,R)
    ln_part=np.log(np.ones_like(R)+mu*SdivR)
    sum_part=np.multiply(X,ln_part)
    sum_result=np.sum(sum_part)
    equation_value=-2*(sum_result-mu)
    return equation_value
def CreateLHCDatasets(c,idx='',usesamebkg=False):
    if idx!='':
        lhc_datasets_properties_records=c.LHC_DATASETS_PROPERTIES[idx:idx+1]
    else:
        lhc_datasets_properties_records=c.LHC_DATASETS_PROPERTIES
    background_only_template=c.BACKGROUND_ONLY_TEMPLATE
    number_of_measurements_in_dataset=c.NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET
    arr_signals_shapes_densities=c.SIGNALS_SHAPES_DENSITIES
    arr_X_axis_signal_low_high_indices=c.X_AXIS_SIGNAL_LOW_HIGH_INDICES
    arr_Y_axis_signal_low_high_indices=c.Y_AXIS_SIGNAL_LOW_HIGH_INDICES
    # Gets the number of different datasets
    num_of_different_lhc_datasets=len(lhc_datasets_properties_records)
    # Sets the background only template for all the measurements in all the LHC datasets
    underlying_distribution_matrix=np.repeat(background_only_template[np.newaxis,:,:],
                                                number_of_measurements_in_dataset,axis=0)
    underlying_distribution_matrix=np.repeat(underlying_distribution_matrix[np.newaxis,:,:,:],
                                                num_of_different_lhc_datasets,axis=0)
    if c.debug:
        print('underlying_distribution_matrix.shape',underlying_distribution_matrix.shape)
    # Define the primary matrices in the LHC measurements datasets (i.e only background contribution)
    LHC_measurements_datasets=np.float32(np.random.poisson(underlying_distribution_matrix))
    if c.debug:
        print('LHC_measurements_datasets.shape',LHC_measurements_datasets.shape)
    # Initializes the signal strength (\mu) of all the measurements
    mu_magnitudes_of_measurements=np.zeros((num_of_different_lhc_datasets,number_of_measurements_in_dataset))
    mu_detected_sanity_check=np.zeros((num_of_different_lhc_datasets,number_of_measurements_in_dataset))
    qzero_detected_sanity_check=np.zeros((num_of_different_lhc_datasets,number_of_measurements_in_dataset))
    # Adds the signals to the suitable LHC measurements datasets
    # Goes over the datasets
    for curr_dataset_index in range(num_of_different_lhc_datasets):
        # Gets the current dataset properties record
        curr_dataset_properties=lhc_datasets_properties_records[curr_dataset_index]
        # # Prints the background only of the first measurement of the current dataset
        # print('This is data set ',curr_dataset_index)
        # PlotMatrix(c,LHC_measurements_datasets[curr_dataset_index,0,:,:],'The background only of the measurement number 0')
        # PlotMatrix(c,LHC_measurements_datasets[curr_dataset_index,number_of_measurements_in_dataset-1,:,:],
        #     'The background only of the measurement number '+str(number_of_measurements_in_dataset-1))
        # Checks if the current dataset has a signal contribution
        if (curr_dataset_properties[c.LHC_DATASET_TYPE] != c.BACKGROUND_ONLY_DATASET):
            # If there is a signal contribution
            # Gets the current signal shape index
            curr_signal_shape_index=int(curr_dataset_properties[c.SIGNAL_DENSITY_SHAPE_INDEX])
            # Gets the number of sigmas the signal should be over the background
            curr_num_sigma_of_signal=int(curr_dataset_properties[c.NUM_OF_SIGMA_OF_SIGNAL])
            # Checks if the signal should be added AFTER or BEFORE the dicing a new template
            if (curr_dataset_properties[c.POISSON_TIMING]==c.BEFORE_SIGNAL_ADDITION):
                # If the signal should be added AFTER Poisson dice of background
                # Gets the LHC measurements in the current dataset
                if usesamebkg:
                    curr_dataset_LHC_measurements=LHC_measurements_datasets[0,:,:,:]
                else:
                    curr_dataset_LHC_measurements=LHC_measurements_datasets[curr_dataset_index,:,:,:]
                # Checks in which method the signal should be set
                if (curr_dataset_properties[c.SIGNAL_GEN_METHOD]==c.LIK_METHOD):
                    # If the signal should be set using the likelihood method
                    # Gets the current signal density
                    # print("MEHERE",curr_signal_shape_index,len(arr_signals_shapes_densities))
                    curr_signal_shape_density=np.copy(arr_signals_shapes_densities[curr_signal_shape_index])
                    # Goes over all the measurements
                    for curr_measurements_index in range(number_of_measurements_in_dataset):
                        # Gets the current measurement background
                        curr_measurements_background=np.copy(curr_dataset_LHC_measurements[curr_measurements_index,:,:])
                        # Calculate the current measurement mu
                        fsolve_result=fsolve(Calcq0MinusZ,x0=100,args=(curr_measurements_background,curr_signal_shape_density,curr_num_sigma_of_signal))
                        #print('fsolve_result: ',fsolve_result)
                        curr_measurement_mu=fsolve_result[0]
                        # Saves the mu's
                        mu_magnitudes_of_measurements[curr_dataset_index,curr_measurements_index]=curr_measurement_mu
                        # Sets the current background and signal measurement
                        curr_background_and_signal_measurement=curr_measurements_background+curr_measurement_mu*curr_signal_shape_density
                        # Sets the current measurement
                        LHC_measurements_datasets[curr_dataset_index,curr_measurements_index,:,:]=np.copy(curr_background_and_signal_measurement)
                        # Performs the sanity check
                        minimize_result=minimize(CalcMinusq0,x0=100,args=(curr_background_and_signal_measurement,curr_measurements_background,curr_signal_shape_density))
                        # Saves the sanity check results
                        mu_detected_sanity_check[curr_dataset_index,curr_measurements_index]=minimize_result['x'][0]
                        qzero_detected_sanity_check[curr_dataset_index,curr_measurements_index]=(-1)*minimize_result['fun']
                elif (curr_dataset_properties[c.SIGNAL_GEN_METHOD]==c.SUM_METHOD):
                    # If the signal should be set according to the sum method,i.e. #\sigma*\sqrt{\sum{B in signal region}}
                    # Sums the background contributions in the signal region (in each LHC measurement),i.e sqrt(B)
                    print(arr_X_axis_signal_low_high_indices)
                    print(arr_X_axis_signal_low_high_indices[curr_signal_shape_index])
                    curr_dataset_signal_region_total_background_contributions=np.sqrt(np.sum(curr_dataset_LHC_measurements[:,arr_X_axis_signal_low_high_indices[curr_signal_shape_index][0]:arr_X_axis_signal_low_high_indices[curr_signal_shape_index][1],arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][0]:arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][1]],axis=(1,2)))
                    # Prints the total background contribution to the signal region of the first measurement in the current dataset
                    print('The total sqrt background contribution to the signal region for measurement number ',0 ,' is ',curr_dataset_signal_region_total_background_contributions[0])
                    print('The total sqrt background contribution to the signal region for measurement number ',number_of_measurements_in_dataset-1 ,
                          ' is ',curr_dataset_signal_region_total_background_contributions[number_of_measurements_in_dataset-1])
                    # Multiplies the background contribution in the signal region with the num of sigmas of signal
                    curr_dataset_signal_region_total_background_contributions *= curr_num_sigma_of_signal
                    # Prints the total background contribution to the signal region of the first measurement in the current dataset WITH the number of sigmas
                    print('The total sqrt background contribution to the signal region WITH THE NUMBER OF ',curr_num_sigma_of_signal,' SIGMAS for measurement number ',0,' is ',
                          curr_dataset_signal_region_total_background_contributions[0])
                    print('The total sqrt background contribution to the signal region WITH THE NUMBER OF ',curr_num_sigma_of_signal,' SIGMAS for measurement number ',number_of_measurements_in_dataset-1,' is ',
                          curr_dataset_signal_region_total_background_contributions[number_of_measurements_in_dataset-1])
                    # Saves the mu's
                    mu_magnitudes_of_measurements[curr_dataset_index,:]=np.copy(curr_dataset_signal_region_total_background_contributions)
                    # Calculates the signal contribution to each LHC measurement in the current dataset (i.e multiplies the background contribution with the signal density shape)
                    curr_dataset_signal_region_total_background_contributions=np.transpose(np.expand_dims(curr_dataset_signal_region_total_background_contributions,axis=0))
                    curr_signal_shape_density=np.expand_dims(arr_signals_shapes_densities[curr_signal_shape_index],axis=0)
                    curr_dataset_signal_contributions=np.einsum("ij,jkl",
                                                                  curr_dataset_signal_region_total_background_contributions,
                                                                  curr_signal_shape_density)
                    # Prints the signal final contribution of the first measurement in the current dataset
                    # PlotMatrix(c,curr_dataset_signal_contributions[0,:,:],'The signal only of the measurement number 0')
                    print('The total area of the signal is ' ,np.sum(curr_dataset_signal_contributions[0,:,:]))
                    print('The S/SQRT(B) (in the signal region ONLY) is ',
                          np.sum(curr_dataset_signal_contributions[0,arr_X_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                      arr_X_axis_signal_low_high_indices[curr_signal_shape_index][1],
                                                                      arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                      arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][1]])/
                          np.sqrt(np.sum(LHC_measurements_datasets[curr_dataset_index,0,
                                                                  arr_X_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                  arr_X_axis_signal_low_high_indices[curr_signal_shape_index][1],
                                                                  arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                  arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][1]])))
                    # PlotMatrix(c,curr_dataset_signal_contributions[number_of_measurements_in_dataset-1,:,:],'The signal only of the measurement number '+str(number_of_measurements_in_dataset-1))
                    print('The total area of the signal is ' ,np.sum(curr_dataset_signal_contributions[number_of_measurements_in_dataset-1,:,:]))
                    print('The S/SQRT(B) (in the signal region ONLY) is ',
                          np.sum(curr_dataset_signal_contributions[number_of_measurements_in_dataset-1,arr_X_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                      arr_X_axis_signal_low_high_indices[curr_signal_shape_index][1],
                                                                      arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                      arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][1]])/
                          np.sqrt(np.sum(LHC_measurements_datasets[curr_dataset_index,number_of_measurements_in_dataset-1,
                                                                  arr_X_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                  arr_X_axis_signal_low_high_indices[curr_signal_shape_index][1],
                                                                  arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][0]:
                                                                  arr_Y_axis_signal_low_high_indices[curr_signal_shape_index][1]])))
                    # Adds the signal contributions to the measurements
                    LHC_measurements_datasets[curr_dataset_index,:,:,:]=np.float32(LHC_measurements_datasets[curr_dataset_index,:,:,:]+curr_dataset_signal_contributions)
    return LHC_measurements_datasets,mu_magnitudes_of_measurements,mu_detected_sanity_check,qzero_detected_sanity_check

## ------ Plots some statistics of the generated datasets
def GenerateArrayLabel(arrArray):
    len_of_array=len(arrArray)
    mean_of_array=np.mean(arrArray)
    median_of_array=np.median(arrArray)
    std_of_array=np.std(arrArray)
    mean_error_of_array=std_of_array/np.sqrt(len_of_array)
    # print('len_of_array',len_of_array)
    # print('int(len_of_array)',int(len_of_array))
    # print('mean_of_array',mean_of_array)
    # print('mean_error_of_array',mean_error_of_array)
    # print('std_of_array',std_of_array)
    #hist_label=str(len_of_array)+' entries in total \n(mean: '+str(mean_of_array)+',\n error on mean: '+str(mean_error_of_array)+',\n std: '+str(std_of_array)+')'
    array_label= '{:,} entries in total \n(mean: {:.3e},error on mean: {:.3e},\n median: {:.3e},\n std: {:.3e})'.format(int(len_of_array),mean_of_array,mean_error_of_array,median_of_array,std_of_array)
    return array_label
def PrintMuDistributionsSamePlot(c,arrMuDistributions,
                                   strDistributionsLabels,
                                   main_file_name='PrintMuDistributions',
                                   plot_title='Distribution',
                                   x_label='Scores',
                                   y_label='Number of results',
                                   unit_division_num=1,
                                   marks_values=None,
                                   marks_values_labels=None,
                                   save_plots=False,
                                   download_plots=False):
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    plt.rcParams["figure.figsize"]=20,10 #12,9 # 4,3
    plt.rcParams["figure.titlesize"]='xx-large'
    plt.rcParams['axes.titlesize']=48 #24
    plt.rcParams['axes.labelsize']=36 #18
    plt.rcParams['xtick.labelsize']=30 #18
    plt.rcParams['ytick.labelsize']=30 #18
    plt.rcParams['legend.fontsize']='x-large' # fontsize : int or float or {'xx-small','x-small','small','medium','large','x-large','xx-large'}
    LEGEND_FONT_SIZE_VAL=24
    LINE_WIDTH_VAL=4
    # Gets the minimum and maximum entries of the distributions
    min_distributions_entry=np.min(arrMuDistributions)
    max_distributions_entry=np.max(arrMuDistributions)
    # Sets the binning of the histograms
    bins_edges=np.linspace(np.floor(min_distributions_entry),
                             np.ceil(max_distributions_entry),
                             num=int(np.ceil(max_distributions_entry)-np.floor(min_distributions_entry))*unit_division_num+1)
    # Gets the number of different distributions
    num_of_distributions=len(strDistributionsLabels)
    # Sets an array for the maximum y entries
    arrMaxYEntries=np.zeros((num_of_distributions,))
    # new figure for the current distribution method
    plt.subplot()
    # Plots all the distributions
    for curr_distribution_index in range(num_of_distributions):
        # Gets the current distribution
        if (num_of_distributions==1):
            curr_distribution=np.copy(arrMuDistributions)
        else:
            curr_distribution=np.copy(arrMuDistributions[curr_distribution_index,:])
        # Plots the current sample+its label+statistics (mean,error on the mean,median,std)
        curr_distribution_label=strDistributionsLabels[curr_distribution_index]+'\n['+GenerateArrayLabel(curr_distribution)+']'
        currEntries,_,_=plt.hist(curr_distribution,
                                    bins=bins_edges,
                                    label=curr_distribution_label,
                                    color='C'+str(curr_distribution_index),alpha=0.8,zorder=1)
        # Gets the current maximum y entry
        arrMaxYEntries[curr_distribution_index]=np.max(currEntries)
    # Plots marks values if there are any
    if (marks_values is not None):
        # Gets the number of marks values
        num_of_marks=len(marks_values_labels)
        # Gets the maximum y entry
        max_y_entry=np.max(arrMaxYEntries)
        # Plots all the marks value
        for curr_mark_values_index in range(num_of_marks):
            if (num_of_marks==1):
                curr_mark=marks_values
            else:
                curr_mark=marks_values[curr_mark_values_index]
        curr_mark_label=marks_values_labels[curr_mark_values_index]
        plt.vlines(curr_mark,
                    0,max_y_entry*1.03,linestyles='dashed',linewidth=LINE_WIDTH_VAL,
                    colors= 'C'+str(num_of_distributions+curr_mark_values_index),label=curr_mark_label,
                    zorder=10)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.xlim((np.floor(min_distributions_entry),
              np.ceil(max_distributions_entry)))
    plt.ylabel(y_label)
    plt.title(fill(plot_title,53))
    # Show figure
    plt.tight_layout()
    # Checks if the plot should be saved
    if (save_plots==True):
        # If the plot should be saved
        # Generates the filename of the current plot
        curr_plot_file_name=GenerateFileName(main_file_name+' WITHOUT LEGEND',c.PLOT_FILE_FORMAT,include_date=True)
        # Saves the current plot
        plt.savefig(curr_plot_file_name)
        # Adds the current plot file name to the list of all the files that should be zipped
        arrAllFileNames.append(curr_plot_file_name)
        # Adds legend
        plt.legend(prop={'size': LEGEND_FONT_SIZE_VAL})
        # Generates the filename of the current plot
        curr_plot_file_name=GenerateFileName(main_file_name+' WITH LEGEND',c.PLOT_FILE_FORMAT,include_date=True)
        # Saves the current plot
        plt.savefig(curr_plot_file_name)
        # Adds the current plot file name to the list of all the files that should be zipped
        arrAllFileNames.append(curr_plot_file_name)
        # Creates a string with all the plots filenames
        all_file_name_string=' '.join(arrAllFileNames)
        # Sets the zip file name
        zip_file_name=GenerateFileName(main_file_name+'_All_plots','.zip',include_date=True)
        # # Compresses the plots files
        # !zip $zip_file_name $all_file_name_string
        # # Downloads the compressed plots image files
        # if (download_plots==True):
        #     files.download(zip_file_name)
    # Shows the plot of the histograms of the individual measurement samples test scores
    plt.legend(prop={'size': LEGEND_FONT_SIZE_VAL})
    plt.show()
    return zip_file_name
def get_gendata_stats(c,mu_magnitudes_per_dataset,qzero_sanity_check):
    # Prints the indices of the bins with zero entries
    print('Indices of bins with zero entries: ',np.argwhere(c.LHC_DATASETS==0))
    print('')
    print('===============================')
    print('')
    print('Printing the mu statististics:')
    print('===============================')
    print('')
    for curr_LHC_dataset_index in range(c.NUM_OF_DIFFERENT_LHC_DATASETS):
        curr_mu_distribution=np.copy(mu_magnitudes_per_dataset[curr_LHC_dataset_index,:])
        print('Information about DATASET ',curr_LHC_dataset_index)
        print('TOTAL SAMPLES of ',len(curr_mu_distribution))
        print('The MINIMUM mu is: ',np.min(curr_mu_distribution))
        print('The MAXIMUM mu is: ',np.max(curr_mu_distribution))
        print('The MEAN mu is: ',np.mean(curr_mu_distribution))
        print('The MEDIAN mu is: ',np.median(curr_mu_distribution))
        print('The STD of mu is: ',np.std(curr_mu_distribution))
        print('Number of NEGATIVE elements',np.count_nonzero(curr_mu_distribution<0))
        print('Number of ZERO elements',np.count_nonzero(curr_mu_distribution==0))
        print('Number of POSITIVE elements',np.count_nonzero(curr_mu_distribution>0))
        print('')
        print('------------------------------------------------------')
        print('')
    print('MAX q0: ',np.max(qzero_sanity_check,axis=1))
    print('MIN q0: ',np.min(qzero_sanity_check,axis=1))
    print('MEAN q0: ',np.mean(qzero_sanity_check,axis=1))
    print('STD q0: ',np.std(qzero_sanity_check,axis=1))
    print('')
    print('------------------------------------------------------')
    print('')
def get_gendata_stats_mudef(c,mu_magnitudes_per_dataset):
    ## a. \mu defined
    ALL_MU_FILE_NAME='ALL_DEFINED_MUS'
    str_x_label=r'$ \mu $'+' value'
    str_y_label='Number of results'
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    # Goes over all the datsets
    for curr_ds_index in range(1,c.NUM_OF_DIFFERENT_LHC_DATASETS):
        # Gets the current dataset string
        curr_ds_string=GetLHCDatasetStringByIndex(c,curr_ds_index)
        curr_ds_signal_gen_method=GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)
        # Sets the main file name of the current conditions
        curr_main_file_name=ALL_MU_FILE_NAME+' '+curr_ds_string+' '+curr_ds_signal_gen_method
        curr_figure_title=r'$ \mu $'+' defined of '+curr_ds_string+' dataset'
        curr_distribution=np.copy(mu_magnitudes_per_dataset[curr_ds_index,:])
        curr_label=[curr_ds_string+' results',]
        curr_zip_file_name=PrintMuDistributionsSamePlot(c,curr_distribution,
                                                        curr_label,
                                                        main_file_name=curr_main_file_name,
                                                        plot_title=curr_figure_title,
                                                        x_label=str_x_label,
                                                        y_label=str_y_label,
                                                        marks_values=None,
                                                        marks_values_labels=None,
                                                        save_plots=True,
                                                        download_plots=False)
        # Adds the current file name to the list of all the files that should be zipped and downloaded
        arrAllFileNames.append(curr_zip_file_name)
    # Creates a string with all the output filenames
    all_file_name_string=' '.join(arrAllFileNames)
    # # Sets the zip file name
    # zip_file_name=GenerateFileName(ALL_MU_FILE_NAME,'.zip',include_date=True)
    # # Compresses the plots files
    # !zip $zip_file_name $all_file_name_string
    # # Downloads the zip file
    # files.download(zip_file_name)
def get_gendata_stats_q0def_vs_q0dec(c,qzero_sanity_check):
    ## b. q0 defined vs q0 detected (q0=z^{2})
    Q0_DEFINED_VS_DETECTED_FILE_NAME='Q0_DEFINED_VS_DETECTED'
    str_x_label=r'$ q_{0} $'+' score'
    str_y_label='Number of results'
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    # Goes over all the datsets
    for curr_ds_index in range(1,c.NUM_OF_DIFFERENT_LHC_DATASETS):
        # Gets the current dataset string
        curr_ds_string=GetLHCDatasetStringByIndex(c,curr_ds_index)
        curr_ds_signal_gen_method=GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)
        curr_ds_signal_strength=0
        if (c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.LHC_DATASET_TYPE]==c.BACKGROUND_SIGNAL_DATASET):
            curr_ds_signal_strength=int(c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.NUM_OF_SIGMA_OF_SIGNAL])
        # Sets the main file name of the current conditions
        curr_main_file_name=Q0_DEFINED_VS_DETECTED_FILE_NAME+' '+curr_ds_string+' '+curr_ds_signal_gen_method
        curr_figure_title=r'$ q_{0} $'+' defined vs detected of '+curr_ds_string+' dataset'
        curr_distribution=np.copy(qzero_sanity_check[curr_ds_index,:])
        curr_label=[r'$ q_{0} $'+' detected',]
        mark_value=curr_ds_signal_strength**2
        mark_label=[r'$ q_{0} $'+' defined',]
        curr_zip_file_name=PrintMuDistributionsSamePlot(c,curr_distribution,
                                                          curr_label,
                                                          main_file_name=curr_main_file_name,
                                                          plot_title=curr_figure_title,
                                                          x_label=str_x_label,
                                                          y_label=str_y_label,
                                                          marks_values=mark_value,
                                                          marks_values_labels=mark_label,
                                                          unit_division_num=100,
                                                          save_plots=True,
                                                          download_plots=False)
        # Adds the current file name to the list of all the files that should be zipped and downloaded
        arrAllFileNames.append(curr_zip_file_name)
    # Creates a string with all the output filenames
    all_file_name_string=' '.join(arrAllFileNames)
    # # Sets the zip file name
    # zip_file_name=GenerateFileName(Q0_DEFINED_VS_DETECTED_FILE_NAME,'.zip',include_date=True)
    # # Compresses the plots files
    # !zip $zip_file_name $all_file_name_string
    # # Downloads the zip file
    # files.download(zip_file_name)
def get_gendata_stats_mudef_vs_mudec(c,mu_magnitudes_per_dataset,mu_sanity_check):    
    ## c. \mu defined vs \mu detected
    MU_DEFINED_VS_DETECTED_FILE_NAME='MU_DEFINED_VS_DETECTED'
    str_x_label=r'$ \mu $'+' value'
    str_y_label='Number of results'
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    # Goes over all the datsets
    for curr_ds_index in range(1,c.NUM_OF_DIFFERENT_LHC_DATASETS):
        # Gets the current dataset string
        curr_ds_string=GetLHCDatasetStringByIndex(c,curr_ds_index)
        curr_ds_signal_gen_method=GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)
        # Sets the main file name of the current conditions
        curr_main_file_name=MU_DEFINED_VS_DETECTED_FILE_NAME+' '+curr_ds_string+' '+curr_ds_signal_gen_method
        curr_figure_title=r'$ \mu $'+' defined vs detected of '+curr_ds_string+' dataset'
        curr_distribution=np.array([np.copy(mu_magnitudes_per_dataset[curr_ds_index,:]),np.copy(mu_sanity_check[curr_ds_index,:])])
        curr_label=[r'$ \mu $'+' defined',r'$ \mu $'+' detected']
        curr_zip_file_name=PrintMuDistributionsSamePlot(c,curr_distribution,
                                                          curr_label,
                                                          main_file_name=curr_main_file_name,
                                                          plot_title=curr_figure_title,
                                                          x_label=str_x_label,
                                                          y_label=str_y_label,
                                                          marks_values=None,
                                                          marks_values_labels=None,
                                                          save_plots=True,
                                                          download_plots=False)
        # Adds the current file name to the list of all the files that should be zipped and downloaded
        arrAllFileNames.append(curr_zip_file_name)
    # Creates a string with all the output filenames
    all_file_name_string=' '.join(arrAllFileNames)
    # # Sets the zip file name
    # zip_file_name=GenerateFileName(MU_DEFINED_VS_DETECTED_FILE_NAME,'.zip',include_date=True)
    # # Compresses the plots files
    # !zip $zip_file_name $all_file_name_string
    # # Downloads the zip file
    # files.download(zip_file_name)
def get_gendata_stats_mudefsum_vs_mudeflik(c,mu_magnitudes_per_dataset):
    ## d. \mu defined sum vs \mu defined likelihood
    MU_SUM_VS_LIKELIHOOD_FILE_NAME='MU_SUM_VS_LIKELIHOOD'
    str_x_label=r'$ \mu $'+' value'
    str_y_label='Number of results'
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    # Goes over all the datasets
    for curr_ds_index in range(1,c.NUM_OF_DIFFERENT_LHC_DATASETS):
        # Gets the current dataset string
        curr_ds_string=GetLHCDatasetStringByIndex(c,curr_ds_index)
        curr_ds_signal_gen_method=GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)
        curr_pair_ds_index=0
        # Searches for the same dataset with the likelihood method
        for curr_second_ds_index in range(2,c.NUM_OF_DIFFERENT_LHC_DATASETS):
            if ((c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.SIGNAL_DENSITY_SHAPE_INDEX]==c.LHC_DATASETS_PROPERTIES[curr_second_ds_index][c.SIGNAL_DENSITY_SHAPE_INDEX]) and
                (c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.NUM_OF_SIGMA_OF_SIGNAL]==c.LHC_DATASETS_PROPERTIES[curr_second_ds_index][c.NUM_OF_SIGMA_OF_SIGNAL]) and
                (c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.POISSON_TIMING]==c.LHC_DATASETS_PROPERTIES[curr_second_ds_index][c.POISSON_TIMING]) and
                (c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.LHC_DATASET_COMMENT]==c.LHC_DATASETS_PROPERTIES[curr_second_ds_index][c.LHC_DATASET_COMMENT]) and
                (c.LHC_DATASETS_PROPERTIES[curr_ds_index][c.SIGNAL_GEN_METHOD] != c.LHC_DATASETS_PROPERTIES[curr_second_ds_index][c.SIGNAL_GEN_METHOD])):
                curr_pair_ds_index=curr_second_ds_index
                break
        curr_pair_ds_signal_gen_method=GetLHCDatasetSignalGenMethodByIndex(c,curr_pair_ds_index)
        # Sets the main file name of the current conditions
        curr_main_file_name=MU_SUM_VS_LIKELIHOOD_FILE_NAME+' '+curr_ds_string
        curr_figure_title=r'$ \mu $'+' defined of SUM vs. LIKELIHOOD based signal definition method of '+curr_ds_string+' dataset'
        curr_distribution=np.array([np.copy(mu_magnitudes_per_dataset[curr_ds_index,:]),np.copy(mu_magnitudes_per_dataset[curr_pair_ds_index,:])])
        curr_label=[curr_ds_signal_gen_method,curr_pair_ds_signal_gen_method]
        curr_zip_file_name=PrintMuDistributionsSamePlot(c,curr_distribution,
                                                          curr_label,
                                                          main_file_name=curr_main_file_name,
                                                          plot_title=curr_figure_title,
                                                          x_label=str_x_label,
                                                          y_label=str_y_label,
                                                          marks_values=None,
                                                          marks_values_labels=None,
                                                          save_plots=True,
                                                          download_plots=False)
        # Adds the current file name to the list of all the files that should be zipped and downloaded
        arrAllFileNames.append(curr_zip_file_name)
    # Creates a string with all the output filenames
    all_file_name_string=' '.join(arrAllFileNames)
    # # Sets the zip file name
    # zip_file_name=GenerateFileName(MU_SUM_VS_LIKELIHOOD_FILE_NAME,'.zip',include_date=True)
    # # Compresses the plots files
    # !zip $zip_file_name $all_file_name_string
    # # Downloads the zip file
    # files.download(zip_file_name)
def PlotAllSignalsNBackgroundNDatasets(c,save_plots=True,download_plots=True,background_template_title='Background template',b_short_signal_title=False):
    # Initializes the array that saves all the files that will be later zipped
    arrAllFileNames=[]
    # Plots the Background template
    arrAllFileNames.append(PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE,strTitle=background_template_title,log_scale=False,save_output=save_plots,main_file_name='Background_template_regular_scale',x_axis=c.X_AXIS,y_axis=c.Y_AXIS))
    arrAllFileNames.append(PlotMatrix(c,c.BACKGROUND_ONLY_TEMPLATE,strTitle=background_template_title,log_scale=True,save_output=save_plots,main_file_name='Background_template_log_scale',x_axis=c.X_AXIS,y_axis=c.Y_AXIS))
    # Plots the signals
    for curr_signal_index in range(c.NUM_OF_DIFFERENT_SIGNAL_DENSITY_SHAPES):
        arrAllFileNames.append(PlotMatrix(c,c.SIGNALS_SHAPES_DENSITIES[curr_signal_index],
                                            strTitle=GetSignalStringByIndex(c,curr_signal_index,b_short_signal_title),
                                            log_scale=False,save_output=save_plots,
                                            main_file_name=GetSignalStringByIndex(c,curr_signal_index),x_axis=c.X_AXIS,y_axis=c.Y_AXIS))
    # Plots example from datasets
    for curr_ds_index in range(c.NUM_OF_DIFFERENT_LHC_DATASETS):
        arrAllFileNames.append(PlotMatrix(c,c.LHC_DATASETS[curr_ds_index,0,:,:],
                                            strTitle='A measurement in the '+GetLHCDatasetStringByIndex(c,curr_ds_index)+' dataset',
                                            log_scale=False,save_output=save_plots,
                                            main_file_name='A measurement in the '+GetLHCDatasetStringByIndex(c,curr_ds_index)+' '+GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)+' dataset '+str(curr_ds_index),x_axis=c.X_AXIS,y_axis=c.Y_AXIS))
        arrAllFileNames.append(PlotMatrix(c,c.LHC_DATASETS[curr_ds_index,0,:,:],
                                            strTitle='A measurement in the '+GetLHCDatasetStringByIndex(c,curr_ds_index)+' dataset',
                                            log_scale=True,save_output=save_plots,
                                            main_file_name='A measurement in the '+GetLHCDatasetStringByIndex(c,curr_ds_index)+' '+ GetLHCDatasetSignalGenMethodByIndex(c,curr_ds_index)+' dataset (log scale)'+str(curr_ds_index),x_axis=c.X_AXIS,y_axis=c.Y_AXIS))
    zip_file_name=''
    # Checks if the plot should be saved
    if (save_plots==True):
        # If the plot should be saved
        # Creates a string with all the plots filenames
        all_file_name_string=' '.join(arrAllFileNames)
        # # Sets the zip file name
        # zip_file_name=GenerateFileName('All_Signals_n_Background_n_Datasets_plots','.zip',include_date=True)
        # # Compresses the plots files
        # !zip $zip_file_name $all_file_name_string
        # # Downloads the compressed plots image files
        # if (download_plots==True):
        #     files.download(zip_file_name)
    return zip_file_name

## ------ General Search - calc test-statistic dists
## Here we implement the functions that are used to retrieve the original and approximated distributions of a given matrix in a nice packed way (i.e, numpy array)
def GetDistributionStringByIndex(c,distribution_index):
    dsitribution_string = 'Unknown distribution index'
    if (distribution_index == c.ORIG_DIST_INDEX):
        distribution_string = 'Original distribution'
    elif (distribution_index == c.POS_APPROX_INDEX):
        distribution_string = r'$ Pois \left( X \right) $' + ' approximation'
    elif (distribution_index == c.POIS_GAUSS_APPROX_INDEX):
        distribution_string = r'$ Pois \left( Gauss \left( X , \sqrt{X} \right) \right) $' + ' approximation'
    return distribution_string
def GenerateApproxMatrixDistribution(c,arrMatrixToApprox, num_of_approximations, approx_method_index):
    # Duplicates the template matrix, i.e the matrix to approximate
    # This is also the approximation of the Poisson parameter for the Pois(X) approximation method
    template_matrix = np.repeat(arrMatrixToApprox[np.newaxis, :, :],
                                num_of_approximations, axis = 0)
    # Checks if the approximation method is Pois(Gauss)
    if (approx_method_index == c.POIS_GAUSS_APPROX_INDEX):
        # If it is the POIS(GAUSS(X, sqrt(X)))
        # estimates the poisson value of the current bin (one poisson parameters estimation for each sample)
        # (see https://indico.cern.ch/category/6015/attachments/192/629/Statistics_introduction.pdf)
        template_matrix = np.random.normal(loc = template_matrix, scale = np.sqrt(template_matrix))
        # set all the negative poiss estimations to be zero (poiss value can only be positive)
        template_matrix[template_matrix < 0] = 0
    # Draws matrices from the approximated distribution
    arrApproxDistribMatrices = np.random.poisson(template_matrix)
    # # Initializes the array of all the samples
    # arrApproxDistribMatrices = np.zeros((num_of_approximations, arrMatrixToApprox.shape[0], arrMatrixToApprox.shape[1]), dtype = arrMatrixToApprox.dtype)
    # # goes over all the bins in the input matrix
    # for curr_dim_1_index in range(arrApproxDistribMatrices.shape[1]):
    #   for curr_dim_2_index in range(arrApproxDistribMatrices.shape[2]):
    #     # gets the number of entries in the current bin
    #     curr_bin_entry = arrMatrixToApprox[curr_dim_1_index, curr_dim_2_index]
    #     # Checks which approximation method should be used
    #     if (approx_method_index == c.POS_APPROX_INDEX):
    #       # If it is just the POIS(X) approximation
    #       # estimates the poisson value of the current bin
    #       curr_bin_poiss_est = curr_bin_entry * np.ones(num_of_approximations)
    #     elif (approx_method_index == c.POIS_GAUSS_APPROX_INDEX):
    #       # If it is the POIS(GAUSS(X, sqrt(X)))
    #       # estimates the poisson value of the current bin (one poisson parameters estimation for each sample)
    #       # (see https://indico.cern.ch/category/6015/attachments/192/629/Statistics_introduction.pdf)
    #       curr_bin_poiss_est = np.random.normal(loc = curr_bin_entry, scale = np.sqrt(curr_bin_entry), size = num_of_approximations)
    #       # set all the negative poiss estimations to be zero (poiss value can only be positive)
    #       curr_bin_poiss_est[curr_bin_poiss_est < 0] = 0
    #     # randomizes the entry for each sample
    #     curr_bin_approx_entries = np.random.poisson(curr_bin_poiss_est)
    #     # sets the value of the current bin in all the samples
    #     arrApproxDistribMatrices[:, curr_dim_1_index, curr_dim_2_index] = curr_bin_approx_entries
    return arrApproxDistribMatrices
def GetDistributionsOfMatrix(c,arrMatrixToApprox, original_distribution):
    # Gets the total number of matrices in the original distribution
    num_of_approximations = original_distribution.shape[0]
    # Initializes the variable the keeps all the distributions (original, approximation 1, approximation 2)
    distribution_of_input_matrix = np.zeros((c.NUM_OF_DISTRIBUTION_METHODS, num_of_approximations, arrMatrixToApprox.shape[0], arrMatrixToApprox.shape[1]),
                                            dtype = arrMatrixToApprox.dtype)
    # Gets the original distribution
    distribution_of_input_matrix[c.ORIG_DIST_INDEX, :, :, :] = np.copy(original_distribution)
    # Gets the POIS(X) distribution approximation for the current matrix
    distribution_of_input_matrix[c.POS_APPROX_INDEX, :, :, :] = GenerateApproxMatrixDistribution(c,arrMatrixToApprox, num_of_approximations, c.POS_APPROX_INDEX)
    # Gets the POIS(GAUSS(X, SQRT(X))) distribution approximation for the current matrix
    distribution_of_input_matrix[c.POIS_GAUSS_APPROX_INDEX, :, :, :] = GenerateApproxMatrixDistribution(c,arrMatrixToApprox, num_of_approximations, c.POIS_GAUSS_APPROX_INDEX)
    return distribution_of_input_matrix
##   Then, we define the functions that are used when we want to arrange all the test-statistic results in histograms and perform Gaussian fits for these distributions
def CreateBinsForHistogram(arrEntries, bin_division = 1):
    # Gets the maximum and minimum values of the edges
    max_bin_ceil_value = np.max(arrEntries)
    min_bin_floor_value = np.min(arrEntries)
    # Calculates the number of bins needed
    num_of_bins = bin_division #bin_division * (max_bin_ceil_value - min_bin_floor_value) + 1
    arrBins = np.linspace(min_bin_floor_value, max_bin_ceil_value, num_of_bins)
    arrBinsCenters = np.array([0.5 * (arrBins[curr_index] + arrBins[curr_index + 1]) for curr_index in range(len(arrBins) - 1)])
    return arrBins, arrBinsCenters
def MyGaussian(x, mean, sigma, a):
    return a * np.exp(-((x - mean)**2 / (2 * sigma**2))) # 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mean)**2 / (2 * sigma**2)))
def PerformGaussianFit(arrXValues, arrYValues):
    proposed_a = np.max(arrYValues)
    proposed_mean = np.sum(arrXValues * arrYValues) / np.sum(arrYValues)
    propsed_std = np.sqrt(np.sum(arrYValues * (arrXValues - proposed_mean)**2) / np.sum(arrYValues))
    #print('proposed (a, mean, std): ', proposed_a, proposed_mean, propsed_std)
    popt, pcov = curve_fit(MyGaussian, xdata = arrXValues, ydata = arrYValues, p0 = [proposed_mean, propsed_std, proposed_a])
    arrResiduals = arrYValues - MyGaussian(arrXValues, *popt)
    return popt, pcov, arrResiduals
def GetStatisticalHypothesisTestStringByIndex(c,stat_hypoth_test_index):
    stat_hypoth_test_string = 'Unknown statistical hypothesis test index'
    if (stat_hypoth_test_index == c.CDF_INDEX):
        stat_hypoth_test_string = 'CDF'
    elif (stat_hypoth_test_index == c.P_VALUE_INDEX):
        stat_hypoth_test_string = r'$p$' + '-value'
    elif (stat_hypoth_test_index == c.Z_INDEX):
        stat_hypoth_test_string = 'Z'
    return stat_hypoth_test_string
