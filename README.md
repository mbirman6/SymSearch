* Run 'python3 mymain.py'
  * Generates datasets = 'sets' of N 28*28 matrices
  * Change N in utils/myconfig.py - NUMBER_OF_LHC_MEASUREMENTS_IN_DATASET (set to 100 for now)
  * Change number and properties of 'sets' in utils/myconfig.py - LHC_DATASETS_PROPERTIES 
    * Each set can be {bkg only, bkg + signal}
    * Different signals can be introduced
    * (for now 2 sets, 1 bkg only, 1 bkg + gaussian signal)
  * Save generated datasets in npy format
    * shape = (number of sets, N, 28, 28)
