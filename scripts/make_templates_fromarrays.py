## general
import sys
import csv
## science
import numpy as np
## mystuff
from myfunctions import *
sys.path.insert(0,"/srv01/agrp/mattiasb/scripts/p3")
from utils import *
from np_utils import *
from plots3 import *

def main():
    path='/storage/agrp/mattiasb/Public/PhenoStudy_share/arrays/'
    # data_path_bkg=path+'me_run5.csv'
    data_path_bkg=path+'table_me_run0.csv'
    data_path_sig_ggh=path+'allsignal/table_ggH_taumu.csv'
    data_path_sig_vbf=path+'allsignal/table_vbfH_taumu.csv'
    ## Get arrays
    columns=["Mcoll","Lep0Pt"]
    cuts={'Mcoll':[30e3,170e3],'Lep0Pt':[10e3,150e3]}
    sig_ggh_data=get_array(columns,data_path_sig_ggh,cuts,isSig=1)
    sig_vbf_data=get_array(columns,data_path_sig_vbf,cuts,isSig=1)
    bkg_data=get_array(columns,data_path_bkg,cuts,isSig=0)
    ## Get ratio of vbfH vs ggH signal events
    lumi_ggh=390.032
    lumi_vbf=6707.808
    N_ggh=sig_ggh_data.shape[0]
    N_vbf=sig_vbf_data.shape[0]
    sig_ratio=(N_vbf/lumi_vbf)/(N_ggh/lumi_ggh)
    print("sig ratio vbf/ggh",sig_ratio)
    ## Matrix defs
    ## Lep0Pt vs Mcoll; 28*28 5GeV bins; Mcoll_min,Lep0Pt_min=30,10GeV; increase_entries is added to each bin in bkg matrix
    binsize=28*5e3/nbins
    print("Binsize",binsize)
    xaxis=[30e3+k*binsize for k in range(nbins+1)]
    yaxis=[10e3+k*binsize for k in range(nbins+1)]
    ## Get signal template
    n_sig_ggh_tot=N_ggh
    n_sig_vbf_tot=int(round(n_sig_ggh_tot*sig_ratio)) ## keep correct ratio vbf/ggh
    sig_ggh_tot=get_matrix(sig_ggh_data,xaxis,yaxis)
    sig_vbf_tot=get_matrix(sig_vbf_data[:n_sig_vbf_tot,:],xaxis,yaxis)
    sig_tot=sig_ggh_tot+sig_vbf_tot
    sig_tpl=sig_tot/np.sum(sig_tot)
    # MatrixPlot(sig_tpl,title="Signal template").showplot()
    ## Get bkg matrix
    n_bkg=bkg_data.shape[0]
    bkg=get_matrix(bkg_data,xaxis,yaxis)#,add_entries=increase_entries)
    # bkg=bkg+1e-6*(bkg==0)
    ## Save 
    np.save("bkg_mue.npy",bkg)
    np.save("sig_hlfv-mutau.npy",sig_tpl)

def get_array(columns,data_path,cuts,isSig):
    ## headers=['Lep0Pt', 'Lep0Eta', 'Lep0Phi', 'Lep1Pt', 'Lep1Eta', 'Lep1Phi', 'MET', 'METPhi', 'MLL', 'MLep0MET', 'MLep1MET', 'Mcoll', 'isSig']
    return get_array_fromcsv(columns,data_path,cuts,isSig)

if __name__=="__main__":
    main()
