## general
import sys
import csv
import timeit
## science
import numpy as np
import matplotlib.pyplot as plt
## mystuff
sys.path.insert(0,"utils")
from utils3 import *
from plots3 import *
from myfunctions import *

def main():
    # data_path_bkg="../SignalChallenge/me_run5.csv"
    data_path_bkg="../SignalChallenge/me_run0.csv"
    data_path_sig_ggh="../SignalChallenge/allsignal/table_ggH_taumu.csv"
    data_path_sig_vbf="../SignalChallenge/allsignal/table_vbfH_taumu.csv"
    ## Matrix defs
    ## Lep0Pt vs Mcoll; 28*28 5GeV bins; Mcoll_min,Lep0Pt_min=30,10GeV; increase_entries is added to each bin in bkg matrix
    columns=["Mcoll","Lep0Pt"]
    xaxis=[30e3+k*5e3 for k in range(29)]
    yaxis=[10e3+k*5e3 for k in range(29)]
    increase_entries=25
    increase_entries=0
    ## Get ratio of vbfH vs ggH signal events
    lumi_ggh=390.032
    lumi_vbf=6707.808
    print("generated luminosities for ggh / vbfh in ifb:",lumi_ggh,"/",lumi_vbf)
    sig_ggh_data=get_array(columns,data_path_sig_ggh,isSig=1)
    sig_vbf_data=get_array(columns,data_path_sig_vbf,isSig=1)
    print("total number of ggh /vbfh events:",sig_ggh_data.shape[0],"/",sig_vbf_data.shape[0])
    n_ggh=sum([(30e3<l[0]<170e3 and 10e3<l[1]<150e3) for l in sig_ggh_data])
    n_vbf=sum([(30e3<l[0]<170e3 and 10e3<l[1]<150e3) for l in sig_vbf_data])
    print("number of ggh /vbfh events in Matrix range:",n_ggh,"/",n_vbf)
    sig_ratio=(n_vbf/lumi_vbf)/(n_ggh/lumi_ggh)
    print("ratio of vbfH vs ggH signal events:",sig_ratio)
    print()
    ## Get signal matrix template (normalized)
    n_sig_ggh_tot=n_ggh
    n_sig_vbf_tot=int(round(n_sig_ggh_tot*sig_ratio)) ## keep correct ratio vbf/ggh
    sig_ggh_tot=get_matrix(sig_ggh_data,xaxis,yaxis)
    sig_vbf_tot=get_matrix(sig_vbf_data[:n_sig_vbf_tot,:],xaxis,yaxis)
    sig_tot=sig_ggh_tot+sig_vbf_tot
    sig_tpl=sig_tot/np.sum(sig_tot)
    # MatrixPlot(sig_tpl,title="Signal template").showplot()
    ## Get background-only matrices
    bkg_data=get_array(columns,data_path_bkg,isSig=0)
    print("total number of background events",bkg_data.shape[0])
    bkg=get_matrix(bkg_data,xaxis,yaxis,add_entries=increase_entries)
    # MatrixPlot(bkg,title="Background events").showplot()
    n_bkg=np.sum(bkg)
    print("number of bkg events in Matrix range:",n_bkg)

    # ## Attempt to build signal from mu_inj - measured signif differs from desired signif due to difference between signal and mu_inj*sig_tpl
    # for zin in [3,5]:
    #     print("Input significance:",zin)
    #     ## Get signal-strength (= number of signal events) for desired significance; number of events per sig
    #     test_tmp,sig_tmp,mu_inj=add_sig2bkg(bkg,sig_tpl,zin) ## test_tmp=bkg+mu_inj*sig_tpl; sig_tmp=mu_inj*sig_tpl
    #     print("  injected mu",mu_inj)
    #     print("  signal fraction:",mu_inj*100/n_bkg)
    #     ## Get sig and bkg+sig matrices based on real events
    #     n_sig=int(round(mu_inj))
    #     n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
    #     n_sig_vbf=n_sig-n_sig_ggh
    #     if n_sig_ggh>n_sig_ggh_tot or n_sig_vbf>n_sig_vbf_tot:
    #         print(Problem)
    #     print("  number of signal events, ggH events, vbfH events",n_sig,n_sig_ggh,n_sig_vbf)
    #     sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
    #     sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
    #     sig=sig_ggh+sig_vbf
    #     test=bkg+sig
    #     # MatrixPlot(sig_tmp-sig).showplot()
    #     ## Check significance obtained with this signal
    #     q0s,pvals,zs,muhats=perform_lik_test([bkg],[test],sig_tpl)#sig/np.sum(sig))
    #     print("  siginificance found with this signal:",zs[0])
    # print()
    
    ## Loop on signal fractions
    rows=[["sig-fraction","n_sig","n_ggh","n_vbf","signif"]]
    chosen_sigfs=[0.1605,0.2725]#0.1920,0.1150] # for addentries 0
    sigfs=sorted([0.1*k for k in range(5)]+[0.5+0.5*k for k in range(4)]+chosen_sigfs)
    sigfs=[.2725]
    startT=timeit.default_timer()
    for sigf in sigfs:
        print(sigf)
        ## Get desired number of events per signal
        n_sig=int(round(sigf*n_bkg/100))
        n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
        n_sig_vbf=n_sig-n_sig_ggh
        if n_sig_ggh>n_sig_ggh_tot or n_sig_vbf>n_sig_vbf_tot:
            print(Problem)
        ## Get sig and bkg+sig matrix
        print('bug number of events less than measured')
        input()
        sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
        sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
        sig=sig_ggh+sig_vbf
        test=bkg+sig
        # MatrixPlot(test-bkg).showplot()
        ## Perform Likelihood test
        q0s,pvals,zs,muhats,b_out=perform_lik_test_new([bkg],[test],sig_tpl)#sig/np.sum(sig))
        if zs==[]:
            rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"fail"])
        else:
            rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"%.4f"%zs[0]])
        print(rows[-1])
        # print("Done (%.1fs)"%(timeit.default_timer()-startT))
    print_table(rows)

    # ## Loop on signal fractions - detail around 3/5 sigma
    # rows=[["sig-fraction","n_sig","n_ggh","n_vbf","signif"]]
    # # for sigf in list(np.linspace(0.1,0.3,21)):
    # # for sigf in list(np.linspace(0.11,0.12,21))+list(np.linspace(0.19,0.20,21)):
    # # for sigf in list(np.linspace(0.16,0.17,21))+list(np.linspace(0.27,0.28,21)):
    # for sigf in list(np.linspace(0.27,0.28,21)):
    #     print(sigf)
    #     ## Get desired number of events per signal
    #     n_sig=int(round(sigf*n_bkg/100))
    #     n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
    #     n_sig_vbf=n_sig-n_sig_ggh
    #     if n_sig_ggh>n_sig_ggh_tot or n_sig_vbf>n_sig_vbf_tot:
    #         print(Problem)
    #     ## Get sig and bkg+sig matrix
    #     sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
    #     sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
    #     sig=sig_ggh+sig_vbf
    #     test=bkg+sig
    #     # MatrixPlot(test-bkg).showplot()
    #     ## Perform Likelihood test
    #     q0s,pvals,zs,muhats,b_out=perform_lik_test_new([bkg],[test],sig_tpl)#sig/np.sum(sig))
    #     rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"%.4f"%zs[0]])
    #     print(rows[-1])
    # print_table(rows)

    # ## Check s/sqrt(b) with bins including >1% signal
    # sigf=0.1150
    # n_sig=int(round(sigf*n_bkg/100))
    # n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
    # n_sig_vbf=n_sig-n_sig_ggh
    # sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
    # sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
    # sig=sig_ggh+sig_vbf
    # s=0
    # b=0
    # for i,row in enumerate(sig_tpl):
    #     for j,cell in enumerate(row):
    #         if sig_tpl[i,j]>=0.02:
    #             print(i,j,sig_tpl[i,j])
    #             s+=sig[i,j]
    #             b+=bkg[i,j]
    # print(s,b,s/np.sqrt(b))
    
def get_matrix(data,xaxis,yaxis,add_entries=0):
    M,_,_=np.histogram2d(data[:,0],data[:,1],bins=(xaxis,yaxis))
    # MatrixPlot(H).showplot()
    return M+add_entries
    
def get_array(columns,data_path,isSig):
    ## headers=['Lep0Pt', 'Lep0Eta', 'Lep0Phi', 'Lep1Pt', 'Lep1Eta', 'Lep1Phi', 'MET', 'METPhi', 'MLL', 'MLep0MET', 'MLep1MET', 'Mcoll', 'isSig']
    headers,data=csv2np(data_path)
    n=0
    for l in data:
        # n+=(l[-1]==0 and 10e3<l[0]<150e3 and 30e3<l[11]<170e3)
        n+=(l[-1]==0 and l[11]<170e3)
    col_idxs=[headers.index(name) for name in columns]
    ## select only rows where last column==isSig, keep only columns specified
    data=data[np.ix_(data[:,-1]==isSig,col_idxs)]
    return data

def csv2np(fpath):
    with open(fpath,'r') as f:
        reader=csv.reader(f,delimiter=',')
        headers=next(reader)
        data=np.array(list(reader)).astype(float)
    return headers,data

if __name__=="__main__":
    main()
