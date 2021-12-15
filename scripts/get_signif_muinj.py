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
    path='../SignalChallenge/'
    # data_path_bkg=path+'me_run5.csv'
    data_path_bkg=path+'me_run0.csv'
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
    print()
    # trows=[["nbins","nparams","signif","time","ncalls"]]    
    trows=[["sonb_th","sigf","nparams","signif","time","ncalls"]]    
    for nbins in [28]:#range(1,29):#[1,2,4,7,14,28,56]:
        print("Nbins:",nbins)
        ## Matrix defs
        ## Lep0Pt vs Mcoll; 28*28 5GeV bins; Mcoll_min,Lep0Pt_min=30,10GeV; increase_entries is added to each bin in bkg matrix
        binsize=28*5e3/nbins
        print("Binsize",binsize)
        xaxis=[30e3+k*binsize for k in range(nbins+1)]
        yaxis=[10e3+k*binsize for k in range(nbins+1)]
        increase_entries=25
        increase_entries=0
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
        bkg=get_matrix(bkg_data,xaxis,yaxis,add_entries=increase_entries)
        bkg=bkg+1e-6*(bkg==0)
        
        # MatrixPlot(bkg,title="Background events").showplot()

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
        
    #     ## Loop on signal fractions
    #     rows=[["sig-fraction","n_sig","n_ggh","n_vbf","signif"]]
    #     chosen_sigfs=[0.1565,0.2635]
    #     sigfs=sorted([0.1*k for k in range(5)]+[0.5+0.5*k for k in range(4)]+chosen_sigfs)
    #     sigfs=[0.25]
    #     startT=timeit.default_timer()
    #     thisT=startT
    #     print("Starting tests")
    #     for sigf in sigfs:
    #         ## Make test matrix (add signal events to bkg)
    #         n_sig=int(round(sigf*n_bkg/100))
    #         n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
    #         n_sig_vbf=n_sig-n_sig_ggh
    #         sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
    #         sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
    #         sig=sig_ggh+sig_vbf
    #         # test=draw_fromtpl(bkg,1)[0,:]+sig
    #         test=bkg+sig
    #         print("sigf nsig nggh nvbf",sigf,n_sig,n_sig_ggh,n_sig_vbf)
    #         # MatrixPlot(test,title="Bkg+sig events").showplot()
    #         ## Perfgorm Likelihood test
    #         q0s,pvals,zs,muhats,b_out,ncalls=perform_lik_test_new([bkg],[test],sig_tpl)#sig/np.sum(sig))
    #         if zs==[]:
    #             rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"fail"])
    #         else:
    #             rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"%.4f"%zs[0]])
    #         print(rows[-1])
    #         newT=timeit.default_timer()
    #         print("Done (%.1fs, %.1fs)"%(newT-thisT,newT-startT))
    #         trows.append([nbins,nbins**2+1,"%.4f"%zs[0],"%.1f"%(newT-thisT),ncalls])
    #         thisT=newT
    #     # print_table(rows)
    # print()
    # print_table(trows)

        ## Loop on fractions of signal bins
        rows=[["sig-fraction","n_sig","n_ggh","n_vbf","signif"]]
        print("Starting tests")
        sigf=0.2635
        # zsigs=['0','5e-6','1e-5','5e-5','1e-4','5e-4','1e-3']
        sonb=sig_tpl/bkg**0.5
        startT=timeit.default_timer()
        thisT=startT
        zsigs=[-1,-1]+list(np.linspace(0,1e-4,21))
        # zsigs=[-1,-1]+list(np.linspace(0,1e-3,21))
        for zsig in zsigs:
            selectbins=sonb>float(zsig)
            # selectbins=sig_tpl>float(zsig)
            ## Make test matrix (add signal events to bkg)
            n_sig=int(round(sigf*n_bkg/100))
            n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
            n_sig_vbf=n_sig-n_sig_ggh
            sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
            sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
            sig=sig_ggh+sig_vbf
            # test=draw_fromtpl(bkg,1)[0,:]+sig
            test=bkg+sig
            print("sigf nsig nggh nvbf",sigf,n_sig,n_sig_ggh,n_sig_vbf)
            # MatrixPlot(test,title="Bkg+sig events").showplot()
            ## Perfgorm Likelihood test
            q0s,pvals,zs,muhats,b_out,ncalls=perform_lik_test_new([bkg],[test],sig_tpl,selectbins)
            if zs==[]:
                rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"fail"])
            else:
                rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"%.4f"%zs[0]])
            print(rows[-1])
            newT=timeit.default_timer()
            print("Done (%.1fs, %.1fs)"%(newT-thisT,newT-startT))
            # trows.append([nbins,nbins**2+1,"%.4f"%zs[0],"%.1f"%(newT-thisT),ncalls])
            siglessbins=sig_tpl[selectbins]
            sigfraction=np.sum(siglessbins)
            # trows.append([zsig,"%.2f"%np.sum(siglessbins),len(siglessbins)+1,"%.4f"%zs[0],"%.1f"%(newT-thisT),ncalls])
            trows.append([zsig,np.sum(siglessbins),len(siglessbins)+1,zs[0],(newT-thisT),ncalls])
            thisT=newT
        # print_table(rows)
    print()
    print_table(trows)
    
        # ## Loop on signal fractions - detail around 3/5 sigma
        # rows=[["sig-fraction","n_sig","n_ggh","n_vbf","signif"]]
        # startT=timeit.default_timer()
        # thisT=startT
        # print("Starting tests")
        # # for sigf in list(np.linspace(0.1,0.3,21)):
        # # for sigf in list(np.linspace(0.11,0.12,21))+list(np.linspace(0.19,0.20,21)):
        # # for sigf in list(np.linspace(0.16,0.17,21))+list(np.linspace(0.27,0.28,21)):
        # for sigf in list(np.linspace(0.155,0.16,11)):
        # # for sigf in list(np.linspace(0.26,0.27,21)):
        #     if sigf==0:
        #         continue
        #     ## Make test matrix (add signal events to bkg)
        #     n_sig=int(round(sigf*n_bkg/100))
        #     n_sig_ggh=int(round(n_sig/(1+sig_ratio)))
        #     n_sig_vbf=n_sig-n_sig_ggh
        #     sig_ggh=get_matrix(sig_ggh_data[:n_sig_ggh,:],xaxis,yaxis)
        #     sig_vbf=get_matrix(sig_vbf_data[:n_sig_vbf,:],xaxis,yaxis)
        #     sig=sig_ggh+sig_vbf
        #     # test=draw_fromtpl(bkg,1)[0,:]+sig
        #     test=bkg+sig
        #     print("sigf nsig nggh nvbf",sigf,n_sig,n_sig_ggh,n_sig_vbf)
        #     # MatrixPlot(test,title="Bkg+sig events").showplot()
        #     ## Perfgorm Likelihood test
        #     q0s,pvals,zs,muhats,b_out,nfcn=perform_lik_test_new([bkg],[test],sig_tpl)#sig/np.sum(sig))
        #     if zs==[]:
        #         rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"fail"])
        #     else:
        #         rows.append(["%.4f"%sigf,n_sig,n_sig_ggh,n_sig_vbf,"%.4f"%zs[0]])
        #     print(rows[-1])
        #     newT=timeit.default_timer()
        #     print("Done (%.1fs, %.1fs)"%(newT-thisT,newT-startT))
        #     thisT=newT
        #     if zs[0]>3:
        #         break
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
    
def get_array(columns,data_path,cuts,isSig):
    ## headers=['Lep0Pt', 'Lep0Eta', 'Lep0Phi', 'Lep1Pt', 'Lep1Eta', 'Lep1Phi', 'MET', 'METPhi', 'MLL', 'MLep0MET', 'MLep1MET', 'Mcoll', 'isSig']
    return get_array_fromcsv(columns,data_path,cuts,isSig)

if __name__=="__main__":
    main()
