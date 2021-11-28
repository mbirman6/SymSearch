from utils3 import *
from plots3 import *

import numpy as np
import matplotlib.pyplot as plt
from iminuit import minimize,Minuit
from scipy.optimize import curve_fit, fsolve#, minimize
from scipy.integrate import quad
from scipy.stats import norm
import numba as nb

txaxis=[30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170]
tyaxis=[10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]
def idx_fromval(val,axis):
    return int((val-axis[0])*1./5)
def idxs_fromvals(vals):
    return [idx_fromval(vals[0],txaxis),idx_fromval(vals[1],tyaxis)]
def val_fromidx(idx,axis):
    return idx*5.+axis[0]
def vals_fromidxs(idxs):
    return [val_fromidx(idxs[0],txaxis),val_fromidx(idxs[1],tyaxis)]
    
# def get_templates(addentries_emu,entries_flat):
#     dct={}
#     ## bkg emu template
#     tplfname="utils/em_reco_none_Mcoll_Lep0Pt_28_28_no_signal.npz"
#     dct["bkg_emu"]=np.load(tplfname)['entries']+addentries_emu
#     ## bkg flat
#     dct["bkg_flat"]=entries_flat*np.ones((28,28))
#     ## signals
#     for sigtype in ['rect_lowstat','rect_highstat','gaus_lowstat','gaus_highstat']:
#         mx,my=[125,125] if 'lowstat' in sigtype else [65,35]
#         std=25 if 'rect' in sigtype else 10
#         sig=np.zeros((28,28))
#         for i in range(28):
#             for j in range(28):
#                 vx,vy=vals_fromidxs([i,j])
#                 if 'rect' in sigtype and mx-std<=vx<mx+std and my-std<=vy<my+std:
#                     sig[i,j]=1
#                 elif 'gaus' in sigtype:
#                     sig[i,j]=myGauss(vx+2.5,mx,std,1)*myGauss(vy+2.5,my,std,1) # get value at bin-center (+2.5)
#         dct['sig_'+sigtype]=sig/totsum(sig)
#     return dct

def get_templates(bkgtypes,sigtypes):
    locdct={'mid':[14,14],'low':[6,6],'high':[20,20]}
    dct={}
    tplfname="utils/em_reco_none_Mcoll_Lep0Pt_28_28_no_signal.npz"
    for bkgtype in bkgtypes:
        bkglab,num=bkgtype.split('-')
        if bkglab=='emu':
            dct['bkg_'+bkgtype]=np.load(tplfname)['entries']+float(num) # num = entries added per bin
        elif bkglab=='flat':
            dct['bkg_'+bkgtype]=float(num)*np.ones((28,28)) # num = entries per bin
    for sigtype in sigtypes:
        siglab,num,loclab=sigtype.split('-')
        num=float(num)
        loc=locdct[loclab]
        sig=np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                # sig[i,j]=i+j
                if siglab=='rect' and loc[0]-num/2<=i<loc[0]+num/2 and loc[1]-num/2<=j<loc[1]+num/2:
                    # sig[i,j]+=50
                    sig[i,j]=1
                elif siglab=='gaus':
                    sig[i,j]=myGauss(i+0.5,loc[0],num,1)*myGauss(j+0.5,loc[1],num,1) # get value at bin-center (+0.5)
        dct["sig_%s"%sigtype]=sig/totsum(sig)
    return dct

def make_template_plots(tpldct,show=False):
    for tpltype,tpl in tpldct.items():
        mplot=MatrixPlot(tpl,"Template: %s"%tpltype)
        if show:
            mplot.showplot()
        else:
            mplot.saveplot("plots/tpl_%s.png"%tpltype)

def make_bkg_plots(bkg,tpl,bkgtype,show=False):
    mplot=MatrixPlot(bkg,"bkg from %s template"%bkgtype)
    if show:
        mplot.showplot()
    else:
        mplot.saveplot("plots/bkg_%s.png"%bkgtype)
    mplot=MatrixPlot(NSigma(bkg,tpl),"NSigma(bkg,tpl) from %s template"%bkgtype)
    if show:
        mplot.showplot()
    else:
        mplot.saveplot("plots/bkgdifftpl_%s.png"%bkgtype)
    
def draw_fromtpl(tpl,N):
    tpls=np.repeat(tpl[np.newaxis,:,:],N,axis=0)
    return np.random.poisson(tpls)

def add_sig2bkg(bkg,sigtpl,zin):
    sigmu=fsolve(Calcq0MinusZ,x0=200,args=(bkg,sigtpl,zin))[0]
    if not np.isclose(Calcq0MinusZ(sigmu,bkg,sigtpl,zin),0):
        print(Problem)
    sig=sigmu*sigtpl
    sigbkg=bkg+sig
    return sigbkg,sig,sigmu

def showcase_bkgsigtypes(tpldct,bkgtypes,sigtypes,showplots):
    N=1
    rows=[['bkg','sig','z','sigmu','sigmax']]
    for signif in [3,5]:
        for bkgtype in bkgtypes:
            bkgtpl=tpldct['bkg_'+bkgtype]
            bkg=draw_fromtpl(bkgtpl,N)[0,:,:] # bkg only
            make_bkg_plots(bkg,bkgtpl,bkgtype,showplots)
            ## Add signal
            for sigtype in sigtypes:
                sigtpl=tpldct['sig_'+sigtype]
                sigbkg,sig,sigmu=add_sig2bkg(bkg,sigtpl,zin=signif)
                rows.append([bkgtype,sigtype,signif,"%.1f"%sigmu,"%.1f"%(np.max(sig))])
    print_table(rows)

def get_sigbkg_datasets(N,sigtpl,zin,bkgs):
    sigbkgs=np.empty(bkgs.shape)
    sigmus_input=np.empty((N))
    for i in range(N):
        bkg=bkgs[i,:,:]
        sigbkg,sig,sigmu=add_sig2bkg(bkg,sigtpl,zin)
        sigbkgs[i,:,:]=sigbkg
        sigmus_input[i]=sigmu
    return sigbkgs,sigmus_input

def perform_nsigma_test(bkgsA,bkgsB,fitargs_bkg=""):
    ## Get Scores
    scores=list(np.mean(NSigma(bkgsB,bkgsA),axis=(1,2)))
    ## Get bkg-only pdf (Hist + Gauss Fit)
    if not fitargs_bkg:
        hplot=HistsPlot([scores])
        hplot.make_hists()
        bins=hplot.bincenters
        binvals=hplot.binvals[0]
        popt,pcovs=curve_fit(myGauss,xdata=bins,ydata=binvals)
        fitargs_bkg=tuple(popt)
    bkgpdf_area,_=quad(myGauss,-np.inf,np.inf,args=fitargs_bkg)
    ## Get p-values and signifs
    pvalues=[]
    signifs=[]
    for score in scores:
        pval,_=quad(myGauss,score,np.inf,args=fitargs_bkg)
        pval/=bkgpdf_area
        cdf=1-pval
        z=norm.ppf(cdf)
        pvalues.append(pval)
        signifs.append(z)
    return scores,pvalues,signifs,fitargs_bkg

def perform_delta_test(bkgsA,bkgsB,fitargs_bkg=""):
    ## Get Scores
    scores=list(np.mean(bkgsB-bkgsA,axis=(1,2)))
    ## Get bkg-only pdf (Hist + Gauss Fit)
    if not fitargs_bkg:
        hplot=HistsPlot([scores])
        hplot.make_hists()
        bins=hplot.bincenters
        binvals=hplot.binvals[0]
        popt,pcovs=curve_fit(myGauss,xdata=bins,ydata=binvals)
        fitargs_bkg=tuple(popt)
    bkgpdf_area,_=quad(myGauss,-np.inf,np.inf,args=fitargs_bkg)
    ## Get p-values and signifs
    pvalues=[]
    signifs=[]
    for score in scores:
        pval,_=quad(myGauss,score,np.inf,args=fitargs_bkg)
        pval/=bkgpdf_area
        cdf=1-pval
        z=norm.ppf(cdf)
        pvalues.append(pval)
        signifs.append(z)
    return scores,pvalues,signifs,fitargs_bkg

def perform_lik_test(bkgsA,bkgsB,sigtpl):
    scores=[]
    pvalues=[]
    signifs=[]
    sigmus={'tot':[],'1sA':[],'restA':[],'1sB':[],'restB':[]}
    for bA,bB in zip(bkgsA,bkgsB):
        ## Get Score + MuHat
        try:
            minres=minimize(CalcMinusq0,x0=[0],args=(bB,bA,sigtpl),method='Nelder-Mead')
            # if not minres['success']:
            #     continue
        except:
            print("problem")
            print(minres)
            # input()
        # print(minres)
        sigmu_tot=minres['x'][0] # muhat from minimization
        q0=-minres['fun'] if sigmu_tot>0 else 0
        sigmus['tot'].append(sigmu_tot)
        scores.append(q0)
        ## Convert muhat to mu_1s*muhat_rest
        sigmu_1sA=fsolve(Calcq0MinusZ,x0=200,args=(bA,sigtpl,1))[0] # mu for 1sigma from bkg estimate
        sigmu_1sB=fsolve(Calcq0MinusZ,x0=200,args=(bB,sigtpl,1))[0] # mu for 1sigma from bkg estimate
        sigmu_restA=sigmu_tot/sigmu_1sA
        sigmu_restB=sigmu_tot/sigmu_1sB
        sigmus['1sA'].append(sigmu_1sA)
        sigmus['restA'].append(sigmu_restA)
        sigmus['1sB'].append(sigmu_1sB)
        sigmus['restB'].append(sigmu_restB)
        ## Get p-values and signifs from asymptotic formula
        z=np.sqrt(q0) #if q0>0 else -np.sqrt(-q0)
        pval=1-norm.cdf(z)
        pvalues.append(pval)
        signifs.append(z)
    return scores,pvalues,signifs,sigmus

A=0
B=0
S=0
nbins=28
@nb.njit(parallel=False,fastmath=True)
def CalcMinusq0_new(x):#,B,A,S):
    mu=x[0]
    b=x[1:]#.reshape((nbins,nbins))
    q0=-2*np.sum(mu*S+2*b+(A+B)*(np.log(A+B)-1)-A*np.log(2*b)-B*np.log(2*(b+mu*S)))
    # print(mu,q0)
    return -q0

# class negq0():
#     def __init__(self,A,B,S):
#         self.A=A
#         self.B=B
#         self.S=S
#     def calc_negq0(self,x):
#         A=self.A
#         B=self.B
#         S=self.S
#         mu=x[0]
#         b=x[1:].reshape((nbins,nbins))
#         q0=-2*np.sum(mu*S+2*b+(A+B)*(np.log(A+B)-1)-A*np.log(2*b)-B*np.log(2*(b+mu*S)))
#         return -q0
    
def perform_lik_test_new(bkgsA,bkgsB,sigtpl,selectbins=np.array(())):
    global A
    global B
    global S
    global nbins
    if selectbins.any():
        S=sigtpl[selectbins]
        nbins=S.shape[0]
    else:
        nbins=sigtpl.shape[0]
        S=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
    scores=[]
    pvalues=[]
    signifs=[]
    sigmus={'tot':[],'1sA':[],'restA':[],'1sB':[],'restB':[]}
    for bA,bB in zip(bkgsA,bkgsB):
        if selectbins.any():
            A=bA[selectbins]
            B=bB[selectbins]
        else:
            A=bA.reshape((1,nbins**2)).squeeze(axis=0)
            B=bB.reshape((1,nbins**2)).squeeze(axis=0)
        ## Fit parameters initial guess
        initmu=np.sum(B-A)
        initbs=(A+B)/2
        initvals=np.concatenate((np.array((initmu,)),initbs))
        nparams=initvals.shape[0]
        limits=[(0,None) for n in range(nparams)]
        Is=range(1,nparams)
        names=['mu']+["b%s"%i for i in Is]
        print(A.shape,B.shape,S.shape,initvals.shape,len(names),len(limits))
        ## Perform fit
        CalcMinusq0_new.recompile() ## needed since change in globals
        m=Minuit(CalcMinusq0_new,initvals,name=names)#,error=1)#,errordef=0.5)
        # m=Minuit(negq0(A,B,S).calc_negq0,initvals,name=names)#,error=1)#,errordef=0.5)
        m.limits=limits
        # m.errordef=Minuit.LIKELIHOOD
        m.errordef=Minuit.LIKELIHOOD
        m.print_level=0
        m.migrad()
        if not m.valid:
            print("WARNING: Problem in minimization")
            continue
        print(m.fmin)
        # print(m.values["mu"],m.fval,np.sqrt(-m.fval))
        sigmu_tot=m.values["mu"]#minres['x'][0] # muhat from minimization
        bs=[]
        for i in range(1,len(initbs)+1):
            bs.append(m.values["b%s"%i])
        b_out=np.array(bs).reshape(A.shape)
        q0=-m.fval#-minres['fun'] if sigmu_tot>0 else 0
        sigmus['tot'].append(sigmu_tot)
        scores.append(q0)
        z=np.sqrt(q0) #if q0>0 else -np.sqrt(-q0)
        pval=1-norm.cdf(z)
        pvalues.append(pval)
        signifs.append(z)
    return scores,pvalues,signifs,sigmus,b_out,m.nfcn
