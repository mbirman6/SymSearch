## general
import os
import sys
import timeit
## science
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.optimize import curve_fit,fsolve
from scipy.integrate import quad
from scipy.stats import norm
import numba as nb
## mystuff
from myfunctions import *
sys.path.insert(0,"/srv01/agrp/mattiasb/scripts/p3")
sys.path.insert(0,"/Users/mattiasbirman/scripts/p3")
from utils import * 
from np_utils import *
from plots3 import *
    
def get_templates(bkgtypes,sigtypes):
    locdct={'mid':[14,14],'low':[6,6],'high':[20,20]}
    dct={}
    for bkgtype in bkgtypes:
        bkglab,num=bkgtype.split('-')
        # ## old
        # if bkglab=='emu':
        #     tplfname="utils/em_reco_none_Mcoll_Lep0Pt_28_28_no_signal.npz"
        #     dct['bkg_'+bkgtype]=np.load(tplfname)['entries']+float(num) # num = entries added per bin
        ## old
        if bkglab=='mue':
            dct['bkg_'+bkgtype]=np.load("bkg_mue.npy")+float(num) # num = entries added per bin
        elif bkglab=='flat':
            dct['bkg_'+bkgtype]=float(num)*np.ones((28,28)) # num = entries per bin
    for sigtype in sigtypes:
        sigsp=sigtype.split('-')
        if sigtype=='hlfv-mutau':
            dct["sig_%s"%sigtype]=np.load("sig_hlfv-mutau.npy")
        else:
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
            dct["sig_%s"%sigtype]=sig/np.sum(sig)
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
                sigmu,_=getmuin(bkg,sigtpl,zin=signif)
                rows.append([bkgtype,sigtype,signif,"%.1f"%sigmu,"%.1f"%(np.max(sig))])
    print_table(rows)

def draw_fromtpl(tpl,N):
    tpls=np.repeat(tpl[np.newaxis,:,:],N,axis=0)
    return np.random.poisson(tpls)

def get_sigbkg_datasets(N,sigtpl,zin,bkgs,lik='L1',doselectbins=False):
    sigbkgs=np.empty(bkgs.shape)
    sigmus_input=np.empty((N))
    for i in range(N):
        bkg=bkgs[i,:,:]
        sigmu,nparams=getmuin(bkg,sigtpl,zin,lik,doselectbins)
        sigbkgs[i,:,:]=bkg+sigmu*sigtpl
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
    
def perform_lik_test_new(bkgsA,bkgsB,sigtpl,selectbins=np.array(())):
    global A
    global B
    global S
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
        print(initmu,A.shape,B.shape,S.shape,initvals.shape,len(names),len(limits))
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

def get_nperjob_persamp(samples):
    bkgtypes=[]
    sigtypes=[]
    for samp in samples:
        b,s=samp.split('_')
        bkgtypes.append(b)
        sigtypes.append(s)
    tpldct=get_templates(list(set(bkgtypes)),list(set(sigtypes)))
    dct={}
    for samp in samples:
        b,s=samp.split('_')
        dct[samp]=get_nperjob(tpldct['bkg_'+b],tpldct['sig_'+s])
    return dct

sonb_th=0.03#3.5e-5
def get_nperjob(bkg,sigtpl):
    # 20e3calls -> 0.4s -> 3000jobs
    # 320e3calls -> 9.2s -> 150jobs
    ## Avoid division by 0
    if np.any(bkg==0):
        bkg=bkg+1e-6*(bkg==0)
    ## Select bins with s/sqrt(b)>th
    sonb=sigtpl/(bkg/np.sum(bkg))**0.5
    selectbins=sonb>sonb_th
    bkg=bkg[selectbins]
    nparams=bkg.shape[0]
    ncalls=0.5*nparams**2+14.9*nparams
    njobs=607.7*np.exp(-8e-6*ncalls)
    njobs=int(round(njobs/50))*50
    # print("nparams,njobs",nparams,njobs)
    return [nparams,njobs]

def comp_mus(samples):
    bkgtypes=[]
    sigtypes=[]
    for samp in samples:
        b,s=samp.split('_')
        bkgtypes.append(b)
        sigtypes.append(s)
    tpldct=get_templates(list(set(bkgtypes)),list(set(sigtypes)))
    dct={}
    for samp in samples:
        print(">>>>",samp)
        b,s=samp.split('_')
        b=tpldct['bkg_'+b]
        s=tpldct['sig_'+s]
        mu,nparams=getmuin(b,s,5,lik='L1',doselectbins=False)
        print('L1',nparams,"%.2f"%mu)
        mu,nparams=minimize(b,b+mu*s,s,lik='L1',doselectbins=False)
        print('M1',nparams,"%.2f"%mu)
        mu,nparams=getmuin(b,s,5,lik='L1',doselectbins=True)
        print('L1',nparams,"%.2f"%mu)
        mu,nparams=minimize(b,b+mu*s,s,lik='L1',doselectbins=True)
        print('M1',nparams,"%.2f"%mu)
        # mu,nparams=getmuin(b,s,5,lik='L2',doselectbins=False)
        # print('L2',nparams,"%.2f"%mu)
        # mu,nparams=minimize(b,b+mu*s,s,lik='L2',doselectbins=False)
        # print('M2',nparams,"%.2f"%mu)
        mu,nparams=getmuin(b,s,5,lik='L2',doselectbins=True)
        print('L2',nparams,"%.2f"%mu)
        mu,nparams=minimize(b,b+mu*s,s,lik='L2',doselectbins=True)
        print('M2',nparams,"%.2f"%mu)

A=0
B=0
S=0
Z=0
@nb.njit(parallel=False,fastmath=True)
def negq0_L1(mu):# A,B,S
    mu=mu[0]
    q0=-2*np.sum(mu*S-B*np.log(1+mu*S/A))
    return -q0
def q0minusZ2_L1(mu):# B,S,Z
    q0=-2*np.sum(mu*S-(B+mu*S)*np.log(1+mu*S/B))
    return q0-Z**2
@nb.njit(parallel=False,fastmath=True)
def negq0_L2(x):# A,B,S
    mu=x[0]
    b=x[1:]
    q0=-2*np.sum(mu*S+2*b+(A+B)*(np.log(A+B)-1)-A*np.log(2*b)-B*np.log(2*(b+mu*S)))
    return -q0
def q0minusZ2_L2(mu):# B,S,Z
    q0=-2*np.sum((2*B+mu*S)*np.log(2*B+mu*S)-B*np.log(2*B)-(B+mu*S)*np.log(2*(B+mu*S)))
    return q0-Z**2
# @nb.njit(parallel=False,fastmath=True)
# def q0minusZ2_L2(x):# B,S,Z
#     mu=x[0]
#     b=x[1:]
#     # q0=-2*np.sum(2*(b-B)+(2*B+mu*S)*np.log(2*B+mu*S)-B*np.log(2*b)-(B+mu*S)*np.log(2*(b+mu*S)))
#     q0=-2*np.sum(2*(B-B)+(2*B+mu*S)*np.log(2*B+mu*S)-B*np.log(2*B)-(B+mu*S)*np.log(2*(B+mu*S)))
#     return (q0-Z**2)**2

def multitest(test,A,B,nbins=784):
    ## axes for summation - A.shape=(28,28) or (N,28,28) or (N,nbins)
    if A.shape[0]==28 and A.shape[1]==28:
        axis=(0,1)
    elif len(A.shape)>2:
        axis=(1,2)
    else:
        axis=(1)
        # nbins=A.shape[1]
    ## perform test
    if test=='Nsigma1':
        return np.sum((B-A)/np.sqrt(A),axis=axis)/np.sqrt(nbins)
    elif test=='Nsigma2':
        return np.sum((B-A)/np.sqrt(A+B),axis=axis)/np.sqrt(nbins)
    elif test=='Skellam1':
        return np.mean(B-A,axis=axis)
    elif test=='Skellam2':
        return np.mean(B-A,axis=axis)/np.sqrt(2)
    elif test=='LDD1':
        # MatrixPlot(-2*(B-A - B*np.log(B/A))).showplot()
        return -2*np.sum(B-A - B*np.log(B/A),axis=axis)
    elif test=='LDD2':
        return -2*np.sum((A+B)*np.log((A+B)/2) - A*np.log(A) - B*np.log(B),axis=axis)

def getmuin(bkg,sigtpl,zin,lik='L1',doselectbins=False):
    ## Avoid division by 0
    if np.any(bkg==0):
        bkg=bkg+1e-6*(bkg==0)
    ## Select bins with s/sqrt(b)>th
    if doselectbins:
        sonb=sigtpl/(bkg/np.sum(bkg))**0.5
        selectbins=sonb>sonb_th
        bkg=bkg[selectbins]
        sigtpl=sigtpl[selectbins]
        nparams=bkg.shape[0]+1
        # print("selectbins %s"%bkg.shape[0])
    else:
        nparams=bkg.shape[0]**2+1
    ## Perform fit
    if lik=='L1':
        sigmu=getmuin_L1(bkg,sigtpl,zin)
    elif lik=='L2':
        sigmu=getmuin_L2(bkg,sigtpl,zin)
    return sigmu,nparams

def minimize(bkgA,bkgB,sigtpl,lik='L1',doselectbins=False):
    ## Avoid division by 0
    if np.any(bkgA==0):
        bkgA=bkgA+1e-6*(bkgA==0)
    ## Select bins with s/sqrt(b)>th
    if doselectbins:
        sonb=sigtpl/(bkgA/np.sum(bkgA))**0.5
        selectbins=sonb>sonb_th
        bkgA=bkgA[selectbins]
        bkgB=bkgB[selectbins]
        sigtpl=sigtpl[selectbins]
        nparams=bkgA.shape[0]+1
        # print("selectbins %s"%bkg.shape[0])
    else:
        nparams=bkgA.shape[0]**2+1
    ## Perform fit
    if lik=='L1':
        sigmu,_,_,_=minimize_L1(bkgA,bkgB,sigtpl)
    elif lik=='L2':
        sigmu,_,_,_,_=minimize_L2(bkgA,bkgB,sigtpl)
    return sigmu,nparams

def minimize_L1(bkgA,bkgB,sigtpl):
    global A
    global B
    global S
    if len(sigtpl.shape)>1:
        nbins=sigtpl.shape[0]
        bkgA=bkgA.reshape((1,nbins**2)).squeeze(axis=0)
        bkgB=bkgB.reshape((1,nbins**2)).squeeze(axis=0)
        sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
    A=bkgA
    B=bkgB
    S=sigtpl
    ## Fit parameters initial guess
    initmu=np.sum(bkgB-bkgA)
    initvals=np.array((initmu,))
    limits=[(None,None)]
    names=['mu']
    # print("minimize_L1",initmu,S.shape)
    ## Perform fit
    negq0_L1.recompile()
    m=Minuit(negq0_L1,initvals,name=names)
    m.limits=limits
    m.errordef=Minuit.LIKELIHOOD
    m.print_level=0
    m.migrad()
    if not m.valid:
        print("WARNING: Problem in minimization")
        return None,None,None,None
    ## Get fit results
    sigmu=m.values["mu"]
    q0=-m.fval
    if q0<0:
        warn("Negative q0, setting to 0",q0)
        q0=0
    signif=np.sqrt(q0)
    pval=1-norm.cdf(signif)
    ## 1 sided or 2 sided mu>0
    if sigmu<0:
        # ## 1-sided
        # q0=0
        # signif=0
        # pval=0.5
        ## 2-sided
        q0=-q0
        signif=-signif
        pval=-pval
    return sigmu,q0,pval,signif

def getmuin_L1(bkg,sigtpl,zin):
    global B
    global S
    global Z
    if len(sigtpl.shape)>1:
        nbins=sigtpl.shape[0]
        bkg=bkg.reshape((1,nbins**2)).squeeze(axis=0)
        sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
    B=bkg
    S=sigtpl
    Z=zin
    initmu=zin*(np.sum(bkg))**0.5
    sigmu=fsolve(q0minusZ2_L1,x0=initmu)[0]
    if not np.isclose(q0minusZ2_L1(sigmu),0):
        print(Problem)
    return sigmu

def minimize_L2(bkgA,bkgB,sigtpl):
    global A
    global B
    global S
    if len(sigtpl.shape)>1:
        nbins=sigtpl.shape[0]
        bkgA=bkgA.reshape((1,nbins**2)).squeeze(axis=0)
        bkgB=bkgB.reshape((1,nbins**2)).squeeze(axis=0)
        sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
    A=bkgA
    B=bkgB
    S=sigtpl
    ## Fit parameters initial guess
    initmu=np.sum(bkgB-bkgA)
    initbs=1*bkgA
    initvals=np.concatenate((np.array((initmu,)),initbs))
    nparams=initvals.shape[0]
    limits=[(None,None)]+[(0,None) for n in range(1,nparams)] # mu can be negative
    Is=range(1,nparams)
    names=['mu']+["b%s"%i for i in Is]
    # print("minimize_L2",initmu,S.shape,len(names))
    ## Perform fit
    negq0_L2.recompile()
    m=Minuit(negq0_L2,initvals,name=names)
    m.limits=limits
    m.errordef=Minuit.LIKELIHOOD
    m.print_level=0
    m.migrad()
    if not m.valid:
        print("WARNING: Problem in minimization")
        return None,None,None,None,None
    ## Get fit results
    sigmu=m.values["mu"]
    bs=[]
    for i in range(1,len(initbs)+1):
        bs.append(m.values["b%s"%i])
    b_out=np.array(bs).reshape(S.shape)
    q0=-m.fval
    if q0<0:
        warn("Negative q0, setting to 0",q0)
        q0=0
    signif=np.sqrt(q0)
    pval=1-norm.cdf(signif)
    ## 1 sided or 2 sided mu>0
    if sigmu<0:
        # ## 1-sided
        # q0=0
        # signif=0
        # pval=0.5
        ## 2-sided
        q0=-q0
        signif=-signif
        pval=-pval
    return sigmu,q0,pval,signif,b_out

def getmuin_L2(bkg,sigtpl,zin):
    global B
    global S
    global Z
    if len(sigtpl.shape)>1:
        nbins=sigtpl.shape[0]
        bkg=bkg.reshape((1,nbins**2)).squeeze(axis=0)
        sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
    B=bkg
    S=sigtpl
    Z=zin
    initmu=zin*(np.sum(bkg))**0.5
    sigmu=fsolve(q0minusZ2_L2,x0=initmu)[0]
    if not np.isclose(q0minusZ2_L2(sigmu),0):
        print(Problem)
    return sigmu

# def getmuin_L2(bkg,sigtpl,zin):
#     global B
#     global S
#     global Z
#     if len(sigtpl.shape)>1:
#         nbins=sigtpl.shape[0]
#         bkg=bkg.reshape((1,nbins**2)).squeeze(axis=0)
#         sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
#     B=bkg
#     S=sigtpl
#     Z=zin
#     ## Fit parameters initial guess
#     initmu=zin*(np.sum(bkg))**0.5
#     initbs=1*bkg
#     initvals=np.concatenate((np.array((initmu,)),initbs))
#     nparams=initvals.shape[0]
#     limits=[(0,None) for n in range(nparams)]
#     Is=range(1,nparams)
#     names=['mu']+["b%s"%i for i in Is]
#     # print("getmuin_L2",Z,initmu,S.shape,len(names))
#     print(initvals.shape)
#     print(initvals)
#     ## Perform fit
#     # q0minusZ2_L2.recompile()
#     m=Minuit(q0minusZ2_L2,initvals,name=names)
#     m.limits=limits
#     m.errordef=Minuit.LIKELIHOOD
#     m.print_level=0
#     m.migrad()
#     print(m)
#     print(m.fval)
#     input()
#     if not m.valid:
#         print("WARNING: Problem in minimization")
#         print(Problem)
#     sigmu=m.values["mu"]
#     print(initmu,sigmu)
#     return sigmu

# def getmuin_L1(bkg,sigtpl,zin):
#     global B
#     global S
#     global Z
#     if len(sigtpl.shape)>1:
#         nbins=sigtpl.shape[0]
#         bkg=bkg.reshape((1,nbins**2)).squeeze(axis=0)
#         sigtpl=sigtpl.reshape((1,nbins**2)).squeeze(axis=0)
#     B=bkg
#     S=sigtpl
#     Z=zin
#     ## Fit parameters initial guess
#     initmu=zin*(np.sum(bkg))**0.5
#     initvals=np.array((initmu,))
#     limits=[(0,None)]
#     names=['mu']
#     print("getmuin_L1",Z,initmu,S.shape,len(names))
#     ## Perform fit
#     q0minusZ2_L1.recompile()
#     m=Minuit(q0minusZ2_L1,initvals,name=names)
#     m.limits=limits
#     m.errordef=Minuit.LIKELIHOOD
#     m.print_level=0
#     m.migrad()
#     if not m.valid:
#         print("WARNING: Problem in minimization")
#         print(Problem)
#     sigmu=m.values["mu"]
#     return sigmu

