## general
import os
import sys
import timeit
## science
import numpy as np
## mystuff
from myfunctions import *
sys.path.insert(0,"/srv01/agrp/mattiasb/scripts/p3")
from utils import *
from np_utils import *
from plots3 import *

def save_results(data,label,savepath):
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    data=np.array(data)
    np.save(savepath+"/%s.npy"%label,data)

def perform_test(test,bkgtpl,sigtpl,bkgsA,bkgsB,sigtype):
    ## Avoid division by 0
    posbins=(bkgsA>0)
    bkgsA=bkgsA*posbins+1e-6*(1-posbins) # set A bins where A<=0 to 1e-6 
    if '1' not in test.split('-')[0]:
        posbins=(bkgsA+bkgsB>0)
        bkgsA=bkgsA+(1e-6-bkgsB)*(1-posbins) # set A bins where A+B<=0 to 1e-6 - B such that A+B=1e-6
    sonb=sigtpl/(bkgtpl/np.sum(bkgtpl))**0.5
    # MatrixPlot(sonb).showplot()
    ## get select bins
    if '-' in test:
        test,selectbins=test.split('-')
        N=bkgsA.shape[0]
        if selectbins=='pos':
            selectbins=(bkgsB-bkgsA>0)
        # elif selectbins=='sonb':
        #     sonb=sigtpl/(bkgtpl/np.sum(bkgtpl))**0.5
        #     sonb=np.repeat(sonb[np.newaxis,:,:],N,axis=0)
        #     selectbins=sonb>0.3
        elif 'sonb' in selectbins:
            th=float(selectbins.split('sonb')[1])
            sonb=sigtpl/(bkgtpl/np.sum(bkgtpl))**0.5
            selectbins=sonb>th
            selectbins=np.repeat(selectbins[np.newaxis,:,:],N,axis=0)
        elif 'sonA' in selectbins:
            th=float(selectbins.split('sonA')[1])
            nA=np.sum(bkgsA,axis=(1,2))[:,None,None] 
            sonA=np.repeat(sigtpl[np.newaxis,:,:],N,axis=0)/(bkgsA/nA)**0.5
            selectbins=sonA>th
        elif selectbins=='Donb':
            n1=np.sum(bkgsB-bkgsA,axis=(1,2),where=(bkgsB-bkgsA)>0)[:,None,None]
            n2=np.sum(bkgsA+bkgsB,axis=(1,2))[:,None,None]
            Donb=((bkgsB-bkgsA)/n1) / ((bkgsA+bkgsB)/n2)**0.5
            # Donb=(bkgsB-bkgsA)/((bkgsA+bkgsB)/2)**0.5
            selectbins=(Donb>0.01)
            # MatrixPlot(Donb[0]).showplot()
        elif 'win' in selectbins:
            winsize=int(selectbins.split('win')[1])
            locdct={'mid':[14,14],'low':[6,6],'high':[20,20],'hlfv':[19,10]}
            if 'hlfv' in sigtype:
                loc=locdct['hlfv']
            else:
                loc=locdct[sigtype.rsplit('-',1)[1]]
            selectbins=np.zeros(shape=(28,28))
            for i in range(28):
                for j in range(28):
                    if loc[0]-winsize/2<=i<loc[0]+winsize/2 and loc[1]-winsize/2<=j<loc[1]+winsize/2:
                        selectbins[i,j]=1
            selectbins=np.repeat(selectbins[np.newaxis,:,:],N,axis=0)
    else:
        selectbins=np.ones(shape=bkgsA.shape)
    ## Nsigma-type tests
    if test[:-1] in ['Nsigma','Skellam','LDD']:
        # have to keep shape so not mix bkgsA,bkgsB pairs - set bkgsB=bkgsA on removed bins
        nbins=np.sum(1*selectbins,axis=(1,2))
        # print(nbins[:10])
        bkgsB=bkgsB*selectbins+bkgsA*(1-selectbins)
        return multitest(test,bkgsA,bkgsB,nbins)
    ## Lik-type tests
    results=[]
    i=-1
    badidxs=[]
    for bA,bB,sbins in zip(bkgsA,bkgsB,selectbins):
        i+=1
        # print(i)
        bA=bA[sbins]
        bB=bB[sbins]
        # if i==0:
        #     print('nbins',bA.shape)
        sigtpl0=sigtpl[sbins]
        nparams=bA.shape[0]+1
        if test=='L1':
            muhat,q0,pval,signif=minimize_L1(bA,bB,sigtpl0)
        elif test=='L2':
            muhat,q0,pval,signif,b_out=minimize_L2(bA,bB,sigtpl0)
        if q0==None:
            badidxs.append(i)
            continue    
        # results.append([muhat,q0])
        results.append(signif)
    return results,badidxs

def main(tests,datafiles,savepath="",jobidx=""):
    ## Get templates per datafile
    datadct={}
    for f in datafiles:
        datadct[f]={}
        samp=f.rsplit('/',1)[1].split('.npy')[0]
        b,s,zinltest=samp.split('_')
        datadct[f]['samp']=samp
        datadct[f]['bkgtype']=b
        datadct[f]['sigtype']=s
    bkgtypes=sorted(list(set([fdct['bkgtype'] for f,fdct in datadct.items()])))
    sigtypes=sorted(list(set([fdct['sigtype'] for f,fdct in datadct.items()])))
    tpldct=get_templates(bkgtypes,sigtypes)
    ## Loop on datafiles
    for f in datafiles:
        print(f)
        ## Get the templates
        bkgtpl=tpldct['bkg_'+datadct[f]['bkgtype']]
        sigtpl=tpldct['sig_'+datadct[f]['sigtype']]
        ## Get the data
        if len(f.split(':'))>1: ## run only on part of the datafile
            isslice=True
            f0,startidx,stopidx=f.split(':')
            startidx=int(startidx)
            stopidx=int(stopidx)
            data=np.load(f0)
            bkgsB=data[0,startidx:stopidx,:,:] ## B-bkg matrices
            sigbkgsB=data[1,startidx:stopidx,:,:] ## B-bkg+sig matrices
        else:
            isslice=False
            data=np.load(f)
            bkgsB=data[0,:,:,:] ## B-bkg matrices
            sigbkgsB=data[1,:,:,:] ## B-bkg+sig matrices 
        N=bkgsB.shape[0]
        tplbkgsA=np.repeat(bkgtpl[np.newaxis,:,:],N,axis=0)
        bkgsA=draw_fromtpl(bkgtpl,N)
        ## Loop on tests
        scoredct={}
        scores=[]
        for test in tests:
            print(test)
            if '1' in test.split('-')[0]: # A is template, B is drawn
                results_bkg,badidxs_bkg=perform_test(test,bkgtpl,sigtpl,tplbkgsA,bkgsB,datadct[f]['sigtype'])
                results_sigbkg,badidxs_sigbkg=perform_test(test,bkgtpl,sigtpl,tplbkgsA,sigbkgsB,datadct[f]['sigtype'])
            else: # A and B are drawn
                results_bkg,badidxs_bkg=perform_test(test,bkgtpl,sigtpl,bkgsA,bkgsB,datadct[f]['sigtype'])
                results_sigbkg,badidxs_sigbkg=perform_test(test,bkgtpl,sigtpl,bkgsA,sigbkgsB,datadct[f]['sigtype'])
            badidxs=sorted(badidxs_bkg+badidxs_sigbkg)
            for c,idx in enumerate(badidxs):
                if idx in badidxs_bkg and idx in badidxs_sigbkg:
                    continue
                if idx in badidxs_bkg:
                    del results_sigbkg[idx-c]
                elif idx in badidxs_sigbkg:
                    del results_bkg[idx-c]
            ## Save
            savedata=np.array((results_bkg,results_sigbkg))
            if isslice:
                save_results(savedata,"%s_%s.%s.%s"%(test,datadct[f]['samp'],startidx,stopidx),savepath)
            else:
                save_results(savedata,"%s_%s"%(test,datadct[f]['samp']),savepath)
            # scores.append([np.mean(results_bkg),np.mean(results_sigbkg)])
            # s0,s1=scores[-1]
            # print(s0,s1,s1-s0)
            scoredct[test]=savedata
    
        # print()
        # ## Get cumul pdfs
        # tests=sorted(scoredct.keys())
        # bpdfs=[]
        # spdfs=[]
        # npdf=np.random.normal(0,1,N)
        # for test in tests:
        #     bpdf=scoredct[test][0,:]
        #     spdf=scoredct[test][1,:]
        #     hplot=HistsPlot([bpdf,spdf],title=test)
        #     hplot.make_hists()
        #     bvals,svals=hplot.binvals
        #     m=0
        #     M=0
        #     for b,s in zip(bvals,svals):
        #         m+=min(b,s)
        #         M+=max(b,s)
        #     print(test,"%.3f"%(1-m/M))
        #     # hplot.showplot()
        #     # bpdfs.append((bpdf-np.mean(bpdf))/np.std(bpdf))
        #     # spdfs.append((spdf-np.mean(bpdf))/np.std(bpdf))
        #     bpdfs.append(bpdf)
        #     spdfs.append(spdf)
        # hplot=HistsPlot([npdf]+bpdfs+spdfs,['N(0,1)']+tests+tests)
        # # hplot.cumulative=-1
        # # hplot.normed=True
        # hplot.cumulative=0
        # hplot.normed=0
        # hplot.legloc='upper right'
        # # hplot.make_hists()
        # hplot.showplot()
            
if __name__=="__main__":
    if len(sys.argv)==5:
        jobname,datafiles,savepath,seed=sys.argv[1:]
        test,jobidx=jobname.split('_')
        np.random.seed(int(seed))
        jobidx='_'+jobidx
        datafiles=[f.strip() for f in open(datafiles).readlines()]
        main([test],datafiles,savepath,jobidx)
    else:
        ## ------------------ OPTIONS ---------------------------------------
        seed=40
        np.random.seed(seed) # allways same random
        ## N tests per datafile
        N=20
        ## input data to run on
        bkgtypes=['mue-25']#,'flat-100']#,'mue-25','mue-0','flat-10000']
        # sigtypes=['rect-1-mid']
        sigtypes=['hlfv-mutau','rect-6-low','gaus-2-low','rect-6-high','gaus-2-high']
        # sigtypes=['hlfv-mutau','rect-6-mid','gaus-2']#-low','gaus-2-high','hlfv-mutau']
        zins=[3]#,5,10]
        ltests=['L2']#,'L2']
        datafiles=[]
        for b in bkgtypes:
            for s in sigtypes:
                if 'flat' in b and 'low' in s:
                    warn('no low in flat, running mid')
                    s=s.replace('low','mid')
                if 'flat' in b and 'high' in s:
                    warn('no high in flat, skipping')
                    continue
                for z in zins:
                    for ltest in ltests:
                        datafiles.append("/srv01/agrp/mattiasb/runners/SymSearch/getdata/merge_outputs/%s_%s_%ssigma%s.npy:0:%s"%(b,s,z,ltest,N))
        ## tests-selectbins to perform
        tests=[
            # 'Nsigma1',
            # 'Nsigma2',
            # 'Skellam1',
            # 'Skellam2',
            # 'Nsigma1-sonb.3',
            # 'Nsigma2-sonb.3',
            # 'Skellam1-sonb.3',
            # 'Skellam2-sonb.3',
            # 'Nsigma1-win5',
            'Nsigma2-win5',
            # 'Nsigma2-win1',
            # 'Skellam1-win5',
            # 'Skellam2-win5',
            # 'L1',
            'L1-sonb.03',
            'L2-sonA.03'
            # 'LDD1-pos',
            # 'LDD2-pos',
        ]
        main(tests,datafiles,savepath="results")
