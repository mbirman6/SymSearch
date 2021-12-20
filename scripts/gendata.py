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

def save_dataset(data,label,savepath):
    if not os.path.isdir(savepath):
        os.makedirs(savepath)
    data=np.array(data)
    np.save(savepath+"/%s.npy"%label,data)

def main(N,bkgtypes,sigtypes,zins,ltests,savepath="",jobidx=""):
    tpldct=get_templates(bkgtypes,sigtypes)
    for bkgtype in bkgtypes:
        print(">>>>>>>> bkgtype %s"%bkgtype)
        bkgtpl=tpldct['bkg_'+bkgtype]
        bkgs=draw_fromtpl(bkgtpl,N)
        ## If separate bkg / bkg+sig
        # if savepath and not os.path.isfile(savepath+"/data_%s_nosig%s.npy"%(bkgtype,jobidx)):
        #     save_dataset(bkgs,"data_%s_nosig%s"%(jobidx,bkgtype),savepath)
        for sigtype in sigtypes:
            print("  >>>>>> sigtype %s"%sigtype)
            sigtpl=tpldct['sig_'+sigtype]
            # get_njobs(bkgtpl,sigtpl)
            # continue
            for zin in zins:
                print("    >>>> signif %s"%zin)
                for ltest in ltests:
                    print("      >> ltest %s"%ltest)
                    t0=timeit.default_timer()
                    sigbkgs,sigmus=get_sigbkg_datasets(N,sigtpl,zin,bkgs,lik=ltest,doselectbins=True)
                    # PATCH - DEBUG
                    # MatrixPlot(bkgs[0]).showplot()
                    # MatrixPlot(sigbkgs[0]).showplot()
                    # MatrixPlot(sigbkgs[0]-bkgs[0]).showplot()
                    # for i in range(bkgs.shape[0]):
                    #     sonb=sigtpl/(bkgtpl/np.sum(bkgtpl))**0.5
                    #     selectbins=sonb>0.03
                    #     bkg=bkgs[i][selectbins]
                    #     sigbkg=sigbkgs[i][selectbins]
                    #     muhat,q0,pval,signif0,b_out=minimize_L2(bkg,sigbkg,sigtpl[selectbins])
                    #     print(signif0)
                    # continue
                    t1=timeit.default_timer()
                    print("         average sigmu",np.mean(sigmus),", time: %.3fs"%(t1-t0))
                    if savepath:
                        ## If separate bkg / bkg+sig
                        # save_dataset(sigbkgs,"data_%s_%s_%ssigma%s%s"%(bkgtype,sigtype,int(zin),ltest,jobidx),savepath)
                        ## If together bkg / bkg+sig
                        savedata=np.array((bkgs,sigbkgs))
                        save_dataset(savedata,"data_%s_%s_%ssigma%s%s"%(bkgtype,sigtype,int(zin),ltest,jobidx),savepath)

if __name__=="__main__":
    if len(sys.argv)==5:
        N,jobname,savepath,seed=sys.argv[1:]
        bkgtype,sigtype,zinltest,jobidx=jobname.split('_')
        zin,ltest=zinltest.split('sigma')
        np.random.seed(int(seed))
        jobidx='_'+jobidx
        main(int(N),[bkgtype],[sigtype],[float(zin)],[ltest],savepath,jobidx)
    else:
        ## ------------------ OPTIONS ---------------------------------------
        seed=42
        np.random.seed(seed) # allways same random
        ## Define different bkgtypes and signaltypes to loop on (see get_templates)
        ## bkgtypes:
        ##   flat-X is flat with X entries in each bin
        ##   emu-X is emu template with X entries added to each bin
        bkgtypes=['mue-25','flat-100']#,'mue-25','mue-0','flat-10000']
        ## sigtypes: siglab-X-loc where loc is high/mid/low for bin location (20,20)/(14,14)/(6,6)
        ##   rect-X-loc is rect signal of size X*X bins
        ##   gaus-X-loc is gaus signal with std=X
        sigtypes=['gaus-2-low']#,'gaus-2-high','hlfv-mutau']
        # sigtypes=['gaus-2-low']
        # sigtypes=['hlfv-mutau']
        ## injected significances to loop on
        zins=[5]#,5,10]
        ## likelihood tests
        ltests=['L2']#L1','L2']
        ## size of samples
        # N=1000
        N=10
        main(N,bkgtypes,sigtypes,zins,ltests=ltests,savepath='')#"datasets")
    

