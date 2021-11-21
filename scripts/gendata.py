## general
import os
import sys
## science
import numpy as np
## mystuff
sys.path.insert(0,"utils")
from utils3 import *
from myfunctions import *

seed=42
np.random.seed(seed) # allways same random

## ------------------ OPTIONS ---------------------------------------
## Define different bkgtypes and signaltypes to loop on (see get_templates)
## bkgtypes:
##   flat-X is flat with X entries in each bin
##   emu-X is emu template with X entries added to each bin
bkgtypes=['flat-100','emu-25']
## sigtypes: siglab-X-loc where loc is high/mid/low for bin location (20,20)/(14,14)/(6,6)
##   rect-X-loc is rect signal of size X*X bins
##   gaus-X-loc is gaus signal with std=X
sigtypes=['rect-2-mid','gaus-2-low','gaus-2-high']
## injected significances to loop on
zins=[3,5,10]
## size of samples
N=1000

def save_dataset(samp,label):
    if not os.path.isdir("datasets"):
        os.makedirs("datasets")
    samp=np.array(samp)
    np.save("datasets/%s.npy"%label,samp)

tpldct=get_templates(bkgtypes,sigtypes)
for bkgtype in bkgtypes:
    print(">>>> bkgtype %s"%bkgtype)
    bkgtpl=tpldct['bkg_'+bkgtype]
    tplbkgsA=[bkgtpl for i in range(N)]
    save_dataset(tplbkgsA,"tplsA_%s"%bkgtype)
    bkgsA,bkgsB=[draw_fromtpl(bkgtpl,N),draw_fromtpl(bkgtpl,N)]
    save_dataset(bkgsA,"bkgsA_%s"%bkgtype)
    save_dataset(bkgsB,"bkgsB_%s"%bkgtype)
    for sigtype in sigtypes:
        print("  >> sigtype %s"%sigtype)
        sigtpl=tpldct['sig_'+sigtype]
        for zin in [3,5,10]:
            sigbkgsB,sigmus=get_sigbkg_datasets(N,sigtpl,zin,bkgsB)
            save_dataset(sigbkgsB,"sigbkgsB_%s_%s_%ssigma"%(bkgtype,sigtype,zin))
