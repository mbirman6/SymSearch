## general
import os
import sys
## science
import numpy as np

def main(haddcmd):
    mergefile=haddcmd[0]
    mergepath=mergefile.rsplit('/',1)[0]
    if not os.path.isdir(mergepath):
        os.makedirs(mergepath)
    npfiles=haddcmd[1:]
    all_arrays=[]
    for npfile in npfiles:
        all_arrays.append(np.load(npfile))
    all_arrays=np.concatenate(all_arrays,axis=1)
    np.save(mergefile,all_arrays)

if __name__=="__main__":
    main(sys.argv[1:])
