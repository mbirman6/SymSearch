## general
import sys
import timeit
## science
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve, minimize
from scipy.integrate import quad
from scipy.stats import norm
## mystuff
sys.path.insert(0,"utils")
from utils3 import *
from plots3 import *
from myfunctions import *

# seed=42
# seed=86
# np.random.seed(seed) # allways same random

showplots=True
# showplots=False
doload=False
doload=True

bkgtypes=['flat-100']#,'emu-25']
# sigtypes=['rect-1','rect-10','rect-28','gaus-1','gaus-2','gaus-10']
sigtypes=['rect-1-mid','rect-28-mid','gaus-2-mid']
sigtypes=['gaus-2-mid']
testtypes=['nsigma','delta']#'lik']
pdftypes=['scores','signifs']

startT=timeit.default_timer()
def main():
    ## ----- Get bkg and sig templates
    # tpldct=get_templates(addentries_emu=25,entries_flat=20)
    tpldct=get_templates(bkgtypes,sigtypes)
    # MatrixPlot(tpldct['sig_gaus-2-mid']).showplot()
    # print(tpldct.keys())
    ## ----- Perform tests per bkg&sig type or load previous results
    if doload:
        for zin in [3]:#,5]:
            # testdct=loadjsonfile("testresults_ErrA_Z%s.json"%zin)
            # testdct=loadjsonfile("testresults_ErrAB_Z%s.json"%zin)
            testdct=loadjsonfile("testresults.json")
            ## ----- Make the plots
            # make_template_plots(tpldct,showplots)
            # showcase_bkgsigtypes(tpldct,bkgtypes,sigtypes,False) # Create 1 dataset per bkg&sig types + plots and stats
            # make_plots(tpldct,testdct,zin)
            # pdfdct=get_pdfs(tpldct,testdct)
            make_plots(tpldct,testdct,zin)
    else:
        zin=3
        # zin=5
        N=1000
        testdct=perform_searches(zin,N,tpldct,saveas="testresults.json")

    # ## Check sigmus
    # N=1000
    # sms={}
    # scs={}
    # labs=[]
    # print("muin sigsize muin/1s_tpl muin/1s_As muin/1s_Bs muhats muhats/1s_tpl muhats/1s_As muhats/1s_Bs")
    # for sigmuin in [0]:#,50,100]:#,100,200,500]:
    #     sms[sigmuin]={}
    #     for std in [1,10,25,70]:#1,5,10,15,20,25]:
    #         mx,my=[100,80]
    #         sigtpl=np.zeros((28,28))
    #         just1=True
    #         siglen=0
    #         for i in range(28):
    #             for j in range(28):
    #                 vx,vy=vals_fromidxs([i,j])
    #                 if mx-std<=vx<mx+std and my-std<=vy<my+std:
    #                     sigtpl[i,j]=1
    #                     if just1:
    #                         siglen+=1
    #             if siglen>0:
    #                 just1=False
    #         sigtpl/=totsum(sigtpl)
    #         bkgtpl=np.ones((28,28))*100
    #         ## PATCH 1bin
    #         # sigtpl=np.ones((1,1))
    #         # bkgtpl=np.ones((1,1))*100
    #         # siglen=1
    #         ### END
    #         bkgsA,bkgsB=[draw_fromtpl(bkgtpl,N),draw_fromtpl(bkgtpl,N)]
    #         # bkgsA=[bkgtpl for i in range(N)]
    #         bkgsB=[bkgtpl for i in range(N)]
    #         testsB=[b+sigmuin*sigtpl for b in bkgsB]
    #         sms[sigmuin][siglen]={}
    #         sms[sigmuin][siglen]['mu_1stpl']=fsolve(Calcq0MinusZ,x0=1,args=(bkgtpl,sigtpl,1))[0] # mu for 1sigma from tpl
    #         # print(sigmuin,siglen,sms[sigmuin][siglen]['mu_1stpl'],Calcq0MinusZ(sms[sigmuin][siglen]['mu_1stpl'],bkgtpl,sigtpl,1),Calcq0MinusZ(siglen*10,bkgtpl,sigtpl,1))
    #         # continue
    #         sms[sigmuin][siglen]['muin_resttpl']=sigmuin/sms[sigmuin][siglen]['mu_1stpl']
    #         # print(sigmuin,siglen,sms[sigmuin][siglen]['mu_1stpl'])
    #         # continue
    #         sms[sigmuin][siglen]['mus_1sA']=[]
    #         sms[sigmuin][siglen]['muins_restA']=[]
    #         sms[sigmuin][siglen]['mus_1sB']=[]
    #         sms[sigmuin][siglen]['muins_restB']=[]
    #         for bA,bB in zip(bkgsA,bkgsB):
    #             sms[sigmuin][siglen]['mus_1sA'].append(fsolve(Calcq0MinusZ,x0=200,args=(bA,sigtpl,1))[0])
    #             sms[sigmuin][siglen]['muins_restA'].append(sigmuin/sms[sigmuin][siglen]['mus_1sA'][-1])
    #             sms[sigmuin][siglen]['mus_1sB'].append(fsolve(Calcq0MinusZ,x0=200,args=(bB,sigtpl,1))[0])
    #             sms[sigmuin][siglen]['muins_restB'].append(sigmuin/sms[sigmuin][siglen]['mus_1sB'][-1])
    #         scores,pvalues,signifs,sigmus=perform_lik_test(bkgsA,testsB,sigtpl)
    #         sms[sigmuin][siglen]['muhats']=sigmus['tot']
    #         sms[sigmuin][siglen]['muhats_restA']=sigmus['restA']
    #         sms[sigmuin][siglen]['muhats_restB']=sigmus['restB']
    #         sms[sigmuin][siglen]['muhats_resttpl']=[]
    #         for muhat in sms[sigmuin][siglen]['muhats']:
    #             sms[sigmuin][siglen]['muhats_resttpl'].append(muhat/sms[sigmuin][siglen]['mu_1stpl'])
    #         print(sigmuin,siglen,"%.2f %.2f+/-%.2f %.2f+/-%.2f %.2f+/-%.2f %.2f+/-%.2f %.2f+/-%.2f %.2f+/-%.2f"%(
    #             sms[sigmuin][siglen]['muin_resttpl'],
    #             np.mean(sms[sigmuin][siglen]['muins_restA']),np.std(sms[sigmuin][siglen]['muins_restA']),
    #             np.mean(sms[sigmuin][siglen]['muins_restB']),np.std(sms[sigmuin][siglen]['muins_restB']),
    #             np.mean(sms[sigmuin][siglen]['muhats']),np.std(sms[sigmuin][siglen]['muhats']),
    #             np.mean(sms[sigmuin][siglen]['muhats_resttpl']),np.std(sms[sigmuin][siglen]['muhats_resttpl']),
    #             np.mean(sms[sigmuin][siglen]['muhats_restA']),np.std(sms[sigmuin][siglen]['muhats_restA']),
    #             np.mean(sms[sigmuin][siglen]['muhats_restB']),np.std(sms[sigmuin][siglen]['muhats_restB'])))
    #     for mu in ['','rest']:
    #         tsms=[]
    #         labs=[]
    #         for siglen in sms[sigmuin]:
    #             labs.append(siglen)
    #             if mu=='rest':
    #                 tsms.append(sms[sigmuin][siglen]['muhats_restB'])
    #             else:
    #                 tsms.append(sms[sigmuin][siglen]['muhats'])
    #         hplot=HistsPlot(tsms,labs)
    #         hplot.histtype='step'
    #         hplot.showplot()
    #         # hplot.saveplot("plots/muhats_%s_muin_%s.png"%(mu,sigmuin))
                  
    # #         sms.append(sigmus)
    # #         scs.append(scores)
    # #         # labs.append(datatype)
    # #         labs.append("%s_%.1f"%(sigmuin,sigmu))
    # #     #     break
    # #     # break
    # # hplot=HistsPlot(scs,labs)
    # # hplot.histtype='step'
    # # hplot.showplot()
    # exit()
            
# def get_pdfs(tpldct,testdct):
#     dct={}
#     ## Get all arrays of values per pdf
#     for pdftype in pdftypes:
#         dct[pdftype]={}
#         dct["cumul"+pdftype]={}
#         for testtype in testtypes:
#             dct[pdftype][testtype]={}
#             dct["cumul"+pdftype][testtype]={}
#             pdfs=[]
#             for bkgtype in bkgtypes:
#                 for sigtype in sigtypes:
#                     if bkgtype=='flat' and 'highstat' in sigtype:
#                         continue
#                     datatype="%s_%s"%(bkgtype,sigtype)
#                     pdfs.append([testdct[datatype]["%s_bkg"%testtype][pdftype],testdct[datatype]["%s_sigbkg"%testtype][pdftype]])
#                     labels.append(datatype)
#             for tpdftype in [pdftype,"cumul"+pdftype]:
#                 if 'cumul' in tpdftype:
#                     documul=1 if pdftype=='pvalues' else -1
#                     donorm=True
#                 else:
#                     documul=False
#                     donorm=False
#                 allpdfs=[p[0]
#                 hplot=HistsPlot(pdfs_bkg+pdfs_sigbkg,labels_bkg+labels_sigbkg)
#                 hplot.histtype='step'
#                 hplot.cumulative=documul
#                 hplot.normed=donorm
#                 hplot.make_hists()
#                 print(tpdftype,testtype)
#                 hplot.showplot()
#                 allbinvals=hplot.binvals
#                 for binvals,label in zip(allbinvals,labels_bkg+labels_sigbkg):
#                     dct[tpdftype][testtype][label]=binvals
                
                
#                 # for i,tsigbkgtype in enumerate(bkgtypes+sigtypes):
#                 #     for i2,sigtype in enumerate(sigtypes):
#                 #         if bkgtype=='flat' and 'highstat' in sigtype:
#                 #             continue
#                 #         ftprs_sigbkg.append([binvals[i1],binvals[i2+2]])
                
#                 # dct[tpdftype][testtype][datatype]
            
def perform_searches(zin,N,tpldct,saveas):
    print(">>>> Datasets size %s"%N)
    testdct={} # dct to save results
    for bkgtype in bkgtypes:
        testdct[bkgtype]={'nsigma':{}}
        ## Draw bkg-only datasets
        bkgtpl=tpldct['bkg_'+bkgtype]
        tplbkgsA=[bkgtpl for i in range(N)]
        bkgsA,bkgsB=[draw_fromtpl(bkgtpl,N),draw_fromtpl(bkgtpl,N)]
        ## NSigma test for BkgOnly
        scores_nsigma_bkg,pvalues_nsigma_bkg,signifs_nsigma_bkg,fitargs_nsigma_bkg=perform_nsigma_test(bkgsA,bkgsB,fitargs_bkg="")
        scores_delta_bkg,pvalues_delta_bkg,signifs_delta_bkg,fitargs_delta_bkg=perform_delta_test(bkgsA,bkgsB,fitargs_bkg="")
        for sigtype in sigtypes:
            print(">>>> Running with bkg \'%s\' and sig \'%s\'"%(bkgtype,sigtype))
            datatype="%s_%s"%(bkgtype,sigtype)
            testdct[datatype]={'nsigma_bkg':{},'nsigma_sigbkg':{},'delta_bkg':{},'delta_sigbkg':{},'lik_bkg':{},'lik_sigbkg':{}}
            ## Make bkg+sig datasets
            sigtpl=tpldct['sig_'+sigtype]
            sigbkgsB,sigmus=get_sigbkg_datasets(N,sigtpl,zin,bkgsB)
            testdct[datatype]["input"]={"sigmus":list(sigmus)} # save input signal strengths
            ## Nsigma test for BkgOnly - copy
            testdct[datatype]['nsigma_bkg']['scores']=scores_nsigma_bkg
            testdct[datatype]['nsigma_bkg']['pvalues']=pvalues_nsigma_bkg
            testdct[datatype]['nsigma_bkg']['signifs']=signifs_nsigma_bkg
            testdct[datatype]['nsigma_bkg']['fitargs']=fitargs_nsigma_bkg
            ## Nsigma test for Bkg+Sig
            scores,pvalues,signifs,_=perform_nsigma_test(bkgsA,sigbkgsB,fitargs_bkg=fitargs_nsigma_bkg)
            testdct[datatype]['nsigma_sigbkg']['scores']=scores
            testdct[datatype]['nsigma_sigbkg']['pvalues']=pvalues
            testdct[datatype]['nsigma_sigbkg']['signifs']=signifs
            ## Delta test for BkgOnly - copy
            testdct[datatype]['delta_bkg']['scores']=scores_delta_bkg
            testdct[datatype]['delta_bkg']['pvalues']=pvalues_delta_bkg
            testdct[datatype]['delta_bkg']['signifs']=signifs_delta_bkg
            testdct[datatype]['delta_bkg']['fitargs']=fitargs_delta_bkg
            ## Delta test for Bkg+Sig
            scores,pvalues,signifs,_=perform_delta_test(bkgsA,sigbkgsB,fitargs_bkg=fitargs_delta_bkg)
            testdct[datatype]['delta_sigbkg']['scores']=scores
            testdct[datatype]['delta_sigbkg']['pvalues']=pvalues
            testdct[datatype]['delta_sigbkg']['signifs']=signifs
            # ## Likelihood test for BkgOnly
            # scores,pvalues,signifs,sigmus=perform_lik_test(tplbkgsA,bkgsB,sigtpl)
            # testdct[datatype]['lik_bkg']['scores']=scores
            # testdct[datatype]['lik_bkg']['pvalues']=pvalues
            # testdct[datatype]['lik_bkg']['signifs']=signifs
            # testdct[datatype]['lik_bkg']['sigmus']=sigmus
            # ## Likelihood test for Bkg+Sig
            # scores,pvalues,signifs,sigmus=perform_lik_test(tplbkgsA,sigbkgsB,sigtpl)
            # testdct[datatype]['lik_sigbkg']['scores']=scores
            # testdct[datatype]['lik_sigbkg']['pvalues']=pvalues
            # testdct[datatype]['lik_sigbkg']['signifs']=signifs
            # testdct[datatype]['lik_sigbkg']['sigmus']=sigmus
            print("Done (%.1fs)"%(timeit.default_timer()-startT))
    ## Save results to file
    if saveas.endswith("json"):
        savejsonfile(testdct,saveas)
    return testdct

def make_plots(tpldct,testdct,zin):
    # ## ----- sigmu_input pdfs
    # for bkgtype in bkgtypes:
    #     sigmus=[]
    #     labels=[]
    #     for sigtype in sigtypes:
    #         if bkgtype=='flat' and 'highstat' in sigtype:
    #             continue
    #         datatype="%s_%s"%(bkgtype,sigtype)
    #         sigmus.append(testdct[datatype]['input']['sigmus'])
    #         labels.append(datatype)
    #     hplot=HistsPlot(sigmus,labels,"Signal Strengths - bkg %s - sig %s Z=%s"%(bkgtype,sigtype,zin))
    #     hplot.nbins=1000
    #     if showplots:
    #         hplot.showplot()
    #     else:
    #         hplot.saveplot("plots/sigmus_%s_Z%s.png"%(datatype,zin))
    # exit()

    # ## ----- Score PDFs b vs b+s
    # labels=["bkg only","bkg+sig"]
    # for pdftype in pdftypes:
    #     for testtype in testtypes:
    #         for bkgtype in bkgtypes:
    #             for sigtype in sigtypes:
    #                 if bkgtype=='flat' and 'highstat' in sigtype:
    #                     continue
    #                 ## PATCH - only for one bkg+sig type
    #                 if bkgtype!=bkgtypes[0] or sigtype!=sigtypes[0]:
    #                     continue
    #                 ## END
    #                 datatype="%s_%s"%(bkgtype,sigtype)
    #                 for histtype in ['pdf','cumulpdf']:
    #                     pdfs=[testdct[datatype]["%s_bkg"%testtype][pdftype],
    #                            testdct[datatype]["%s_sigbkg"%testtype][pdftype]]
    #                     testlabel='NSigma' if testtype=='nsigma' else 'Likelihood'
    #                     if histtype=='cumulpdf': # draw side by side
    #                         tpdfs=[pdfs]
    #                         tlabels=[labels]
    #                         documul=1 if pdftype=='pvalues' else -1
    #                         donorm=True
    #                         legloc='lower right' if pdftype=='pvalues' else 'upper right'
    #                     else:
    #                         tpdfs=pdfs+[]
    #                         tlabels=labels
    #                         documul=False
    #                         donorm=False
    #                         legloc='upper right'
    #                     plt.close()
    #                     print(pdftype,testtype,datatype)
    #                     hplot=HistsPlot(tpdfs,tlabels,"%s %s - %s - bkg %s - sig %s Z=%s"%(testlabel,histtype,pdftype,bkgtype,sigtype,zin))
    #                     hplot.cumulative=documul
    #                     hplot.normed=donorm
    #                     hplot.legloc=legloc
    #                     # if testtype=='lik':
    #                     #     # hplot.nbins=100
    #                     #     hplot.logy=True
    #                     if testtype=='nsigma' and pdftype=='scores' and histtype=='pdf': # get fits
    #                         hplot.make_hists()
    #                         bins=hplot.bincenters
    #                         popt=testdct[datatype]["%s_bkg"%testtype]['fitargs']
    #                         hplot.fits.append(myGauss(bins,*(np.array(popt))))
    #                         hplot.make_fits()
    #                     if showplots:
    #                         hplot.showplot()
    #                     else:
    #                         hplot.saveplot("plots/%sbbs_%s_%s_%s_Z%s.png"%(histtype,testtype,pdftype,datatype,zin))

    ## ----- Scores PDFs comp bkgs or sigbkgs
    testlabdct={'nsigma':'NSigma','delta':'Delta','lik':'Likelihood'}
    rocdct={}
    for pdftype in pdftypes+['sigmus']:
        rocdct[pdftype]={}
        for testtype in testtypes:
            if pdftype=='sigmus':
                continue
            if pdftype=='sigmus' and testtype!='lik':
                continue
            rocdct[pdftype][testtype]={}
            testlabel=testlabdct[testtype]
            pdfs_input=[]
            labels_input=[]
            pdfs_bkg=[]
            labels_bkg=[]
            pdfs_sigbkg=[]
            labels_sigbkg=[]
            for bkgtype in bkgtypes:
                for sigtype in sigtypes:
                    if bkgtype=='flat' and 'highstat' in sigtype:
                        continue
                    datatype="%s_%s"%(bkgtype,sigtype)
                    rocdct[pdftype][testtype][datatype]={}
                    pdfs_bkg.append(testdct[datatype]["%s_bkg"%testtype][pdftype])
                    labels_bkg.append(datatype+"_bkg")
                    pdfs_sigbkg.append(testdct[datatype]["%s_sigbkg"%testtype][pdftype])
                    labels_sigbkg.append(datatype+"_sigbkg")
                    if pdftype=='sigmus':
                        pdfs_input.append(testdct[datatype]['input']['sigmus']['tot'])
                        labels_input.append(datatype+"_input")
            for histtype in ['pdf','cumulpdf']:
                if pdftype=='sigmus' and histtype=='cumulpdf':
                    continue
                if histtype=='cumulpdf':
                    documul=1 if pdftype=='pvalues' else -1
                    donorm=True
                    legloc='lower right' if pdftype=='pvalues' else 'upper right'
                else:
                    documul=False
                    donorm=False
                    legloc='upper right'

                ## All together
                print(pdftype,testtype)
                hplot=HistsPlot(pdfs_bkg+pdfs_sigbkg+pdfs_input,labels_bkg+labels_sigbkg+labels_input,"%s %s - %s"%(testlabel,histtype,pdftype))
                hplot.histtype='step'
                hplot.cumulative=documul
                hplot.normed=donorm
                if pdftype=='sigmus':
                    hplot.nbins=200
                    hplot.ylim=[0,150]
                hplot.make_hists()
                hplot.legloc=legloc
                # if testtype=='lik':
                #     hplot.logy=True
                ## Get binvals for ROC curves
                binvals=np.copy(hplot.binvals)
                if histtype=='cumulpdf':
                    for datatype in rocdct[pdftype][testtype]:
                        for i,lab in enumerate(labels_bkg):
                            if datatype in lab:
                                rocdct[pdftype][testtype][datatype]=[binvals[i],binvals[i+len(labels_bkg)]]
                                break
                if showplots:
                    hplot.showplot()
                else:
                    hplot.saveplot("plots/%s_%s_%scomp.png"%(histtype,testtype,pdftype))
                # ## bkg comp plot
                # hplot_bkg=HistsPlot(pdfs_bkg,labels_bkg,"%s %s - %s - bkg only"%(testlabel,histtype,pdftype))
                # hplot_bkg.histtype='step'
                # hplot_bkg.cumulative=documul
                # hplot_bkg.normed=donorm
                # hplot_bkg.legloc=legloc
                # # if testtype=='lik':
                # #     hplot_bkg.logy=True
                # if showplots:
                #     hplot_bkg.showplot()
                # else:
                #     hplot_bkg.saveplot("plots/%s_%s_%s_bkgcomp.png"%(histtype,testtype,pdftype))
                # ## sig comp plot
                # hplot_sigbkg=HistsPlot(pdfs_sigbkg,labels_sigbkg,"%s %s - %s - bkg + sig Z=%s"%(testlabel,histtype,pdftype,zin))
                # hplot_sigbkg.histtype='step'
                # hplot_sigbkg.cumulative=documul
                # hplot_sigbkg.normed=donorm
                # hplot_sigbkg.legloc=legloc
                # # if testtype=='lik':
                # #     hplot_sigbkg.logy=True
                # if showplots:
                #     hplot_sigbkg.showplot()
                # else:
                #     hplot_sigbkg.saveplot("plots/%s_%s_%s_sigbkgcomp_Z%s.png"%(histtype,testtype,pdftype,zin))
    
    # ## ----- Make ROC curves per pdftypes
    # labels=["bkg only","bkg+sig"]
    # rocdct={}
    # for bkgtype in bkgtypes:
    #     for sigtype in sigtypes:
    #         if bkgtype=='flat' and 'highstat' in sigtype:
    #             continue
    #         datatype="%s_%s"%(bkgtype,sigtype)
    #         rocdct[datatype]={}
    #         for pdftype in pdftypes:
    #             rocdct[datatype][pdftype]={}
    #             for testtype in testtypes:
    #                 ## Get cumul pdfs
    #                 pdfs=testdct[datatype][testtype][pdftype]
    #                 hplot=HistsPlot(pdfs,labels)
    #                 hplot.histtype='step'
    #                 hplot.cumulative=1 if pdftype=='pvalues' else -1
    #                 hplot.normed=True
    #                 if pdftype=='pvalues':
    #                     hplot.make_axes()
    #                     hplot.binedges=sorted(addarrays([np.zeros((1)),np.logspace(-22,-1,200),np.linspace(0.1,1,200),hplot.binedges],unique=True))
    #                 # print(datatype,testtype,pdftype)
    #                 hplot.make_hists()
    #                 # hplot.showplot()
    #                 # hplot.saveplot("%s_%s_%s.png"%(datatype,testtype,pdftype))
    #                 # plt.close()
    #                 fprs,tprs=hplot.binvals
    #                 tmp([hplot.binvals],"%s_%s_%s.png"%(datatype,testtype,pdftype))
    #                 rocdct[datatype][pdftype][testtype]=[fprs,tprs]
    #             # ## Get ROC curve
    #             # ## PATCH - only for one bkg+sig type
    #             # if bkgtype!=bkgtypes[0] or sigtype!=sigtypes[0]:
    #             #     continue
    #             # dct=rocdct[datatype][pdftype]
    #             # points=[[np.linspace(0,1,2),np.linspace(0,1,2)],dct['nsigma'],dct['lik']]
    #             # styles=['k--','b+-','r+-']
    #             # labels=['','nsigma','lik']
    #             # title="ROC from %s - %s Z%s"%(pdftype,datatype,zin)
    #             # cplot=CurvePlot(points,labels,title,styles)
    #             # cplot.legloc='lower right'
    #             # if showplots:
    #             #     cplot.showplot()
    #             # else:
    #             #     cplot.saveplot("plots/rocsimple_%s_%s_Z%s.png"%(pdftype,datatype,zin))
    
    ## ----- Make ROC comp plots
    points=[[np.linspace(0,1,2),np.linspace(0,1,2)]]
    styles=['k--']
    labels=['']
    pdftype='signifs' if 'signifs' in pdftypes else pdftypes[0]
    title="ROC from %s - Z%s"%(pdftype,zin)
    for testtype in testtypes:
        for sigt in ['rect','gaus']:
            for datatype in rocdct[pdftype][testtype]:
                if sigt not in datatype:
                    continue
                # print(datatype,pdftype,testtype)
                # print(rocdct[datatype][pdftype][testtype])
                points+=[rocdct[pdftype][testtype][datatype]]
                styles+=['+-']
                labels+=["%s_%s"%(testtype,datatype)]
    cplot=CurvePlot(points,labels,title,styles)
    cplot.legloc='lower right'
    if showplots:
        cplot.showplot()
    else:
        cplot.saveplot("plots/roccomp_Z%s.png"%zin)

if __name__=="__main__":
    main()
