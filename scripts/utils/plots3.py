import os
import numpy as np
import matplotlib.pyplot as plt
from utils3 import *

class MatrixPlot():
    def __init__(self,arr,title=''):
        plt.close()
        self.arr=np.transpose(arr)
        self.title=title
        self.origin='lower'
        self.xlim=[30,170]
        self.ylim=[10,150]
    def make_axes(self):
        self.extent=[self.xlim[0],self.xlim[1],self.ylim[0],self.ylim[1]]
    def getplot(self):
        self.make_axes()
        plt.imshow(self.arr,extent=self.extent,origin=self.origin)
        plt.title(self.title)
        plt.colorbar()
        plt.tight_layout()
    def showplot(self):
        self.getplot()
        plt.show()
        plt.close()
    def saveplot(self,savename):
        self.getplot()
        savedir=savename.rsplit('/',1)[0] if '/' in savename else '.'
        if not os.path.isdir(savedir):
            print("Creating new directory:",savedir)
            os.makedirs(savedir)
        plt.savefig(savename)
        print("%s created"%savename)
        plt.close()

class HistsPlot():
    def __init__(self,hists,labels='',title=''):
        plt.close()
        self.hists=hists
        self.title=title
        self.labels=labels
        self.legloc='upper right'
        self.nbins=50
        self.binedges='' # if specified overwrites nbins
        self.xlim=''
        self.ylim=''
        self.logy=False
        self.fits=[]
        self.cumulative=False
        self.normed=False
        # self.histtype='bar'
        self.histtype='step'
        self.gotaxes=False
        self.gothists=False
    def make_axes(self):
        self.allvals=addarrays(self.hists)
        ## Get Xaxis bins and range
        if any(self.binedges):
            self.nbins=len(self.binedges)-1
            self.xlim=[self.binedges[0],self.binedges[-1]]
        else:
            if not self.xlim: # find xaxis range
                xmin=np.min(self.allvals)
                xmax=np.max(self.allvals)
                diff=xmax-xmin
                offset=0.1
                self.xlim=[xmin-offset*diff,xmax+offset*diff]
            self.binedges=np.linspace(self.xlim[0],self.xlim[1],self.nbins+1)
        self.bincenters=[self.binedges[i]+(self.binedges[i+1]-self.binedges[i])/2 for i in range(self.nbins)]
        self.gotaxes=True
    def make_hists(self):
        if not self.gotaxes:
            self.make_axes()
        self.binvals=[]
        for i,h in enumerate(self.hists):
            lab=self.labels[i] if self.labels else ''
            alpha=.8 if i==0 else .6
            tbinvals,bla,bla1=plt.hist(h,bins=self.binedges,label=lab,alpha=alpha,
                                           cumulative=self.cumulative,
                                           density=self.normed,
                                           histtype=self.histtype,
                                           )
            if self.ylim:
                plt.ylim(self.ylim)
            self.binvals.append(tbinvals)
        self.gothists=True
    def make_fits(self):
        for f in self.fits:
            plt.plot(self.bincenters,f,'-',alpha=0.5)
    def getplot(self):
        if not self.gotaxes:
            self.make_axes()
        if not self.gothists:
            self.make_hists()
        if self.logy:
            plt.yscale('log',nonposy='clip')
        plt.title(self.title)
        if self.labels:
            plt.legend(loc=self.legloc)
        plt.tight_layout()
    def showplot(self):
        self.getplot()
        plt.show()
        plt.close()
    def saveplot(self,savename):
        self.getplot()
        savedir=savename.rsplit('/',1)[0] if '/' in savename else '.'
        if not os.path.isdir(savedir):
            print("Creating new directory:",savedir)
            os.makedirs(savedir)
        plt.savefig(savename)
        print("%s created"%savename)
        plt.close()

class CurvePlot():
    def __init__(self,points,labels='',title='',styles=''):
        plt.close()
        self.points=points
        self.title=title
        self.labels=labels
        self.styles=styles
        self.legloc='upper right'
        self.xlim=''
        self.ylim=''
        self.logy=False
        self.fits=[]
        self.gotaxes=False
        self.gotcurves=False
    def make_axes(self):
        self.gotaxes=True
    def make_curves(self):
        if not any(self.labels):
            self.labels=['' for k in self.points]
        if not self.styles:
            self.styles=['' for k in self.points]
        for i,[xs,ys] in enumerate(self.points):
            plt.plot(xs,ys,self.styles[i],label=self.labels[i])
        self.gotcurves=True
    def make_fits(self):
        for f in self.fits:
            plt.plot(self.bincenters,f,'-',alpha=0.5)
    def getplot(self):
        if not self.gotaxes:
            self.make_axes()
        if not self.gotcurves:
            self.make_curves()
        if self.logy:
            plt.yscale('log',nonposy='clip')
        plt.title(self.title)
        if self.labels:
            plt.legend(loc=self.legloc)
        plt.tight_layout()
    def showplot(self):
        self.getplot()
        plt.show()
        plt.close()
    def saveplot(self,savename):
        self.getplot()
        savedir=savename.rsplit('/',1)[0] if '/' in savename else '.'
        if not os.path.isdir(savedir):
            print("Creating new directory:",savedir)
            os.makedirs(savedir)
        plt.savefig(savename)
        print("%s created"%savename)
        plt.close()
        
