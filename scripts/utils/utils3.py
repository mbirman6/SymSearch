import os
import pprint
import json
import numpy as np

def csv2np(f):
    with open(path, 'r') as f:
        reader=csv.reader(f,delimiter=',')
        headers=next(reader)
        data=np.array(list(reader)).astype(float)
    return headers,data

def printkeys(dct):
    print(list(dct.keys()))

def totsum(arr):
    return np.sum(arr)

def addarrays(arrs,unique=False):
    add=np.copy(arrs[0])
    for i,arr in enumerate(arrs):
        if i==0:
            continue
        add=np.concatenate((add,arr))
    if unique:
        add=np.unique(add)
    return add

def myGauss(x,mean,std,amp):
    return amp*np.exp(-((x-mean)**2/(2*std**2)))

def Calcq0MinusZ(mu,B,S,z):
    q0=-2*totsum(mu*S-(B+mu*S)*np.log(1+mu*S/B))
    return q0-z**2

def CalcMinusq0(mu,B,T,S):
    q0=-2*totsum(mu*S-B*np.log(1+mu*S/T))
    return -q0
    
# def CalcMinusq0(mu,X,R,S):
#     equation_value=2*totsum(R-X + mu*S + R*np.log((R+X)/(2*R)) + X*np.log((R+X)/(2*(R+mu*S))))
#     return equation_value
    
def NSigma(test,ref):
    # return (test-ref)/np.sqrt(test+ref)
    return (test-ref)/(test+ref)

def pretty_print(obj,title='',Indent=2):
    if title:
        info1(title)
    pp=pprint.PrettyPrinter(indent=Indent)
    pp.pprint(obj)
    print

def print_table(rows):
    print
    rows=[[str(val) for val in row] for row in rows]
    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))
    print()

def savejsonfile(obj,pathfile,indent=2,sort=True):
    with open(pathfile,"w") as f:
        json.dump(obj,f,indent=indent,sort_keys=sort)
    
def loadjsonfile(pathfile):
    if not os.path.isfile(pathfile):
        f=open(pathfile,'w')
        f.write('{}')
        f.close()
    with open(pathfile) as f:
        return byteify(json.load(f))

def byteify(input):
   # Encodes any input from unicode to normal string
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.items()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, str):
        try:
            return input.decode()#'utf-8')
        except:
            return input
    else:
        return input
