'''
Script that test Trotter error calculations and can be used as a template to reproduce Trotter error  plots in https://arxiv.org/pdf/2210.10189.pdf
'''


#import tequila as tq

from tequila.trotter_err.main_trot import EstTrotErrParal
from tequila.trotter_err.qubtap_utils import GetQubSymsandEigs
from tequila.hamiltonian import QubitHamiltonian
import pickle
import pandas as pd
from matplotlib import pyplot as plt

path_Frags="./data/Frag_Lib/"
path_Hams="./data/ham_lib/"
import openfermion
import numpy as np

def loadFrags(mol,meth,path_Frags=path_Frags):
    '''
    Utility to load pre-calculated fragments.
    Input: mol, string for name of molecule. It can be either h2,lih,beh2,h2o,nh3
    meth: name of Hamiltonian partition method.
    path_Frags: path to location where Hamiltonian fragments are stored.
    '''
    #FileName=path_Frags+meth+'/'+mol+'_'+meth+'Frags'
    FileName=path_Frags+meth+'/'+mol+'_'+meth+'FiltFrags'
    f=open(FileName,'rb')
    dat=pickle.load(f)

    nqubs=dat['n_qubits']
    Frags=dat['grouping']

    f.close()

    return nqubs,Frags

def load_hamiltonian(mol_name,pathham=path_Hams):

    with open(pathham + mol_name + '_fer.bin', 'rb') as f:
        Hf = pickle.load(f)

    return Hf

def BuildNumSpinOps(nqubs):
    #Build the number and spin fermionic operators...
    norb=nqubs//2
    # Sx
    Sx = openfermion.hamiltonians.sx_operator(norb)
    # Sy
    Sy =  openfermion.hamiltonians.sy_operator(norb)
    # Sz
    Sz = openfermion.hamiltonians.sz_operator(norb)
    # N
    # S squared operator
    S_sq = Sx**2+Sy**2+Sz**2
    Nop = openfermion.hamiltonians.number_operator(nqubs)

    return Nop,S_sq,Sz

mols=['h2','lih','beh2','h2o','nh3']
Dictnelecs={}
Dictnelecs['h2']=2
#Dictnelecs['lih']=4
#Dictnelecs['beh2']=6
#Dictnelecs['h2o']=10
#Dictnelecs['nh3']=10

#nelecs=[2,4,6,10,10]
meths=['FC-SI','LR','LR-LCU','GFRO','GFROLCU','SD-GFRO','FRO','FC-LF']

GlobDict={}
GlobDict['mol']=[]
GlobDict['nqubs']=[]
GlobDict['method']=[]
GlobDict['Nfrags']=[]
GlobDict['alpha']=[]
GlobDict['sym_alpha']=[]


for mol in mols:
    for meth in meths:

        hferm=load_hamiltonian(mol)

        nqubs,Frags=loadFrags(mol,meth)
        print("Current molecule:",mol)
        print("Current method:",meth)
        #'Bare' trotter errors...
        alpha_2=EstTrotErrParal(Frags, nqubs)

        #symmetry-projected results...
        if meth=='FC-SI':
            bkHam=openfermion.bravyi_kitaev(hferm)
            tq_bkHam=QubitHamiltonian.from_openfermion(bkHam)

            SymOps,EigSyms=GetQubSymsandEigs(tq_bkHam,nqubs)

            DictSym={}
            DictSym['SymOps']=SymOps
            DictSym['QNumbs']=EigSyms

        else:
            DictSym={}
            nelec=Dictnelecs[mol]
            Nop,S_sq,Sz=BuildNumSpinOps(nqubs)
            DictSym['SymOps']=[Nop,S_sq,Sz]
            DictSym['QNumbs']=[nelec,0,0]

        alpha_2_sym=EstTrotErrParal(Frags, nqubs, SymDict=DictSym)

        GlobDict['mol'].append(mol)
        GlobDict['method'].append(meth)
        GlobDict['nqubs'].append(nqubs)
        GlobDict['Nfrags'].append(len(Frags))
        GlobDict['alpha'].append(alpha_2)
        GlobDict['sym_alpha'].append(alpha_2_sym)

pdResults=pd.DataFrame(GlobDict)
#saving results as json file...
pdResults.to_json('GlobTrotErrRes.json', orient='records')

width = 0.8
Nmols=len(mols)
indices = np.arange(Nmols)

#xticks=[r'H$_2$','LiH',r'BeH$_{2}$',r'H$_{2}$O',r'NH$_{3}$']
xticks=mols

#group results in two sets...
Meths1=['FC-SI','LR','LR-LCU','GFRO','GFROLCU','SD-GFRO']
Meths2=['FRO','FC-LF']

fig, axs = plt.subplots(2, 2)
#Plot 1...
for meth in Meths2:
    meth_rows=pdResults.loc[pdResults['method'] == meth]

    axs[0,0].scatter(np.arange(Nmols),meth_rows['alpha'],label=meth)
    axs[0,0].plot(np.arange(Nmols),meth_rows['alpha'])


axs[0, 0].set_xticks(np.arange(Nmols))
axs[0, 0].set_xticklabels(xticks,fontsize=12)


axs[0,0].legend(fontsize=4,loc='upper left')
axs[0,0].set_title('Trotter errors',fontsize=12)
axs[0,0].set_ylabel(r'$\alpha$',fontsize=18)


#Plot 2...
for meth in Meths1:
    meth_rows=pdResults.loc[pdResults['method'] == meth]

    axs[1,0].scatter(np.arange(Nmols),meth_rows['alpha'],label=meth)
    axs[1,0].plot(np.arange(Nmols),meth_rows['alpha'])

axs[1, 0].set_xticks(np.arange(Nmols))
axs[1, 0].set_xticklabels(xticks,fontsize=12)


axs[1,0].legend(fontsize=4,loc='upper left')
axs[1,0].set_title('Trotter errors',fontsize=12)
axs[1,0].set_ylabel(r'$\alpha$',fontsize=18)

#Plot 3...
for meth in Meths2:
    meth_rows=pdResults.loc[pdResults['method'] == meth]

    axs[0,1].scatter(np.arange(Nmols),meth_rows['sym_alpha'],label=meth)
    axs[0,1].plot(np.arange(Nmols),meth_rows['sym_alpha'])


axs[0, 1].set_xticks(np.arange(Nmols))
axs[0, 1].set_xticklabels(xticks,fontsize=12)

axs[0,1].legend(fontsize=4,loc='upper left')
axs[0,1].set_title('Symmetry Projected Trotter errors',fontsize=10)
axs[0,1].set_ylabel(r'$\alpha$',fontsize=18)

#Plot 4...
for meth in Meths1:
    meth_rows=pdResults.loc[pdResults['method'] == meth]

    axs[1,1].scatter(np.arange(Nmols),meth_rows['sym_alpha'],label=meth)
    axs[1,1].plot(np.arange(Nmols),meth_rows['sym_alpha'])


axs[1, 1].set_xticks(np.arange(Nmols))
axs[1, 1].set_xticklabels(xticks,fontsize=12)


axs[1,1].legend(fontsize=4,loc='upper left')
axs[1,1].set_title('Symmetry Projected Trotter errors',fontsize=10)
axs[1,1].set_ylabel(r'$\alpha$',fontsize=18)



fig.tight_layout()
plt.savefig("./AlphasPlots.pdf",bbox_inches='tight', dpi=150)
