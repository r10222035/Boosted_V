#!/usr/bin/env python
# coding: utf-8
# python plot_from_root.py 

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import colors

def get_mass(sample_dir):
    sample_name = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']
    m = [[], [], []]
    N = [0,0,0]
    for i in range(3):
        with open(os.path.join(sample_dir, f'{sample_name[i]}.count'), 'r') as f:
            N[i] = int(f.read())
        print(f'Get mass from {sample_dir}')
        with open(os.path.join(sample_dir, f'{sample_name[i]}_properties.txt'), 'r') as f:
            f.readline()
            for line in tqdm(range(N[i])):
                data = f.readline().strip().split(',')[2:]
                m[i].append(float(data[0]))               
    return m

def get_Qk(sample_dir):
    sample_name = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']
    Q = [[], [], []]
    N = [0,0,0]
    for i in range(3):
        with open(os.path.join(sample_dir, f'{sample_name[i]}.count'), 'r') as f:
            N[i] = int(f.read())
        print(f'Get Qk from {sample_dir}')
        with open(os.path.join(sample_dir, f'{sample_name[i]}_properties.txt'), 'r') as f:
            f.readline()
            for line in tqdm(range(N[i])):
                data = f.readline().strip().split(',')[2:]
                Q[i].append(float(data[1]))               
    return Q

def get_jet_image(sample_dir):
    sample_name = ['VBF_H5pp_ww_jjjj', 'VBF_H5mm_ww_jjjj', 'VBF_H5z_zz_jjjj']
    pT = [0,0,0]
    Qk = [0,0,0]
    N = [0,0,0]    
    for i in range(3):
        with open(os.path.join(sample_dir, f'{sample_name[i]}.count'), 'r') as f:
            N[i] = int(f.read())
        print(f'Get jet image from {sample_dir}')  
        with open(os.path.join(sample_dir, f'{sample_name[i]}.npy'),'rb') as f:
            for fig_id in tqdm(range(N[i])):
                data = np.load(f, allow_pickle=True)[1:]
                pT[i] += data[0]
                Qk[i] += data[1]    
    return {'pT': pT,'Qk': Qk, 'N': N}

def plot_mass_distribution(m):
    fig, ax = plt.subplots(1,1, figsize=(6,5))
    ax.hist(m[0], color='b', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^+$')
    ax.hist(m[1], color='r', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^-$')
    ax.hist(m[2], color='k', histtype='step', density=True, bins=100, range=[60,120], label=r'$Z$')

    ax.legend()
    ax.set_xlim([60, 120])
    ax.set_ylim([0, 0.06])

    ax.set_xlabel('Mass (GeV)')
    plt.savefig('figures/jet_mass_distribution', facecolor='White', dpi=300)

def plot_charge_distribution(Qk_kappa_list):
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    for i, (Qk, kappa) in enumerate(Qk_kappa_list):

        ax[i].hist(Qk[0], color='b', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^+$')
        ax[i].hist(Qk[1], color='r', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^-$')
        ax[i].hist(Qk[2], color='k', histtype='step', density=True, bins=100, range=[-2,2], label=r'$Z$')

        ax[i].legend()
        ax[i].set_xlim([-2,2])

        ax[i].set_xlabel(r'$\mathcal{Q}_\kappa$')

        ax[i].set_title(f'Jet charge ($\kappa = {kappa}$)')

    plt.savefig('figures/jet_charge_distribution', facecolor='White', dpi=300)
        
def plot_jet_image(jet_image):
    N = jet_image['N']
    pT = jet_image['pT']
    Qk = jet_image['Qk']
    particle_type = ['W^+', 'W^-', 'Z']
          
    # PT distribution
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    for i in range(3): 
        # P_T distribution
        imag = ax[i].imshow(pT[i]/N[i], origin='upper', extent=[-0.8, 0.8, -0.8, 0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6 ,vmin=0), cmap='Blues')
        plt.colorbar(imag, ax=ax[i], shrink=0.9)
        ax[i].set_title(f'$\\langle {particle_type[i]}\\rangle: p_T$ channel')
        ax[i].set_xlabel('$\phi^\prime$')
        ax[i].set_ylabel('$\eta^\prime$')
        
    plt.savefig('figures/jet_image_PT', facecolor='White', dpi=300)

    # Qk distribution
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    for i in range(3):
        imag = ax[i].imshow(Qk[i]/N[i], origin='upper', extent=[-0.8, 0.8, -0.8, 0.8], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
        plt.colorbar(imag, ax=ax[i], shrink=0.9)
        ax[i].set_title(f'$\\langle {particle_type[i]}\\rangle: Q_\\kappa$ channel')
        ax[i].set_xlabel('$\phi^\prime$')
        ax[i].set_ylabel('$\eta^\prime$')
        
    plt.savefig('figures/jet_image_Qk', facecolor='White', dpi=300)
    
    # "Z - W+" PT distribution
    fig, ax = plt.subplots(1,1, figsize=(5,5))    
    imag = ax.imshow(pT[2]/N[2] - pT[0]/N[0], origin='upper', extent=[-0.8, 0.8, -0.8, 0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6), cmap='coolwarm_r')
    plt.colorbar(imag, ax=ax, shrink=0.9)
    ax.set_title(r'$\langle Z \rangle-\langle W^+\rangle: p_T$ channel')
    ax.set_xlabel('$\phi^\prime$')
    ax.set_ylabel('$\eta^\prime$')

    plt.savefig('figures/jet_image_PT_Z-W+', facecolor='White', dpi=300)

def main():
    
    m = get_mass(sample_dir='/home/r10222035/boosted_V_ML_test/sample/samples_kappa0.15-1000k-J/')
    plot_mass_distribution(m)
    
    Qk02 = get_Qk(sample_dir='/home/r10222035/boosted_V_ML_test/sample/samples_kappa0.2-1000k-J/')
    Qk03 = get_Qk(sample_dir='/home/r10222035/boosted_V_ML_test/sample/samples_kappa0.3-1000k-J/')
    Qk06 = get_Qk(sample_dir='/home/r10222035/boosted_V_ML_test/sample/samples_kappa0.6-1000k-J/')
    plot_charge_distribution([(Qk02, 0.2), (Qk03, 0.3), (Qk06, 0.6)])
    
    jet_image = get_jet_image(sample_dir='/home/r10222035/boosted_V_ML_test/sample/samples_kappa0.15-1000k-J/')
    plot_jet_image(jet_image)   

if __name__ == '__main__':
    main()