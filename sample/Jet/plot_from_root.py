#!/usr/bin/env python
# coding: utf-8
# python plot_from_root.py <kappa value> <root_file_path>
# python plot_from_root.py /home/r10222035/boosted_V_ML_test/sample/VBF_H5pp_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/r10222035/boosted_V_ML_test/sample/VBF_H5mm_ww_jjjj/Events/run_01/tag_1_delphes_events.root /home/r10222035/boosted_V_ML_test/sample/VBF_H5z_zz_jjjj/Events/run_01/tag_1_delphes_events.root

import sys
import os

import ROOT as r
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/')
r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/external/')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/classes/DelphesClasses.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTreeReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootConfReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTask.h"')
r.gSystem.Load("/usr/local/Delphes-3.4.2/build/libDelphes")
r.gSystem.Load("/usr/local/lib/libDelphes")

def DeltaR(eta1,phi1, eta2,phi2):
    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR

def PtEtaPhiM(px, py, pz, e):
    E, px ,py, pz = e, px, py, pz  
    P = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    eta = 1./2.*np.log((P + pz)/(P - pz))
    phi = np.arctan(py/px)
    m = np.sqrt(np.sqrt((E**2 - px**2 - py**2 - pz**2)**2))

    return pt, eta, phi, m

def std_phi(phi):
    if phi > np.pi:
        return phi - 2.*np.pi
    elif phi < -np.pi:
        return phi + 2.*np.pi
    else:
        return phi
        
def preprocess(jet, constituents, kappa):
    pt_quadrants = [0., 0., 0., 0.]
    eta_flip, phi_flip = 1., 1.
    pts, etas, phis, Q_kappas = [], [], [], []
    
    for consti in constituents:
        try:
            pts.append(consti.PT)
            etas.append(consti.Eta)
            phis.append(consti.Phi)
            Q_kappas.append((consti.Charge)*(consti.PT)**kappa/(jet.PT)**kappa)
        except:
            pts.append(consti.ET)
            etas.append(consti.Eta)
            phis.append(consti.Phi)
            Q_kappas.append(0.)
            
    pts = np.array(pts)
    etas = np.array(etas)
    phis = np.array(phis)

    v_std_phi = np.vectorize(std_phi)
    if np.var(phis) > 0.5:
        phis += np.pi
        phis = v_std_phi(phis)

    eta_central = (pts*etas).sum()/pts.sum()
    phi_central = (pts*phis).sum()/pts.sum()       

    s_etaeta = (pts * (etas - eta_central)**2).sum() / pts.sum()
    s_phiphi = (pts * (phis - phi_central)**2).sum() / pts.sum()
    s_etaphi = (pts * (etas - eta_central) * (phis - phi_central)).sum() / pts.sum()

    angle = -np.arctan((-s_etaeta + s_phiphi + np.sqrt((s_etaeta - s_phiphi)**2 + 4. * s_etaphi**2))/(2. * s_etaphi))
    
    eta_shift, phi_shift = etas - eta_central, v_std_phi(phis - phi_central)
    eta_rotat, phi_rotat = eta_shift * np.cos(angle) - phi_shift * np.sin(angle), phi_shift * np.cos(angle) + eta_shift * np.sin(angle)
    
    def quadrant_max(eta, phi, pt):
        if eta > 0. and phi > 0.:
            pt_quadrants[0] += pt
        elif eta > 0. and phi < 0.:
            pt_quadrants[1] += pt
        elif eta < 0. and phi < 0.:
            pt_quadrants[2] += pt
        elif eta < 0. and phi > 0.:
            pt_quadrants[3] += pt
            
    np.vectorize(quadrant_max)(eta_rotat, phi_rotat, pts)

    if np.argmax(pt_quadrants) == 1:
        phi_flip = -1.
    elif np.argmax(pt_quadrants) == 2:
        phi_flip = -1.
        eta_flip = -1.
    elif np.argmax(pt_quadrants) == 3:
        eta_flip = -1.

    eta_news = eta_rotat * eta_flip
    phi_news = phi_rotat * phi_flip

    return pts, eta_news, phi_news, Q_kappas

def sample_selection(File, kappa=0.15, max_event=1000000):

    json_data = []
    count = [0,0,0,0]
    for event_id, event in tqdm(enumerate(File)):
        if event_id > max_event:
            break
            
        # pt in (350, 450) GeV & |eta| < 1 for 1 jets
        Jets = []
        for jet in event.Jet:
            if abs(jet.PT - 400) > 50 or abs(jet.Eta) > 1:
                continue
            Jets.append(jet)

        if len(Jets) < 1:
            continue
        count[0] += 1
    
        # Delta R of vector bosons decayed quarks < 0.6
        merging = [False, False]
        for particle in event.Particle:
            if particle.PID in [255, -255, 257]:
                H5 = particle
                while event.Particle[H5.D1].PID == H5.PID:
                    H5 = event.Particle[H5.D1]

                V1, V2 = event.Particle[H5.D1], event.Particle[H5.D2]
                while event.Particle[V1.D1].PID == V1.PID:
                    V1 = event.Particle[V1.D1]
                while event.Particle[V2.D1].PID == V2.PID:
                    V2 = event.Particle[V2.D1]

                q1, q2 = event.Particle[V1.D1], event.Particle[V1.D2]
                q3, q4 = event.Particle[V2.D1], event.Particle[V2.D2]
                if DeltaR(q1.Eta, q1.Phi, q2.Eta, q2.Phi) < 0.6:
                    merging[0] = True
                if DeltaR(q3.Eta, q3.Phi, q4.Eta, q4.Phi) < 0.6:
                    merging[1] = True
                break

        if (not merging[0]) and (not merging[1]):
            continue
        count[1] += 1
    
        # metch jet and vector boson    
        jet_V = []
        for jet in Jets:
            if DeltaR(V1.Eta, V1.Phi, jet.Eta, jet.Phi) < 0.1 and merging[0]:
                jet_V.append((jet, V1))
            if DeltaR(V2.Eta, V2.Phi, jet.Eta, jet.Phi) < 0.1 and merging[1]:
                jet_V.append((jet, V2))

        # generate pT, Qk histograms and save to .npy file
        for jet, V in jet_V:
            
            # pre-process jet constituents
            constituents = [consti for consti in jet.Constituents if consti != 0]
            pt_news, eta_news, phi_news, Q_kappas = preprocess(jet, constituents, kappa)

            hpT, _, _ = np.histogram2d(eta_news, phi_news, range=((-0.8, 0.8), (-0.8, 0.8)), bins=(75, 75), weights=pt_news)
            hQk, _, _ = np.histogram2d(eta_news, phi_news, range=((-0.8, 0.8), (-0.8, 0.8)), bins=(75, 75), weights=Q_kappas)

            json_obj = {'particle_type': '', 'pT': '', 'Qk': '', 'jet_Mass':'', 'jet_Charge':''}

            if V.PID == 24:
                json_obj['particle_type'] = 'W+'
            elif V.PID == -24:
                json_obj['particle_type'] = 'W-'
            elif V.PID == 23:
                json_obj['particle_type'] = 'Z'
            else:
                print('Something wrong')

            json_obj['pT'] = hpT
            json_obj['Qk'] = hQk
            json_obj['jet_Mass'] = jet.Mass
            json_obj['jet_Charge'] = sum(Q_kappas)

            json_data.append(json_obj)

    print(count)

    return json_data

def plot_mass_distribution(json_data):
    mWp, mWm, mZ = [], [], []

    for js in json_data:
        if js['particle_type'] == 'W+':
            mWp.append(js['jet_Mass'])
        elif js['particle_type'] == 'W-':
            mWm.append(js['jet_Mass'])
        elif js['particle_type'] == 'Z':
            mZ.append(js['jet_Mass'])


    fig, ax = plt.subplots(1,1, figsize=(6,5))
    ax.hist(mWp, color='b', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^+$')
    ax.hist(mWm, color='r', histtype='step', density=True, bins=100, range=[60,120], label=r'$W^-$')
    ax.hist(mZ,  color='k', histtype='step', density=True, bins=100, range=[60,120], label=r'$Z$')

    ax.legend()
    ax.set_xlim([60, 120])

    ax.set_xlabel('Mass (GeV)')
    ax.set_ylim([0, 0.065])
    plt.savefig('figures/jet_mass_distribution', facecolor='White', dpi=300)

def plot_charge_distribution(Qk_kappa_list):
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    for i, (Qk, kappa) in enumerate(Qk_kappa_list):
        QkWp, QkWm, QkZ = [], [], []
        for js in Qk:
            if js['particle_type'] == 'W+':
                QkWp.append(js['jet_Charge'])
            elif js['particle_type'] == 'W-':
                QkWm.append(js['jet_Charge'])
            elif js['particle_type'] == 'Z':
                QkZ.append(js['jet_Charge'])

        ax[i].hist(QkWp, color='b', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^+$')
        ax[i].hist(QkWm, color='r', histtype='step', density=True, bins=100, range=[-2,2], label=r'$W^-$')
        ax[i].hist(QkZ,  color='k', histtype='step', density=True, bins=100, range=[-2,2], label=r'$Z$')

        ax[i].legend()
        ax[i].set_xlim([-2,2])

        ax[i].set_xlabel(r'$Q_k$')

        ax[i].set_title(f'Jet charge ($\kappa = {kappa}$)')
        plt.savefig('figures/jet_charge_distribution', facecolor='White', dpi=300)

        
def plot_jet_image(json_data):
    pT = [0,0,0]
    Qk = [0,0,0]
    N = [0,0,0]
    particle_type = ['W^+', 'W^-', 'Z']

    for js in json_data:
        if js['particle_type'] == 'W+':
            N[0] += 1
            pT[0] += js['pT']
            Qk[0] += js['Qk']
        elif js['particle_type'] == 'W-':
            N[1] += 1
            pT[1] += js['pT']
            Qk[1] += js['Qk']
        elif js['particle_type'] == 'Z':
            N[2] += 1
            pT[2] += js['pT']
            Qk[2] += js['Qk']
            
    # PT distribution
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    for i in range(3):
        imag = ax[i].imshow(pT[i]/N[i], extent=[-0.8, 0.8, 0.8, -0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.4,vmin=0), cmap='Blues')
        plt.colorbar(imag, ax=ax[i], shrink=0.9)
        ax[i].set_title(f'$\\langle {particle_type[i]}\\rangle: p_T$ channel')
        ax[i].set_xlabel(r'$\phi^\prime$')
        ax[i].set_ylabel(r'$\eta^\prime$') 
        
    plt.savefig('figures/jet_image_PT', facecolor='White', dpi=300)

    # Qk distribution
    fig, ax = plt.subplots(1,3, figsize=(18,5))
    for i in range(3):
        imag = ax[i].imshow(Qk[i]/N[i], extent=[-0.8, 0.8, 0.8, -0.8], interpolation='nearest', norm=colors.SymLogNorm(linthresh=0.001, linscale=0.6,vmax=5e-2,vmin=-5e-2), cmap='coolwarm_r')
        plt.colorbar(imag, ax=ax[i], shrink=0.9)
        ax[i].set_title(f'$\\langle {particle_type[i]}\\rangle: Q_\\kappa$ channel')
        ax[i].set_xlabel(r'$\phi^\prime$')
        ax[i].set_ylabel(r'$\eta^\prime$')  
        
    plt.savefig('figures/jet_image_Qk', facecolor='White', dpi=300)
    
    # "Z - W+" PT distribution
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    imag = ax.imshow(pT[2]/N[2] - pT[0]/N[0], extent=[-0.8, 0.8, 0.8, -0.8], norm=colors.SymLogNorm(linthresh=0.1, linscale=0.6), cmap='coolwarm_r')
    plt.colorbar(imag, ax=ax, shrink=0.9)
    ax.set_title(r'$\langle Z \rangle-\langle W^+\rangle: p_T$ channel')
    ax.set_xlabel(r'$\phi^\prime$')
    ax.set_ylabel(r'$\eta^\prime$')

    plt.savefig('figures/jet_image_PT_Z-W+', facecolor='White', dpi=300)

def main():
    
    # read all root files. 
    File = r.TChain('Delphes;1')
    for rootfile in sys.argv[1:]:
        File.Add(rootfile)

    N = File.GetEntries()
    print(f'Number of events: {N}')
    
    Qk02 = sample_selection(File, kappa=0.2)
    Qk03 = sample_selection(File, kappa=0.3)
    Qk06 = sample_selection(File, kappa=0.6)
    
    plot_mass_distribution(Qk02)
    plot_charge_distribution([(Qk02, 0.2), (Qk03, 0.3), (Qk06, 0.6)])
    
    json_data = sample_selection(File, kappa=0.15)
    
    plot_jet_image(json_data)
    

if __name__ == '__main__':
    main()