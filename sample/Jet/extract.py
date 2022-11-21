#!/usr/bin/env python
# coding: utf-8
# python extract.py <kappa value> <root_file_path>
# selection: The events containg 1 vector boson satisfies the conditions are included

import sys
import re
import os

import numpy as np
import pandas as pd
import ROOT as r
from tqdm import tqdm

r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/')
r.gROOT.ProcessLine('.include /usr/local/Delphes-3.4.2/external/')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/classes/DelphesClasses.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTreeReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootConfReader.h"')
r.gInterpreter.Declare('#include "/usr/local/Delphes-3.4.2/external/ExRootAnalysis/ExRootTask.h"')
r.gSystem.Load("/usr/local/Delphes-3.4.2/install/lib/libDelphes")

def DeltaR(eta1,phi1, eta2,phi2):
    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5
    return dR

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

def sample_selection(File, kappa, pbar, imageWriter):
    n_sample = 0
    # storing particle type, jet mass, jet charge
    data = []
    for event_id, event in tqdm(enumerate(File)):
        
        # pt in (350, 450) GeV & |eta| < 1 for 1 jets
        Jets = []
        for jet in event.Jet:
            if abs(jet.PT - 400) > 50 or abs(jet.Eta) > 1:
                continue
            Jets.append(jet)

        if len(Jets) < 1:
            pbar.update(1)
            continue
        
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
            pbar.update(1)
            continue

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
        
            image = np.array([np.array(json_obj['particle_type']), hpT, hQk], dtype=object)
            np.save(imageWriter, image)
            
            data.append([json_obj['particle_type'], jet.Mass, sum(Q_kappas)])
            
            n_sample += 1
            
        pbar.update(1)
        
    return n_sample, data
     
def main():
    # the input root files should be same type
    kappa = float(sys.argv[1])
    
    # read all root files. 
    chain = r.TChain('Delphes')
    for rootfile in sys.argv[2:]:
        chain.Add(rootfile)
        
    nevent = chain.GetEntries()
    
    output_dir = '/'
    for text in sys.argv[2].split('/'):
        match = re.match('VBF_H5(pp_ww|mm_ww|z_zz)_jjjj', text)
        if match:
            input_file_name = match.group()
            break
        output_dir = os.path.join(output_dir, text)

    output_dir = os.path.join(output_dir, f'samples_kappa{kappa}-{nevent//1000}k')
    
    # create output directory
    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir, 0o777)
    
    image_name = os.path.join(output_dir, f'{input_file_name}.npy')
    count_name = os.path.join(output_dir, f'{input_file_name}.count')
    property_file = os.path.join(output_dir, f'{input_file_name}_properties.txt')
        
    with tqdm(total=chain.GetEntries()) as pbar, open(image_name, 'wb') as imagewriter:
        n_sample, data = sample_selection(chain, kappa, pbar, imagewriter)
        
    with open(count_name, 'w') as f:
        f.write(f'{n_sample}')
        
    # write jet mass & Qk to _properties.txt
    data = np.array(data)
    df = pd.DataFrame([])
    df['particle type'] = data[:,0]
    df['jet mass'] = data[:,1].astype(float)
    df['jet charge'] = data[:,2].astype(float)
    df.to_csv(property_file)  
        
if __name__ == '__main__':
    main()