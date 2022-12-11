#!/usr/bin/env python
# coding: utf-8
# python extract-resolution-resize.py <kappa value> <resolution> <root_file_path>
# python extract-resolution-resize.py 0.15 25 <root_file_path>
# extract full event training data

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
        
def preprocess(constituents, kappa):
    # No eta shift
    # No rotation
    pts, etas, phis, Q_kappas = [], [], [], []
    
    for consti in constituents:
        try:
            pts.append(consti.PT)
            etas.append(consti.Eta)
            phis.append(consti.Phi)
            Q_kappas.append((consti.Charge)*(consti.PT)**kappa)
        except:
            pts.append(consti.ET)
            etas.append(consti.Eta)
            phis.append(consti.Phi)
            Q_kappas.append(0.)
            
    pts = np.array(pts)
    etas = np.array(etas)
    phis = np.array(phis)
    Q_kappas = np.array(Q_kappas) / pts.sum()**kappa

    v_std_phi = np.vectorize(std_phi)
    if np.var(phis) > 0.5:
        phis += np.pi
        phis = v_std_phi(phis)
    # centralization
    phi_central = (pts * phis).sum() / pts.sum()       
    
    eta_shift, phi_shift = etas, v_std_phi(phis - phi_central)
    # flipping
    def quadrant_max(eta, phi, pt):
        if eta > 0. and phi > 0.:
            pt_quadrants[0] += pt
        elif eta > 0. and phi < 0.:
            pt_quadrants[1] += pt
        elif eta < 0. and phi < 0.:
            pt_quadrants[2] += pt
        elif eta < 0. and phi > 0.:
            pt_quadrants[3] += pt
            
    pt_quadrants = [0., 0., 0., 0.]
    np.vectorize(quadrant_max)(eta_shift, phi_shift, pts)

    eta_flip, phi_flip = 1., 1.
    if np.argmax(pt_quadrants) == 1:
        phi_flip = -1.
    elif np.argmax(pt_quadrants) == 2:
        phi_flip = -1.
        eta_flip = -1.
    elif np.argmax(pt_quadrants) == 3:
        eta_flip = -1.

    eta_news = eta_shift * eta_flip
    phi_news = phi_shift * phi_flip

    return pts, eta_news, phi_news, Q_kappas

def sample_selection(File, kappa, res, pbar, imageWriter):
    n_sample = 0
    data = []
    for event_id, event in tqdm(enumerate(File)):
        # PT in (350, 450) GeV & |eta| < 1 for 2 jets
        Jets = []
        for jet in event.Jet:
            if jet.PT < 350 or jet.PT > 450 or abs(jet.Eta) > 1:
                continue
            Jets.append(jet)

        if len(Jets) < 2:
            pbar.update(1)
            continue
        
        # Delta R of vector bosons decayed quarks < 0.6
        merging = False
        for particle in event.Particle:
            if particle.PID in [255, -255, 257, 256, -256]:
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
                if DeltaR(q1.Eta, q1.Phi, q2.Eta, q2.Phi) < 0.6 and DeltaR(q3.Eta, q3.Phi, q4.Eta, q4.Phi) < 0.6:
                    merging = True
                break

        if not merging:
            pbar.update(1)
            continue

        event_jet = []    
        # match jet and vector boson
        for jet in Jets:
            if DeltaR(V1.Eta, V1.Phi, jet.Eta, jet.Phi) < 0.1 or DeltaR(V2.Eta, V2.Phi, jet.Eta, jet.Phi) < 0.1:
                event_jet.append(jet)

        if len(event_jet) < 2:
            pbar.update(1)
            continue

        # Get two highest PT jets except vector boson jets as forward jets
        # The jets are ordered by PT in default
        for jet in event.Jet:
            if jet not in event_jet:
                event_jet.append(jet)
                if len(event_jet) == 4:
                    break
                    
        if len(event_jet) < 4:
            pbar.update(1)
            continue
            
        # PT > 30 GeV
        if event_jet[3].PT < 30:
            pbar.update(1)
            continue
            
        # |dEta| > 2
        if abs(event_jet[2].Eta - event_jet[3].Eta) < 2:
            pbar.update(1)
            continue
        
        # pre-process jet constituents
        constituents = [consti for jet in event_jet for consti in jet.Constituents if consti != 0]
        pt_news, eta_news, phi_news, Q_kappas = preprocess(constituents, kappa)

        plot_range = [[-3,3], [-3,3]]
        resolutions = [res, res]
        hpT, _, _ = np.histogram2d(eta_news, phi_news, range=plot_range, bins=resolutions, weights=pt_news)
        hQk, _, _ = np.histogram2d(eta_news, phi_news, range=plot_range, bins=resolutions, weights=Q_kappas)
        hpT = hpT.repeat(75/res, axis=0).repeat(75/res, axis=1)
        hQk = hQk.repeat(75/res, axis=0).repeat(75/res, axis=1)
        
        json_obj = {'event_type': '', 'pT': '', 'Qk': '', 'event_mass':'', 'event_Charge':''}

        if V1.PID == 24 and V2.PID == 24:
            json_obj['event_type'] = 'W+W+'
        elif V1.PID == -24 and V2.PID == -24:
            json_obj['event_type'] = 'W-W-'
        elif V1.PID == 23 and V2.PID == 23:
            json_obj['event_type'] = 'ZZ'
        elif (V1.PID == 24 and V2.PID == 23) or (V1.PID == 23 and V2.PID == 24):
            json_obj['event_type'] = 'W+Z'
        elif (V1.PID == -24 and V2.PID == 23) or (V1.PID == 23 and V2.PID == -24):
            json_obj['event_type'] = 'W-Z'
        elif (V1.PID == 24 and V2.PID == -24) or (V1.PID == -24 and V2.PID == 24):
            json_obj['event_type'] = 'W+W-'
        else:
            print('Something wrong')

        image = np.array([np.array(json_obj['event_type']), hpT, hQk], dtype=object)
        np.save(imageWriter, image)
        
        jet_mass = [jet.Mass for jet in event_jet]
        data.append([json_obj['event_type'], sum(jet_mass), sum(Q_kappas)])
        
        n_sample += 1           
        pbar.update(1)
        
    return n_sample, data
     
def main():
    # the input root files should be same type
    kappa = float(sys.argv[1])
    res = int(sys.argv[2])
    
    # read all root files. 
    chain = r.TChain('Delphes')
    for rootfile in sys.argv[3:]:
        chain.Add(rootfile)
        
    nevent = chain.GetEntries()
    
    output_dir = '/'
    for text in sys.argv[3].split('/'):
        match = re.match('VBF_H5(pp_ww|mm_ww|z_zz|p_wz|m_wz|z_ww)_jjjj', text)
        if match:
            input_file_name = match.group()
            break
        output_dir = os.path.join(output_dir, text)

    output_dir = os.path.join(output_dir, f'event_samples_kappa{kappa}-{nevent//1000}k-{res}x{res}-75x75')
    
    # create output directory
    if (not os.path.isdir(output_dir)):
        os.mkdir(output_dir, 0o777)
    
    image_name = os.path.join(output_dir, f'{input_file_name}.npy')
    count_name = os.path.join(output_dir, f'{input_file_name}.count')
    property_file = os.path.join(output_dir, f'{input_file_name}_properties.txt')
        
    with tqdm(total=chain.GetEntries()) as pbar, open(image_name, 'wb') as imagewriter:
        n_sample, data = sample_selection(chain, kappa, res, pbar, imagewriter)
        
    with open(count_name, 'w') as f:
        f.write(f'{n_sample}')
        
    # write event mass & Qk to _properties.txt
    data = np.array(data)
    df = pd.DataFrame([])
    df['event type'] = data[:,0]
    df['event mass'] = data[:,1].astype(float)
    df['event charge'] = data[:,2].astype(float)
    df.to_csv(property_file)
        
if __name__ == '__main__':
    main()