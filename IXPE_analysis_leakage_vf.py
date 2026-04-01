#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:52:41 2025

@author: asullivan
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.insert(2,'/Users/asullivan/Stanford/ICARUS_Xray/')
from Constants import Constants
import ixpeobssim.evt.kislat2015
from ixpeobssim.irf import load_irf_set, modf, arf
import numbers
from datetime import datetime
import glob as glob
from astropy.io import fits
from astropy.table import Table, vstack
from scipy import interpolate
from scipy.optimize import minimize
import os
import leakagelib_v9
import copy
import scipy.stats
from scipy.stats import norm
from  background import ixpe_background

const = Constants()
IXPE_ref_time_MJD_d = 57754
IXPE_ref_time_MJD_frac = 0.00080074074074


def Convert_MJD_IXT(time):
    t=time-(IXPE_ref_time_MJD_d+IXPE_ref_time_MJD_frac)
    return t*const.day



class photon_region(object):
    def __init__(self, time, X, Y, Q, U, PI, W_MOM, NN=False, W_NN=None, src=None, bg_prob=False, BG_PROB=None):

        if src is None:
            src=np.where(time>0)
        #print(len(time[src]))
        self.time = time[src]
        self.X = X[src]
        self.Y = Y[src]
        self.Q = Q[src]
        self.U = U[src]
        self.PI = PI[src]
        self.E = convert_PI_to_energy(self.PI)
        self.W_MOM = W_MOM[src]
        if NN:
            self.W_NN=W_NN[src]
        if bg_prob:
            self.BG_PROB=BG_PROB[src]
        self.phase = None
        #print('Minimum moment weight is {}'.format(min(self.W_MOM)))

class IXPE_fits(object):
    def __init__(self, filename, hkname, du_id, NN=False, plot_modf=False, save_modf_plot=False, bg_prob=False):
        self.filename = filename
        file = fits.open(filename)
        data=file[1].data
        self.NN=NN
        self.bg_prob=bg_prob
        self.set_irf = load_irf_set(du_id =du_id)
        self.hkname = hkname
        print(len(data['TIME']))
        if bg_prob:
            if self.NN:
    
                self.total_image = photon_region(data['TIME'], data['X'], data['Y'], data['Q'], data['U'], data['PI'], data['W_MOM'], NN=self.NN, W_NN=data['W_NN'], bg_prob=bg_prob, BG_PROB=data['BG_PROB'])
                self.Make_NN_modf(plot=plot_modf, save_plot= save_modf_plot)
            else:
                self.total_image = photon_region(data['TIME'], data['X'], data['Y'], data['Q'], data['U'], data['PI'], data['W_MOM'], bg_prob=bg_prob,  BG_PROB=data['BG_PROB'])
        else:
            if self.NN:
    
                self.total_image = photon_region(data['TIME'], data['X'], data['Y'], data['Q'], data['U'], data['PI'], data['W_MOM'], NN=self.NN, W_NN=data['W_NN'])
                self.Make_NN_modf(plot=plot_modf, save_plot= save_modf_plot)
            else:
                self.total_image = photon_region(data['TIME'], data['X'], data['Y'], data['Q'], data['U'], data['PI'], data['W_MOM'])

    def Make_NN_modf(self, Emin=2, Emax=8, nbins = 15, plot=True, save_plot=False, total_modf=True, use_premade=True):
        '''function which invents the NN modulation factor by assuming the modulation factor is the mean weight at a given energy'''
        if use_premade:
            self.Energy_modf, self.modf_NN=np.load('/Users/asullivan/Stanford/IXPE/nn_modulation_factor.npy')
            nbins_use=len(self.Energy_modf)
        else:
            nbins_use=nbins
            bin_use = np.linspace(Emin, Emax+(Emax-Emin)/(nbins-1), nbins +1)
            modf=np.empty(nbins)
            for i in range(0, nbins):
    
                mask = (self.total_image.E>=bin_use[i]) & ( self.total_image.E<bin_use[i + 1])
                
                modf[i]=np.mean(self.total_image.W_NN[mask])
    
            self.Energy_modf = bin_use
            self.modf_NN = modf
            if plot:
                fig, ax = plt.subplots(1, figsize=(8, 6))
                ax.plot(self.Energy_modf[:len(self.Energy_modf)-1], self.modf_NN, color='red', label='NN Modulation')
                ax.plot(self.Energy_modf[:len(self.Energy_modf)-1], self.set_irf.modf(self.Energy_modf[:len(self.Energy_modf)-1]), color ='blue', label ='Moments Modulation')
                ax.set_xlabel('Energy (keV)')
                ax.set_ylabel('Modulation factor $\mu$')
                ax.legend()
                if save_plot:
                    fig.savefig('NN_modf_plot.png', dpi=300)
                    fig.savefig('NN_modf_plot.pdf', dpi=300)
        self.modf_func = interpolate.interp1d(self.Energy_modf[:nbins_use], self.modf_NN)
        

    def NN_Modf(self, E):
        '''a function that computes the NN modulation factor for a given energy'''
        return self.modf_func(E)
        

    def compute_psf_width(self, x, y, r=50, numbins=100, savefig=False, savename='temp_psf', dpi=300):
        rad=np.sqrt((x-self.total_image.X)**2+(x-self.total_image.Y)**2)

        bin_use = np.linspace(0, r, numbins +1)
        cts=np.empty(numbins)
        for i in range(0, numbins):
            mask = (rad >= bin_use[i]) & (rad < bin_use[i + 1])
            cts[i]=len(rad[mask])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(bin_use[:numbins], cts/bin_use[1:]**2, color='b')
        ax.set_xlabel('r (pixels)')
        ax.set_ylabel('cts')
        ax.set_xlim((0, r))
        if savefig:
            fig.savefig(savename+'.pdf', dpi=dpi)
            fig.savefig(savename+'.png', dpi=dpi)
        
    def select_region(self, x, y, r, annulus=False, rin=0, mask=None):

        if mask is None:
            if annulus:
                reg=np.where(((self.total_image.X-x)**2+(self.total_image.Y-y)**2< r**2) & ((self.total_image.X-x)**2+(self.total_image.Y-y)**2> rin**2), True, False)
            else:
                reg=np.where((self.total_image.X-x)**2+(self.total_image.Y-y)**2< r**2, True, False)
            if self.NN:
                return photon_region(self.total_image.time, self.total_image.X, self.total_image.Y, self.total_image.Q, self.total_image.U, self.total_image.PI, self.total_image.W_MOM, NN=self.NN, W_NN=self.total_image.W_NN, src=reg), reg
            else:
                return photon_region(self.total_image.time, self.total_image.X, self.total_image.Y, self.total_image.Q, self.total_image.U, self.total_image.PI, self.total_image.W_MOM,  src=reg), reg
        else:
            if annulus:
                reg=np.where(((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2< r**2) & ((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2> rin**2), True, False)
            else:
                reg=np.where((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2< r**2, True, False)
            if self.NN:    
                return photon_region(self.total_image.time[mask], self.total_image.X[mask], self.total_image.Y[mask], self.total_image.Q[mask], self.total_image.U[mask], self.total_image.PI[mask], self.total_image.W_MOM[mask], NN=self.NN, W_NN=self.total_image.W_NN[mask], src=reg), reg
            else:
                return photon_region(self.total_image.time[mask], self.total_image.X[mask], self.total_image.Y[mask], self.total_image.Q[mask], self.total_image.U[mask], self.total_image.PI[mask], self.total_image.W_MOM[mask], src=reg), reg
    def src_region(self, x, y, r):
        self.src_r = r
        self.src_x = x
        self.src_y = y
        self.src_photons, self.src_reg = self.select_region(x, y, r)
    def bkg_region(self, x, y, r, annulus=False, rin=0):
        self.bkg_x = x
        self.bkg_y = y        
        self.bkg_r = r
        self.annulus=annulus
        if annulus:
            self.bkg_rin=rin
        else:
            self.bkg_rin=0
        self.bkg_photons, self.bkg_reg = self.select_region(x, y, r, annulus=annulus, rin=rin)  

    def make_LC(self, timebin, Emin, Emax, weights=1):
        '''weights can be one of 0 (meaning no weights), 1 (meaning moments), and 2 (meaning neural net weights)'''
        Echoose_src=np.where((self.src_photons.E>Emin) & ( self.src_photons.E<Emax))
        Echoose_bkg = np.where((self.bkg_photons.E>Emin) & ( self.bkg_photons.E<Emax))
        #print(Echoose_src)
        if weights>0:
            # calculate the effective areas and modulation factors for every photon
            src_energies=self.src_photons.E[Echoose_src]
            bkg_energies=self.bkg_photons.E[Echoose_bkg]

            src_aeffs = self.set_irf.aeff(src_energies)/self.set_irf.aeff(src_energies)
            bkg_aeffs = self.set_irf.aeff(bkg_energies)/self.set_irf.aeff(bkg_energies)
            if weights == 1:
                src_modfs = self.set_irf.modf(src_energies)
                bkg_modfs = self.set_irf.modf(bkg_energies)
            elif weights == 2:
                src_modfs = self.NN_Modf(np.array(src_energies))
                bkg_modfs = self.NN_Modf(np.array(bkg_energies))
            
        min_time=np.min(self.total_image.time )
        max_time=np.max(self.total_image.time)
        #print(max_time-min_time)%timebin
        numbins=int((max_time-min_time)/timebin+1)
        self.t_LC=np.linspace(min_time, min_time+numbins*timebin, numbins)
        self.flux_LC_src = np.empty(numbins)
        self.error_LC_src = np.empty(numbins)
        self.flux_LC_bkg = np.empty(numbins)
        self.error_LC_bkg = np.empty(numbins)
        self.timebin = timebin
        for tbin in range(0, numbins):
            times_E_src = self.src_photons.time[Echoose_src]
            times_bin_src=np.where((self.t_LC[tbin]<self.src_photons.time[Echoose_src])& (self.t_LC[tbin]+timebin>self.src_photons.time[Echoose_src]))
            #print(times_bin_src)
            times_bin_bkg=np.where((self.t_LC[tbin]<self.bkg_photons.time[Echoose_bkg]) & (self.t_LC[tbin]+timebin>self.bkg_photons.time[Echoose_bkg]))
            times_E_bkg = self.bkg_photons.time[Echoose_bkg]
            if weights==1:
                MOM_src = self.src_photons.W_MOM[Echoose_src]
                #mu_perbin[tbin] = np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]*src_modfs[phases_bin_src])/np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])
                Wsqr=np.sum((MOM_src[times_bin_src]/src_aeffs[times_bin_src])**2)
                #Wsqr_perbin[tbin]=Wsqr
                #print(np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                self.flux_LC_src[tbin] = np.sum(MOM_src[times_bin_src]/src_aeffs[times_bin_src])
                self.flux_LC_src[tbin] /=timebin
                self.error_LC_src[tbin] = np.sqrt(Wsqr)/timebin

                MOM_bkg = self.bkg_photons.W_MOM[Echoose_bkg]
                self.flux_LC_bkg[tbin] = np.sum(MOM_bkg[times_bin_bkg]/bkg_aeffs[times_bin_bkg])
                Wsqr_bkg=np.sum((MOM_bkg[times_bin_bkg]/bkg_aeffs[times_bin_bkg])**2)
                self.error_LC_bkg[tbin] = np.sqrt(Wsqr_bkg)/timebin
                self.flux_LC_bkg[tbin] /= timebin

            elif weights == 2:
                NN_src = self.src_photons.W_NN[Echoose_src]
                #mu_perbin[tbin] = np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]*src_modfs[phases_bin_src])/np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])
                Wsqr=np.sum((NN_src[times_bin_src]/src_aeffs[times_bin_src])**2)
                #Wsqr_perbin[tbin]=Wsqr
                #print(np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                self.flux_LC_src[tbin] = np.sum(NN_src[times_bin_src]/src_aeffs[times_bin_src])
                self.flux_LC_src[tbin] /=timebin
                self.error_LC_src[tbin] = np.sqrt(Wsqr)/timebin

                NN_bkg = self.bkg_photons.W_NN[Echoose_bkg]
                self.flux_LC_bkg[tbin] = np.sum(NN_bkg[times_bin_bkg]/bkg_aeffs[times_bin_bkg])
                Wsqr_bkg=np.sum((NN_bkg[times_bin_bkg]/bkg_aeffs[times_bin_bkg])**2)
                self.error_LC_bkg[tbin] = np.sqrt(Wsqr_bkg)/timebin
                self.flux_LC_bkg[tbin] /= timebin            
            else:    
                self.flux_LC_src[tbin] = len(times_E_src[times_bin_src])
                self.error_LC_src[tbin] = np.sqrt(self.flux_LC_src[tbin])/timebin
                self.flux_LC_src[tbin] /= timebin
            

                self.flux_LC_bkg[tbin] = len(times_E_bkg[times_bin_bkg])
                self.error_LC_bkg[tbin] = np.sqrt(self.flux_LC_bkg[tbin])/timebin
                self.flux_LC_bkg[tbin] /= timebin
            
            #self.flux_LC_src[tbin] = len(times_E_src[times_bin_src])
            #self.error_LC_src[tbin] = np.sqrt(self.flux_LC_src[tbin])/timebin
            #self.flux_LC_src[tbin] /= timebin
            
            #times_bin_bkg=np.where((self.t_LC[tbin]<self.bkg_photons.time[Echoose_bkg]) & (self.t_LC[tbin]+timebin>self.bkg_photons.time[Echoose_bkg]))
            #times_E_bkg = self.bkg_photons.time[Echoose_bkg]
            #self.flux_LC_bkg[tbin] = len(times_E_bkg[times_bin_bkg])
            #self.error_LC_bkg[tbin] = np.sqrt(self.flux_LC_bkg[tbin])/timebin
        
            #self.flux_LC_bkg[tbin] /= timebin
    
    def make_LC_v2(self, timebin, Emin, Emax, weights=1, exclude_r=None):
        '''weights can be one of 0 (meaning no weights), 1 (meaning moments), and 2 (meaning neural net weights)'''
        
        Echoose = np.where((self.total_image.E>Emin) & ( self.total_image.E<Emax), True, False)
        
        #Echoose_src=np.where((self.src_photons.E>Emin) & ( self.src_photons.E<Emax))
        #Echoose_bkg = np.where((self.bkg_photons.E>Emin) & ( self.bkg_photons.E<Emax))
        #print(Echoose_src)
        #if weights>0:
            # calculate the effective areas and modulation factors for every photon
        #    src_energies=self.src_photons.E[Echoose_src]
        #    bkg_energies=self.bkg_photons.E[Echoose_bkg]

        #    src_aeffs = self.set_irf.aeff(src_energies)/self.set_irf.aeff(src_energies)
        #   bkg_aeffs = self.set_irf.aeff(bkg_energies)/self.set_irf.aeff(bkg_energies)
        #    if weights == 1:
        #        src_modfs = self.set_irf.modf(src_energies)
        #        bkg_modfs = self.set_irf.modf(bkg_energies)
        #    elif weights == 2:
        #        src_modfs = self.NN_Modf(np.array(src_energies))
        #        bkg_modfs = self.NN_Modf(np.array(bkg_energies))
            
        min_time=np.min(self.total_image.time )
        max_time=np.max(self.total_image.time)
        
        
            
        
       
        
        if exclude_r is not None:
            print("excluding r<{}".format(exclude_r))
            excluded_r = np.where(((self.total_image.X-self.src_x)**2+ (self.total_image.Y-self.src_y)**2<exclude_r**2), False, True)
            print(self.total_image.X**2+ self.total_image.Y**2)
            joint_mask = np.where((excluded_r) & (Echoose))
        else:
            joint_mask=Echoose
            
        
        times_total=self.total_image.time[Echoose]
        times_use=self.total_image.time[joint_mask]
        
        print("len of times total {}".format(len(times_total)))
        print("len of times use {}".format(len(times_use)))
        
        #print(max_time-min_time)%timebin
        numbins=int((max_time-min_time)/timebin+1)
        self.t_LC=np.linspace(min_time, max_time+(max_time-min_time)/numbins+1, numbins+1)
        
        self.total_counts_LC = np.empty(numbins)
        
        self.time_index=np.empty(len(times_total))
        print("time index_length",format(len(self.time_index)))
        self.bin_list=np.array(range(0, numbins))
        for tbins in range(0, numbins):
            time_mask = np.where((self.t_LC[tbins]<=times_total) & (self.t_LC[tbins+1]>times_total), True, False)
            time_mask_cut = np.where((self.t_LC[tbins]<=times_use) & (self.t_LC[tbins+1]>times_use), True, False)
            self.total_counts_LC[tbins]=len(times_use[time_mask_cut])
            
            self.time_index[time_mask] = tbins
        
      
    
    def make_orbital_LC_leakege(self, nbins, Emin, Emax, Porb, TASC, weights=1, use_weights=True, bkg_sub=0):
        E_mask=np.where((self.total_image.E>Emin) & ( self.total_image.E<Emax), True, False)
        self.src_photons_Elim, self.src_reg_Elim = self.select_region(self.src_x, self.src_y, self.src_r, mask=E_mask)
        self.src_photons_Elim.phase = Convert_Phase_normalized(self.src_photons_Elim.time, TASC, Porb)
        
        binwidth=1./(nbins)
        
        self.phases_orb_LC=np.linspace(0.0, 1.0-binwidth, nbins)
        self.flux_orb_LC_src = np.empty(nbins)
        self.error_orb_LC_src = np.empty(nbins)
        self.flux_orb_LC_bkg = np.empty(nbins)
        self.error_orb_LC_bkg = np.empty(nbins)

        self.Q_LC_orb_src = np.empty(nbins)
        self.U_LC_orb_src = np.empty(nbins)
        self.Qerr_LC_orb_src = np.empty(nbins)
        self.Uerr_LC_orb_src = np.empty(nbins)
        self.phases_LC_tot = Convert_Phase_normalized(self.t_LC, TASC, Porb)
        N_perbin=np.empty(nbins)
        source = leakagelib_v9.source.Source.no_image(True)
        IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (self.filename, self.hkname), energy_cut=(Emin, Emax))
        IXPE_data.retain(self.src_reg_Elim)
        #IXPE_data.cut(E_mask)
        IXPE_data.explicit_center(self.src_x, self.src_y)
        self.exposure_time=np.empty(nbins)
        for tbin in range(0, nbins):
            IXPE_data_copy = copy.deepcopy(IXPE_data)
            time_mask = np.where((tbin*binwidth<self.src_photons_Elim.phase) & ((tbin+1)*binwidth>self.src_photons_Elim.phase), True, False)
            
            IXPE_data_copy.retain(time_mask)
            settings = leakagelib_v9.ps_fit.FitSettings([IXPE_data_copy])
            settings.add_point_source() # Point source component
            settings.add_background() # Background component
            #settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized
            settings.fix_flux("bkg", 1)
            settings.set_initial_flux("src", 0.02)
            settings.set_initial_qu("src", (0.05, 0))
            settings.apply_circular_roi(self.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
            fitter = leakagelib_v9.ps_fit.Fitter([IXPE_data_copy], settings)
            result = fitter.fit()
            phases_exposure=np.where((self.phases_orb_LC[tbin]<self.phases_LC_tot)& (self.phases_orb_LC[tbin]+binwidth>self.phases_LC_tot))
            
            exposure= np.sum(self.exposure_clean[phases_exposure])
            print(result)
            #print('exposure time: {}'.format(exposure ))
            self.exposure_time[tbin]=self.timebin*exposure 
            self.flux_orb_LC_src[tbin]=result.params[('f', 'src')]#/self.exposure_time[tbin]
            self.Q_LC_orb_src[tbin]=result.params[('q', 'src')]#/self.exposure_time[tbin]
            self.U_LC_orb_src[tbin]=result.params[('u', 'src')]#/self.exposure_time[tbin]
            self.error_orb_LC_src[tbin]=result.sigmas[('f', 'src')]#/self.exposure_time[tbin]
            self.Qerr_LC_orb_src[tbin]=result.sigmas[('q', 'src')]#/self.exposure_time[tbin]
            self.Uerr_LC_orb_src[tbin]=result.sigmas[('u', 'src')]#/self.exposure_time[tbin]
            
            self.flux_orb_LC_bkg[tbin]=1.#result.params[('f', 'bkg')]
            #self.Q_LC_orb_bkg[tbin]=0#result.params[('q', 'bkg')]
            #self.U_LC_orb_bkg[tbin]=0#esult.params[('u', 'bkg')]
            
            self.error_LC_bkg[tbin] = 0.
            #print(result)
            #print(result.params[('f', 'src')])
            #print(result[('q', 'src')])
            #print(result[('u', 'src')])
            
            N_perbin[tbin] = len(IXPE_data_copy.evt_xs)#len(self.src_photons_Elim.phase[time_mask])
        self.N_perbin=N_perbin
        self.phases_orb_LC+=binwidth/2.
 
    
    def fitter(self, params,TASC, Porb, Emin=2, Emax=8, weights=1):
        if weights == 1:

            
            modf_func= self.set_irf.modf
        elif weights ==2:

            modf_func= self.NN_Modf
        E_mask=np.where((self.total_image.E>Emin) & ( self.total_image.E<Emax), True, False)
        self.src_photons_Elim, self.src_reg_Elim = self.select_region(self.src_x, self.src_y, self.src_r, mask=E_mask)
        
        self.src_photons_Elim.phase = Convert_Phase_normalized(self.src_photons_Elim.time, TASC, Porb)
        likelihood_num=likelihood(stripedwind_model, self.src_photons_Elim, params, modf_func)
        return likelihood_num
        
        
    def make_orbital_LC(self, nbins, Emin, Emax, Porb, TASC, weights=1, use_weights=True, bkg_sub=0, params={'Q0':1.0, 'U0':1.0}):
        '''weights can be one of 0 (meaning no weights), 1 (meaning moments), and 2 (meaning neural net weights)'''
            
        Echoose_src=np.where((self.src_photons_clean.E>Emin) & ( self.src_photons_clean.E<Emax), True, False)
        Echoose_bkg = np.where((self.bkg_photons_clean.E>Emin) & ( self.bkg_photons_clean.E<Emax), True, False)
        E_mask=np.where((self.total_image.E>Emin) & ( self.total_image.E<Emax), True, False)
        self.src_photons_Elim, self.src_reg_Elim = self.select_region(self.src_x, self.src_y, self.src_r, mask=E_mask)
        
        self.src_photons_Elim.phase = Convert_Phase_normalized(self.src_photons_Elim.time, TASC, Porb)
        
        print('orbital LC len_leakage {}'.format(len(self.src_photons_Elim.E)))
        print('orbital LC len_leakage p3 {}'.format(len(self.src_photons_clean.E[Echoose_src])))
        if weights>0:
            # calculate the effective areas and modulation factors for every photon
            src_energies=self.src_photons_clean.E[Echoose_src]
            bkg_energies=self.bkg_photons_clean.E[Echoose_bkg]

            src_aeffs = self.set_irf.aeff(src_energies)/self.set_irf.aeff(src_energies)
            bkg_aeffs = self.set_irf.aeff(bkg_energies)/self.set_irf.aeff(bkg_energies)
            if weights == 1:
                src_modfs = self.set_irf.modf(src_energies)
                bkg_modfs = self.set_irf.modf(bkg_energies)
                
                modf_func= self.set_irf.modf
            elif weights ==2:
                src_modfs = self.NN_Modf(src_energies)
                bkg_modfs = self.NN_Modf(bkg_energies)
                modf_func= self.NN_Modf
                
        
        self.src_photons_clean.phase = Convert_Phase_normalized(self.src_photons_clean.time, TASC, Porb)
        self.bkg_photons_clean.phase = Convert_Phase_normalized(self.bkg_photons_clean.time, TASC, Porb)
        binwidth=1/nbins
        self.phases_LC_tot = Convert_Phase_normalized(self.t_LC, TASC, Porb)
        
        self.phases_orb_LC=np.linspace(0.0, 1.0-binwidth, nbins)
        self.flux_orb_LC_src = np.empty(nbins)
        self.error_orb_LC_src = np.empty(nbins)
        self.flux_orb_LC_bkg = np.empty(nbins)
        self.error_orb_LC_bkg = np.empty(nbins)

        self.Q_LC_orb_src = np.empty(nbins)
        self.U_LC_orb_src = np.empty(nbins)
        self.Qerr_LC_orb_src = np.empty(nbins)
        self.Uerr_LC_orb_src = np.empty(nbins)

        self.Q_LC_orb_bkg = np.empty(nbins)
        self.U_LC_orb_bkg = np.empty(nbins)
        self.Qerr_LC_orb_bkg = np.empty(nbins)
        self.Uerr_LC_orb_bkg = np.empty(nbins)
        self.exposure_time=np.empty(nbins)
        self.mu = 1 # modulation factor
        self.MDP =np.empty(nbins)

        min_time=np.min(self.total_image.time )
        max_time=np.max(self.total_image.time)
        N_perbin=np.empty(nbins)
        mu_perbin = np.empty(nbins)
        Wsqr_perbin = np.empty(nbins)
        #exPer
        for tbin in range(0, nbins):
            
            phases_E_src = self.src_photons_clean.phase[Echoose_src]
            Q_E_src = self.src_photons_clean.Q[Echoose_src]
            U_E_src = self.src_photons_clean.U[Echoose_src]
            phases_bin_src=np.where((self.phases_orb_LC[tbin]<self.src_photons_clean.phase[Echoose_src])& (self.phases_orb_LC[tbin]+binwidth>self.src_photons_clean.phase[Echoose_src]))
            phases_exposure=np.where((self.phases_orb_LC[tbin]<self.phases_LC_tot)& (self.phases_orb_LC[tbin]+binwidth>self.phases_LC_tot))
            
            exposure= np.sum(self.exposure_clean[phases_exposure])
            #print('exposure time: {}'.format(exposure ))
            self.exposure_time[tbin]=self.timebin*exposure 
            #self.exposure_time[tbin]=1
            exposure_time =1#This local variable is set to 1 because we include the exposure time in a later calculation
            #print('exposure is {}'.format(exposure))
            Q_E_bkg = self.bkg_photons_clean.Q[Echoose_bkg]
            U_E_bkg = self.bkg_photons_clean.U[Echoose_bkg]
            phases_bin_bkg=np.where((self.phases_orb_LC[tbin]<self.bkg_photons_clean.phase[Echoose_bkg]) & (self.phases_orb_LC[tbin]+binwidth>self.bkg_photons_clean.phase[Echoose_bkg]))
            phases_E_bkg = self.bkg_photons_clean.phase[Echoose_bkg]
            #print(exposure_time)
            #print(times_bin_src)
            #print(times_bin_src)
            
            if weights==1:
                if use_weights:
                    MOM_src = self.src_photons_clean.W_MOM[Echoose_src]
                    MOM_bkg = self.bkg_photons_clean.W_MOM[Echoose_bkg]
                else:
                    MOM_src = np.ones(len(self.src_photons_clean.W_MOM))
                    MOM_bkg = np.ones(len(self.bkg_photons_clean.W_MOM))
                #MOM_src = self.src_photons_clean.W_MOM[Echoose_src]
                mu_perbin[tbin] = np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]*src_modfs[phases_bin_src])/np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])
                Wsqr=np.sum((MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])**2)
                Wsqr_perbin[tbin]=Wsqr
                #print(np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                
                
                
                self.flux_orb_LC_bkg[tbin] = np.sum(MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])
                Wsqr_bkg=np.sum((MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2)
                self.error_orb_LC_bkg[tbin] = np.sqrt(Wsqr_bkg)/exposure_time
                self.flux_orb_LC_bkg[tbin] /= exposure_time
                mu_bkg = np.sum(MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]*bkg_modfs[phases_bin_bkg])/np.sum(MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])
                self.Q_LC_orb_bkg[tbin] = np.sum(Q_E_bkg[phases_bin_bkg]*MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]/bkg_modfs[phases_bin_bkg])/exposure_time
                self.U_LC_orb_bkg[tbin] = np.sum(U_E_bkg[phases_bin_bkg]*MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]/bkg_modfs[phases_bin_bkg])/exposure_time
                self.Qerr_LC_orb_bkg[tbin] = np.sqrt(np.sum((MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2/bkg_modfs[phases_bin_bkg]-self.Q_LC_orb_bkg[tbin]**2/self.flux_orb_LC_bkg[tbin]**2)))/exposure_time
                self.Uerr_LC_orb_bkg[tbin] = np.sqrt(np.sum((MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2/bkg_modfs[phases_bin_bkg]-self.U_LC_orb_bkg[tbin]**2/self.flux_orb_LC_bkg[tbin]**2)))/exposure_time
                if bkg_sub==2:
                    I_src = np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])
                    Q_src= np.sum(Q_E_src[phases_bin_src]*MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])
                    U_src=np.sum(U_E_src[phases_bin_src]*MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])

                    if self.annulus:
                        A_ratio = self.src_r**2/(self.bkg_r**2-self.bkg_rin**2)
                    else:
                        A_ratio= self.src_r**2/(self.bkg_r**2)
                    
                    I_f=I_src-self.flux_orb_LC_bkg[tbin]*exposure_time*A_ratio
                    Q_f=Q_src-self.Q_LC_orb_bkg[tbin]*exposure_time*A_ratio
                    U_f=U_src-self.U_LC_orb_bkg[tbin]*exposure_time*A_ratio

                    I_f_e=np.sqrt(Wsqr+A_ratio**2*Wsqr_bkg)
                    sig_Q_src_sqr=np.sum((MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2./src_modfs[phases_bin_src]**2- (Q_f/I_f)**2))
                    sig_U_src_sqr=np.sum((MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2./src_modfs[phases_bin_src]**2- (U_f/I_f)**2))

                    sig_Q_bkg_sqr = np.sum((MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2./bkg_modfs[phases_bin_bkg]**2- (Q_f/I_f)**2))
                    sig_U_bkg_sqr = np.sum((MOM_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2./bkg_modfs[phases_bin_bkg]**2- (U_f/I_f)**2))

                    Q_f_e=np.sqrt(sig_Q_src_sqr+A_ratio**2*sig_Q_bkg_sqr)
                    U_f_e=np.sqrt(sig_U_src_sqr+A_ratio**2*sig_U_bkg_sqr)

                    self.flux_orb_LC_src[tbin] =I_f/exposure_time
                    self.error_orb_LC_src[tbin]=I_f_e/exposure_time

                    self.Q_LC_orb_src[tbin]=Q_f/exposure_time
                    self.Qerr_LC_orb_src[tbin]=Q_f_e/exposure_time
                    

                    self.U_LC_orb_src[tbin]=U_f/exposure_time
                    self.Uerr_LC_orb_src[tbin]=U_f_e/exposure_time
                    
                    
                else:
                    self.flux_orb_LC_src[tbin] = np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])
                    self.flux_orb_LC_src[tbin] /=exposure_time
                    #print(np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                    self.error_orb_LC_src[tbin] = np.sqrt(Wsqr)/exposure_time
                    self.Q_LC_orb_src[tbin] = np.sum(Q_E_src[phases_bin_src]*MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])/exposure_time
                    self.U_LC_orb_src[tbin] = np.sum(U_E_src[phases_bin_src]*MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])/exposure_time
                #self.Qerr_LC_orb_src[tbin] = np.sqrt(Wsqr)*(2/mu_perbin[tbin]**2-self.Q_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)**0.5/exposure_time
               # self.Uerr_LC_orb_src[tbin] = np.sqrt(Wsqr)*(2/mu_perbin[tbin]**2-self.U_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)**0.5/exposure_time
                    self.Qerr_LC_orb_src[tbin] = np.sqrt(np.sum((MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2/src_modfs[phases_bin_src]**2-self.Q_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)))/exposure_time
                    self.Uerr_LC_orb_src[tbin] = np.sqrt(np.sum((MOM_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2/src_modfs[phases_bin_src]**2-self.U_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)))/exposure_time
            elif weights==2:
                if use_weights:
                    
                    NN_src = self.src_photons_clean.W_NN[Echoose_src]
                    NN_bkg = self.bkg_photons_clean.W_NN[Echoose_bkg]
                else:
                    NN_src = np.ones(len(self.src_photons_clean.W_NN))
                    NN_bkg = np.ones(len(self.bkg_photons_clean.W_NN))

                mu_perbin[tbin] = np.sum(NN_src[phases_bin_src]/src_aeffs[phases_bin_src]*src_modfs[phases_bin_src])/np.sum(NN_src[phases_bin_src]/src_aeffs[phases_bin_src])
                Wsqr=np.sum((NN_src[phases_bin_src]/src_aeffs[phases_bin_src])**2)
                Wsqr_perbin[tbin]=Wsqr
                #print(np.sum(MOM_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                
                self.flux_orb_LC_bkg[tbin] = np.sum(NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])
                Wsqr_bkg=np.sum((NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2)
                self.error_orb_LC_bkg[tbin] = np.sqrt(Wsqr_bkg)/exposure_time
                self.flux_orb_LC_bkg[tbin] /= exposure_time
                mu_bkg = np.sum(NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]*bkg_modfs[phases_bin_bkg])/np.sum(NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])
                self.Q_LC_orb_bkg[tbin] = np.sum(Q_E_bkg[phases_bin_bkg]*NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]/bkg_modfs[phases_bin_bkg])/exposure_time
                self.U_LC_orb_bkg[tbin] = np.sum(U_E_bkg[phases_bin_bkg]*NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg]/bkg_modfs[phases_bin_bkg])/exposure_time
                self.Qerr_LC_orb_bkg[tbin] = np.sqrt(np.sum((NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2/bkg_modfs[phases_bin_bkg]**2-self.Q_LC_orb_bkg[tbin]**2/self.flux_orb_LC_bkg[tbin]**2)))/exposure_time
                self.Uerr_LC_orb_bkg[tbin] = np.sqrt(np.sum((NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2/bkg_modfs[phases_bin_bkg]**2-self.U_LC_orb_bkg[tbin]**2/self.flux_orb_LC_bkg[tbin]**2)))/exposure_time

                if bkg_sub==2:
                    I_src = np.sum(NN_src[phases_bin_src]/src_aeffs[phases_bin_src])
                    Q_src= np.sum(Q_E_src[phases_bin_src]*NN_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])
                    U_src=np.sum(U_E_src[phases_bin_src]*NN_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])

                    if self.annulus:
                        A_ratio = self.src_r**2/(self.bkg_r**2-self.bkg_rin**2)
                    else:
                        A_ratio= self.src_r**2/(self.bkg_r**2)
                    #print('Intensity with NEW method {}'.format(I_src))
                    I_f=I_src-self.flux_orb_LC_bkg[tbin]*exposure_time*A_ratio
                    Q_f=Q_src-self.Q_LC_orb_bkg[tbin]*exposure_time*A_ratio
                    U_f=U_src-self.U_LC_orb_bkg[tbin]*exposure_time*A_ratio

                    I_f_e=np.sqrt(Wsqr+A_ratio**2*Wsqr_bkg)
                    sig_Q_src_sqr=np.sum((NN_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2./src_modfs[phases_bin_src]**2- (Q_f/I_f)**2))
                    sig_U_src_sqr=np.sum((NN_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2./src_modfs[phases_bin_src]**2- (U_f/I_f)**2))

                    sig_Q_bkg_sqr = np.sum((NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2./bkg_modfs[phases_bin_bkg]**2- (Q_f/I_f)**2))
                    sig_U_bkg_sqr = np.sum((NN_bkg[phases_bin_bkg]/bkg_aeffs[phases_bin_bkg])**2*(2./bkg_modfs[phases_bin_bkg]**2- (U_f/I_f)**2))
        
                    Q_f_e=np.sqrt(sig_Q_src_sqr+A_ratio**2*sig_Q_bkg_sqr)
                    U_f_e=np.sqrt(sig_U_src_sqr+A_ratio**2*sig_U_bkg_sqr)
                    #print("area ratio {}".format(A_ratio))
                    #print("Q error fraction {}".format(Q_f/I_f))
                    if  sig_Q_src_sqr < 0:
                        print('Warning: Q err will be nan in time bin {}'.format(tbin))
                        
                    if  sig_U_src_sqr < 0:
                        print('Warning: Q err will be nan in time bin {}'.format(tbin))
                    
                    self.flux_orb_LC_src[tbin] =I_f/exposure_time
                    self.error_orb_LC_src[tbin]=I_f_e/exposure_time

                    self.Q_LC_orb_src[tbin]=Q_f/exposure_time
                    self.Qerr_LC_orb_src[tbin]=Q_f_e/exposure_time

                    self.U_LC_orb_src[tbin]=U_f/exposure_time
                    self.Uerr_LC_orb_src[tbin]=U_f_e/exposure_time  
                else:
                    #print('Intensity with bkg method {}'.format(I_src))
                    self.flux_orb_LC_src[tbin] = np.sum(NN_src[phases_bin_src]/src_aeffs[phases_bin_src])
                    #print('Intensity with OLD method {}'.format(self.flux_orb_LC_src[tbin]))
                    self.flux_orb_LC_src[tbin] /=exposure_time
                    #print(np.sum(NN_src[phases_bin_src]/src_aeffs[phases_bin_src]))
                    self.error_orb_LC_src[tbin] = np.sqrt(Wsqr)/exposure_time
                    self.Q_LC_orb_src[tbin] = np.sum(Q_E_src[phases_bin_src]*NN_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])/exposure_time
                    self.U_LC_orb_src[tbin] = np.sum(U_E_src[phases_bin_src]*NN_src[phases_bin_src]/src_aeffs[phases_bin_src]/src_modfs[phases_bin_src])/exposure_time
                    #self.Qerr_LC_orb_src[tbin] = np.sqrt(Wsqr)*(2/mu_perbin[tbin]**2-self.Q_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)**0.5/exposure_time
                    #self.Uerr_LC_orb_src[tbin] = np.sqrt(Wsqr)*(2/mu_perbin[tbin]**2-self.U_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)**0.5/exposure_time
                    self.Qerr_LC_orb_src[tbin] = np.sqrt(np.sum((NN_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2/src_modfs[phases_bin_src]**2-self.Q_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)))/exposure_time
                    self.Uerr_LC_orb_src[tbin] = np.sqrt(np.sum((NN_src[phases_bin_src]/src_aeffs[phases_bin_src])**2*(2/src_modfs[phases_bin_src]**2-self.U_LC_orb_src[tbin]**2/self.flux_orb_LC_src[tbin]**2)))/exposure_time                    
                
            else:
                self.flux_orb_LC_src[tbin] = len(phases_E_src[phases_bin_src])
                self.Q_LC_orb_src[tbin] = np.sum(Q_E_src[phases_bin_src])/exposure_time
                self.U_LC_orb_src[tbin] = np.sum(U_E_src[phases_bin_src])/exposure_time

                self.flux_orb_LC_src[tbin] /= exposure_time
                self.Qerr_LC_orb_src[tbin] = np.sqrt(np.sum(Q_E_src[phases_bin_src]**2))/exposure_time#np.sqrt(1/len(phases_E_src[phases_bin_src])*(0.5*self.flux_orb_LC_src[tbin]**2-0.25*self.mu**2*self.Q_LC_orb_src[tbin]**2))
                self.Uerr_LC_orb_src[tbin] = np.sqrt(np.sum(U_E_src[phases_bin_src]**2))/exposure_time#np.sqrt(1/len(phases_E_src[phases_bin_src])*(0.5*self.flux_orb_LC_src[tbin]**2-0.25*self.mu**2*
                self.flux_orb_LC_bkg[tbin] = len(phases_E_bkg[phases_bin_bkg])
                self.error_orb_LC_bkg[tbin] = np.sqrt(self.flux_orb_LC_bkg[tbin])/exposure_time
                self.flux_orb_LC_bkg[tbin] /= exposure_time
                mu_perbin[tbin]=1
            N_perbin[tbin] = len(phases_E_src[phases_bin_src])
            
           
            
            

            

        self.phases_orb_LC+=binwidth/2.
        self.PD_src = Polarization_frac(self.Q_LC_orb_src, self.U_LC_orb_src, self.flux_orb_LC_src)
        self.EVPA_src = EVPA(self.Q_LC_orb_src, self.U_LC_orb_src)
        
        self.PD_src_error = PD_error(self.Q_LC_orb_src, self.U_LC_orb_src, self.flux_orb_LC_src, self.Qerr_LC_orb_src, self.Uerr_LC_orb_src, self.error_orb_LC_src)
        self.EVPA_src_error = EVPA_error(self.Q_LC_orb_src, self.U_LC_orb_src,  self.Qerr_LC_orb_src, self.Uerr_LC_orb_src)
        if weights>0:
            #print(self.flux_orb_LC_src[0], self.error_orb_LC_src[0])
            self.mu=mu_perbin
            self.MDP= MDP_mom(mu_perbin, self.flux_orb_LC_src, self.error_orb_LC_src**2)
        else:
            self.MDP= MDP_no_mom(self.mu, N_perbin)
        self.N_perbin=N_perbin
        
        #self.EVPA_error = np.sqrt()
        
        
    def excise_background_flares(self, threshold=5, bg_prob_thresh=0.5):
        reject = np.where(abs((self.flux_LC_bkg-np.mean(self.flux_LC_bkg)))/np.std(self.flux_LC_bkg)> threshold)
        self.flux_LC_src_clean=   self.flux_LC_src
        self.error_LC_src_clean=   self.error_LC_src
        self.flux_LC_src_clean[reject] = 0
        self.error_LC_src_clean[reject] = 0
        self.flux_LC_bkg[reject] = 0
        #print(self.flux_LC_src)
        self.exposure_clean = np.where(self.flux_LC_bkg>0, 1, 0)
        excluded_parts=find_zero_blocks(self.exposure_clean)
        time_excluded = self.t_LC[excluded_parts]
        #print(time_excluded)
        photon_times_keep, mask = remove_between_values(self.total_image.time, time_excluded)
        mask = [True]*len(self.total_image.time)
        if self.bg_prob:
            mask_src=np.where((self.total_image.BG_PROB<bg_prob_thresh) & (mask))
        else:
            mask_src=mask
        if self.bg_prob:
            mask_bkg=np.where((self.total_image.BG_PROB>bg_prob_thresh) & (mask))
        else:
            mask_bkg=mask
        self.bkg_photons_clean, self.bkg_reg_clean = self.select_region(self.bkg_x, self.bkg_y, self.bkg_r,annulus=self.annulus, rin=self.bkg_rin, mask=mask_bkg) 

        self.src_photons_clean, self.src_reg_clean = self.select_region(self.src_x, self.src_y, self.src_r, mask=mask_src) 
        #for excluded_part in excluded_parts:
        #    src_photons_clean = photon_region(self.total_image.time, self.total_image.X, self.total_image.Y, self.total_image.Q, self.total_image.U, self.total_image.PI, self.total_image.W_MOM, reg)
        self.flux_LC_bkg_clean=   self.flux_LC_bkg
        self.error_LC_bkg_clean=   self.error_LC_bkg
        self.flux_LC_bkg_clean[reject] = 0
        self.error_LC_bkg_clean[reject] = 0
        print("Percentage of exposure removed {}".format(len(self.flux_LC_src_clean[reject])/len(self.flux_LC_src_clean)))
    #def calculate_exposure(self):
        
    def compute_spectrum(self, nbins, Emin=2, Emax=8):
        bin_use = np.linspace(Emin, Emax, nbins +1)
        cts=np.empty(nbins)
        for i in range(0, nbins):

            mask = (self.src_photons.E>=bin_use[i]) & ( self.src_photons.E<bin_use[i + 1])
            cts[i]=len(self.src_photons.E[mask])

        self.spec_counts = cts
        self.spec_Energy = bin_use
        
        
        
        #print(reject)
    
 
model_icarus = np.loadtxt('/Users/asullivan/Stanford/J1723/IXPE/Modeling/J1723_ixpe_model.txt')

model_phase = model_icarus[:, 0]
model_pd = model_icarus[:, 2]
model_evpa = model_icarus[:, 3]
model_phase[0]=0.0
model_phase[len(model_phase)-1]=1.0
Q_model=model_pd*np.cos(2.*np.pi*model_evpa/180.)
U_model =model_pd*np.sin(2.*np.pi*model_evpa/180.)
Q_model_interp=interpolate.interp1d(model_phase, Q_model)
U_model_interp=interpolate.interp1d(model_phase, U_model)


def stripedwind_model(phase):
    Q=Q_model_interp(phase)
    U=U_model_interp(phase)
    
    return Q, U

def likelihood(model, photons, params, modf, NN=False, weights=False):
    phase_i=photons.phase
    E_i = photons.E
    Q_model_i, U_model_i = model(phase_i)
    Q_i=photons.Q
    U_i=photons.U
    mu_i = modf(E_i)
    Q0= params['Q0']
    U0=params['U0']
    if weights:
        if NN:
            W_i = photons.W_NN
        else:
            W_i = photons.W_MOM
    else:
        W_i=1.
    #print(Q_i)
    #print(U_i)
    print(np.sum(W_i*mu_i*(Q_model_i*Q_i*Q0+U0*U_model_i*U_i)))
    l_i= 1.+W_i*mu_i*(Q_model_i*Q_i*Q0+U0*U_model_i*U_i)
    return np.sum(np.log(l_i))
       
def subtract_bkg_middle(I_src, I_bkg, r_src, r_bkg, annulus=False, r_bkg_in=0):
    if annulus:
        I_sub=I_src-I_bkg*r_src**2/(r_bkg**2-r_bkg_in**2)
    else:
        I_sub=I_src-I_bkg*r_src**2/r_bkg**2
    return I_sub
    
def subtract_bkg(I_src, Ie_src, I_bkg,  Ie_bkg, r_src, r_bkg, annulus=False, r_bkg_in=0):
    if annulus:
        I_sub=I_src-I_bkg*r_src**2/(r_bkg**2-r_bkg_in**2)
        Ie_sub = np.sqrt(Ie_src**2+Ie_bkg**2*(r_src**2/(r_bkg**2-r_bkg_in**2))**2)

        #print("Percent error = {}".format(Ie_sub/I_sub))
    else:
        #print('background ratio is {}'.format(r_src**2/r_bkg**2))
        I_sub=I_src-I_bkg*r_src**2/r_bkg**2
        Ie_sub = np.sqrt(Ie_src**2+Ie_bkg**2*(r_src**2/r_bkg**2)**2)
    print('Ierror ratio: {}'.format(Ie_sub/Ie_src))
    return I_sub, Ie_sub 

def Polarization_frac(Q, U, I):
    return np.sqrt(Q**2+U**2)/I



def PD_error(Q, U, I, Qe, Ue, Ie):
    dPDdq=Q/(I*np.sqrt(Q**2+U**2))    
    dPDdu=U/(I*np.sqrt(Q**2+U**2))
    dPDdi=-Polarization_frac(Q, U, I)/I
    PD = Polarization_frac(Q, U, I)
    N = (Q**2. + U**2)**2
    PD_error=PD * np.sqrt(Q**2. / N*Qe**2 + 
            U**2. / N * Ue**2 + (Ie/ I)**2.)
    return PD_error#np.sqrt(dPDdq**2*Qe**2+dPDdu**2*Ue**2+dPDdi**2*Ie**2)

def EVPA(Q, U):
     return np.arctan2(U, Q)*180./np.pi/2.

def EVPA_error(Q, U, Qe, Ue):
    dEVPAdu = 1/(Q*(1+(U/Q)**2))
    dEVPAdq = -U/Q**2/(1+(U/Q)**2)
    return np.sqrt(dEVPAdu**2*Ue**2+dEVPAdq**2*Qe**2)*180./np.pi/2


def polarization_calc_v2(q, u, qe, ue, nsig=1., n_dist=1000000, nbins=1000, mid_evpa=True):
    
    q_dist = np.random.normal(q, qe, n_dist)
    u_dist= np.random.normal(u, ue, n_dist)

    pd_dist = Polarization_frac(q_dist, u_dist, 1.)    
    evpa_dist = EVPA(q_dist, u_dist)
    mid_pt=0.5
    CInsig_upper=mid_pt+nsig*0.3413
    CInsig_lower=mid_pt-nsig*0.3413
    
    sorted_pd=sorted(pd_dist)

    sorted_evpa=sorted(evpa_dist)
    
    #Mid_pd =np.min(sorted_pd[int(mid_pt*n_dist): ])
    #sig_upper_pd = np.min(sorted_pd[int(CInsig_upper*n_dist): ])-Mid_pd
    #sig_lower_pd = Mid_pd-np.min(sorted_pd[int(CInsig_lower*n_dist): ])
    
    #Mid_evpa =np.min(sorted_evpa[int(mid_pt*n_dist): ])
    #sig_evpa_upper = np.min(sorted_evpa[int(CInsig_upper*n_dist): ])-Mid_evpa
    #sig_evpa_lower = Mid_evpa-np.min(sorted_evpa[int(CInsig_lower*n_dist): ])
    
    hist_pd_x, hist_pd_y= histogram(pd_dist, nbins)
    
    
    #plt.plot(hist_pd_x, hist_pd_y)
    
    hist_pdmod_y=hist_pd_y/hist_pd_x
    
    
    hist_pdmod_y=hist_pdmod_y/np.sum(hist_pdmod_y)
    #plt.plot(hist_pd_x, hist_pdmod_y)
    
    Mid_pd, sig_upper_pd, sig_lower_pd=find_interval(hist_pd_x, hist_pdmod_y)
    
    
    
    if mid_evpa:
        Mid_evpa =  EVPA(q, u)
        evpa_dist = ( evpa_dist-Mid_evpa+90)%180-90
        evpa_dist_sorted = sorted(evpa_dist)
        sig_evpa_upper = np.min(evpa_dist_sorted[int(CInsig_upper*n_dist): ])
        sig_evpa_lower = -np.min(evpa_dist_sorted[int(CInsig_lower*n_dist): ])
    else:
        
        hist_evpa_x, hist_evpa_y= histogram(evpa_dist, nbins)
        plt.plot(hist_evpa_x, hist_evpa_y)
        Mid_evpa, sig_evpa_upper, sig_evpa_lower=find_interval(hist_evpa_x, hist_evpa_y/np.sum(hist_evpa_y))
    #print(Mid_pd/sig_lower_pd)
    
    return Mid_pd, sig_upper_pd, sig_lower_pd, Mid_evpa, sig_evpa_upper, sig_evpa_lower

def polarization_lc(q_arr, u_arr, qe_arr, ue_arr, nsig=1., n_dist=100000000):
    pd_arr=np.empty(len(q_arr))
    pd_upper_arr=np.empty(len(q_arr))
    pd_lower_arr=np.empty(len(q_arr))
    
    evpa_arr=np.empty(len(q_arr))
    evpa_upper_arr=np.empty(len(q_arr))
    evpa_lower_arr=np.empty(len(q_arr))
    for n, q, in enumerate(q_arr):
        polarization_val=polarization_calc_v2(q_arr[n], u_arr[n], qe_arr[n], ue_arr[n], nsig=nsig, n_dist=n_dist)
        
        pd_arr[n]= polarization_val[0]
        pd_upper_arr[n]=polarization_val[1]
        pd_lower_arr[n]=polarization_val[2]
        
        evpa_arr[n]=polarization_val[3]
        evpa_upper_arr[n]=polarization_val[4]
        evpa_lower_arr[n]=polarization_val[5]
        
    return pd_arr, pd_upper_arr, pd_lower_arr, evpa_arr, evpa_upper_arr, evpa_lower_arr

def histogram(x, nbins):
    
    bin_use = np.linspace(np.min(x), np.max(x)+(np.min(x)-np.max(x))/(nbins-1), nbins +1)
    x_return=0.5*(bin_use[:nbins]+bin_use[1:])
    y=np.empty(nbins)
    for i in range(0, nbins):

        mask = (x>=bin_use[i]) & ( x<bin_use[i + 1])

        y[i]=len(x[mask])
    
    
    return x_return, y

def find_interval(hist_x, hist_y, CI=0.68):
    line=np.max(hist_y)
    interval = 0.01*line
    
    width_CI=0
    while width_CI < 0.68:
        dist_needed=hist_y-line
        xs_needed=np.where(dist_needed>0)
        width_CI=np.sum(hist_y[xs_needed])
        line -=interval
        
    selected_x=hist_x[xs_needed]
    selected_y=dist_needed[xs_needed]
    
    
    
    
    selected_y/=np.sum(selected_y)
    
    select=0
    n=0
    while select<0.5:
        select+=selected_y[n]
        n+=1

        print("n reported: {}".format(n))
    
    mid_x=selected_x[n-1]
    max_x=np.max(selected_x)
    min_x=np.min(selected_x)
    
    return mid_x, max_x-mid_x, mid_x-min_x

def PD_calc(Q, U, I, Qe, Ue, Ie):
    return Polarization_frac(Q, U, I), PD_error(Q, U, I, Qe, Ue, Ie)
def EVPA_calc(Q, U, Qe, Ue):
    return EVPA(Q, U), EVPA_error(Q, U, Qe, Ue)

def MDP_mom(mu, I, Wsqr):
    #print(Wsqr)
    #print(I)
    return 4.29*np.sqrt(Wsqr)/(mu*I)
def MDP_no_mom(mu, N):
     return 4.29/(mu*np.sqrt(N))

def convert_PI_to_energy(PI):
    return 0.02+0.04*PI

def Convert_Phase_normalized(t, XMM_T, P0):
    '''Converts XMM time to a normalized orbit (i.e. one which goes from 0 to 1)'''
    shift=t-XMM_T
    phase=np.remainder(shift, P0)/P0
    #print(shift/P0-(shift/P0).astype(int))
    
    return phase

def bin_orbit(phase, rate, error, n, clean=False, weights=False):
    '''Generates bins for the orbits given rate as a function of phase and number of bins
    '''
    interval=1/n
    
    r=np.empty(n)
    e=np.empty(n)
    phn=phase[phase<1]
    
    ph=np.linspace(0, 1-interval, n)
    #print(ph)
    rman=rate[phase<1]
    eman=error[phase<1]
    eman=np.where(rman==0, 1/deltat, eman)
    for j in range(0, n):
        rman1=rman[phn<(j+1)*interval]
        phn1=phn[phn<(j+1)*interval]
        
        rman2=rman1[phn1>j*interval]
        
        
        
        eman1=eman[phn<(j+1)*interval]
        eman2=eman1[phn1>j*interval]
        if clean:
            if weights:
                d=np.where((rman2>2*np.std(rman2)+np.average(rman2, weights=1/eman2**2)) | (rman2<-2*np.std(rman2)+np.average(rman2, weights=1/eman2**2)))

            else:
                d=np.where((rman2>2*np.std(rman2)+np.average(rman2)) | (rman2<-2*np.std(rman2)+np.average(rman2)))
            rman2=np.delete(rman2, d)
            eman2=np.delete(eman2, d)
        r[j]=np.sum(rman2/(eman2)**2)/np.sum(1/(eman2)**2)
        #print(len(eman2))
        W=np.sum(1/eman2**2)
        
        if len(rman2)>1:
            Ste=1/np.sqrt(len(rman2)-1)*np.sqrt(np.sum((rman2-r[j])**2/eman2**2)/W)
        else:
            Ste = eman2[0]
        #print(W**(-1/2))
        #print(np.sum(1/(eman2**2*W)**2)**(1/2))
        e[j]=Ste#np.std(r[j])*(np.sum(1/(eman2**2*W)**2)**(1/2))
    ph=ph+interval/2
    return ph, r, e

def find_zero_blocks(arr):
    zero_blocks = []
    in_block = False
    start = -1

    for i in range(len(arr)):
        if arr[i] == 0 and not in_block:
            # Start of a block of zeros
            start = i
            in_block = True
        if arr[i] == 1 and in_block:
            # End of a block of zeros
            zero_blocks.append((start, i - 1))
            in_block = False

    # If the array ends with a block of zeros
    if in_block:
        zero_blocks.append((start, len(arr) - 1))

    return zero_blocks

def average_datasets(fluxes, errors):
    mean_flux = sum(fluxes)
    mean_flux/=len(fluxes)
    mean_error=0
    for error in errors:
        mean_error +=error**2
    mean_error = np.sqrt(mean_error)/len(errors)
    return mean_flux, mean_error

def add_datasets(fluxes, errors, weights=None):
    if weights is None:
        flux_sum = sum(fluxes)
        error_sum= np.zeros(len(fluxes[0]))
        for error in errors:
            error_sum += error**2
        error_sum=error_sum**(0.5)
    else:
        flux_sum= np.zeros(len(fluxes[0]))
        error_sum= np.zeros(len(fluxes[0]))
        weight_sum= sum(weights)
        for n, error in enumerate(errors):
           #print('fluxes {}'.format(fluxes[n]))
           flux_sum+=weights[n]*fluxes[n]
           error_sum += (weights[n]*error)**2
        error_sum=error_sum**(0.5)/weight_sum
        flux_sum/=weight_sum
    return flux_sum, error_sum

def remove_between_values(arr, value_pairs):
    # Create a boolean mask initialized to True (keeping all elements)
    mask = np.ones(len(arr), dtype=bool)
    
    # Loop through each pair and set the mask to False for elements within the range
    for low, high in value_pairs:
        mask[(arr > low) & (arr < high)] = False  # Exclude values between low and high (exclusive)
    
    # Return the filtered array where the mask is True
    return arr[mask], mask

class standard_pipeline(object):
    def __init__(self, filename, hkfilename, du_id, srcd1_x, srcd1_y, srcd1_r, bkgd1_x, bkgd1_y, bkgd1_r, Pb, Ephemeris_IXPE, do_lc=True, do_fit=False, nbins = 5, timebin=1000, Emin=2, Emax=8, weights=1, use_weights=True, excise_bkg_flares=False, Emin_flare=2, Emax_flare=8, backgroundthresh=5, annulus=False, rin=0, compute_spectrum=False, specbins=10, bkg_sub=0, bg_prob=False, bg_prob_thresh=0.75):
        if weights ==2:
            self.NN=True
        else:
            self.NN=False
        self.Pb = Pb
        self.Emin = Emin
        self.Emax = Emax
        self.Ephemeris_IXPE=Ephemeris_IXPE
        self.weights = weights
        self.detector =IXPE_fits(filename, hkfilename, du_id=du_id, NN=self.NN, bg_prob=bg_prob)
        self.detector.src_region(srcd1_x, srcd1_y, srcd1_r)
        self.detector.bkg_region(bkgd1_x, bkgd1_y, bkgd1_r, annulus=annulus, rin=rin)
        if do_lc:
            self.detector.make_LC(timebin, Emin_flare, Emax_flare, weights=weights)
            self.detector.excise_background_flares(threshold=backgroundthresh, bg_prob_thresh=bg_prob_thresh)
            self.detector.make_orbital_LC(nbins, Emin, Emax, Pb, Ephemeris_IXPE, weights=weights, use_weights=use_weights, bkg_sub=bkg_sub)
        #self.detector.make_orbital_LC_leakege(nbins, Emin, Emax, Pb, Ephemeris_IXPE, weights=weights, use_weights=use_weights, bkg_sub=bkg_sub)
        if compute_spectrum:
            self.detector.compute_spectrum(specbins, Emin=Emin, Emax=Emax)
            
    def fitter(self, params):
         likelihood_num= self.detector.fitter(params, self.Ephemeris_IXPE, self.Pb, Emin=self.Emin, Emax=self.Emax, weights=self.weights)
         return likelihood_num
            
class standard_pipeline_leakage(object):
    def __init__(self, filename, hkfilename,  du_id, srcd1_x, srcd1_y, srcd1_r, bkgd1_x, bkgd1_y, bkgd1_r, Pb, Ephemeris_IXPE, nbins = 5, timebin=1000, Emin=2, Emax=8, weights=1, use_weights=True, excise_bkg_flares=False, Emin_flare=2, Emax_flare=8, backgroundthresh=5, annulus=False, rin=0, compute_spectrum=False, specbins=10, bkg_sub=0, bg_prob=False, bg_prob_thresh=0.75):
        if weights ==2:
            NN=True
        else:
            NN=False
        self.detector =IXPE_fits(filename, hkfilename, du_id=du_id, NN=NN, bg_prob=bg_prob)
        self.detector.src_region(srcd1_x, srcd1_y, srcd1_r)
        self.detector.bkg_region(bkgd1_x, bkgd1_y, bkgd1_r, annulus=annulus, rin=rin)
        self.detector.make_LC(timebin, Emin_flare, Emax_flare, weights=weights)
        self.detector.excise_background_flares(threshold=backgroundthresh, bg_prob_thresh=bg_prob_thresh)
        #self.detector.make_orbital_LC(nbins, Emin, Emax, Pb, Ephemeris_IXPE, weights=weights, use_weights=use_weights, bkg_sub=bkg_sub)
        self.detector.make_orbital_LC_leakege(nbins, Emin, Emax, Pb, Ephemeris_IXPE, weights=weights, use_weights=use_weights, bkg_sub=bkg_sub)
        if compute_spectrum:
            self.detector.compute_spectrum(specbins, Emin=Emin, Emax=Emax)

def check_array(variable):
    if isinstance(variable, numbers.Number):
        return True
    elif isinstance(variable, (list, np.ndarray)):
        return False

class run_pipeline(object):
    def __init__(self, direct_list, hk_direct_list, srcd1_x_list, srcd1_y_list, srcd1_r_list, bkgd1_x_list, bkgd1_y_list, bkgd1_r_list, Pb_list, Ephemeris_IXPE_list, nbins = 5, timebin=1000, Emin=2, Emax=8, bkg_sub=0, det_use = [1, 2, 3], annulus=False, bkgd1_rin=0, weights=1, use_weights=True, compute_spectrum=False, specbins=10,bg_thresh =5,  bg_prob=False, bg_prob_thresh=0.5, combine_all=False):
        if check_array(nbins):
            nbins_list=np.full(len(direct_list), nbins)
        else:
            nbins_list=nbins
        if check_array(timebin):
            timebin_list=np.full(len(direct_list), timebin)
        else:
            timebin_list=timebin
        if check_array(Emin):
            Emin_list=np.full(len(direct_list), Emin)
        else:
            Emin_list=Emin
        if check_array(Emax):
            Emax_list=np.full(len(direct_list), Emax)
        else:
            Emax_list=Emax

        if check_array(bkg_sub):
            bkg_sub_list=np.full(len(direct_list), bkg_sub)
        else:
            bkg_sub_list=bkg_sub
        if isinstance(annulus, bool):
            annulus_list=[annulus]*len(direct_list)
        else:
            annulus_list=annulus
        if check_array(bkgd1_rin):
            bkgd1_rin_list=np.full(len(direct_list), bkgd1_rin)
        else:
            bkgd1_rin_list=bkgd1_rin
        if check_array(weights):
            weights_list=np.full(len(direct_list), weights)
        else:
            weights_list=weights
        if isinstance(compute_spectrum, bool):
            compute_spectrum_list=[compute_spectrum]*len(direct_list)
        else:
            compute_spectrum_list=compute_spectrum
        if check_array(specbins):
            specbins_list=np.full(len(direct_list), specbins)
        else:
            specbins_list=specbins

        if isinstance(use_weights, bool):
            use_weights_list=[use_weights]*len(direct_list)
        else:
            use_weights_list=use_weights
        if isinstance(bg_prob, bool):
            bg_prob_list=[bg_prob]*len(direct_list)
        else:
            bg_prob_list=bg_prob           
        if check_array(bg_prob_thresh):
            bg_prob_thresh_list=np.full(len(direct_list), bg_prob_thresh)
        else:
            bg_prob_thresh_list=bg_prob_thresh
        if check_array(bg_thresh):
            bg_thresh_list=np.full(len(direct_list), bg_thresh)
        else:
            bg_thresh_list=bg_thresh
        self.I_src = []
        self.I_bkg = [] 
        self.Ie_src = []
        self.Ie_bkg = []
        self.Q_src = []
        self.Q_bkg = []
        self.Qe_src = []
        self.Qe_bkg = []            
        self.U_src =[]
        self.U_bkg = []
        self.Ue_src = []
        self.Ue_bkg = []
        self.PD_src =[]
        self.PDerr_src =[]
        self.EVPA_src = []
        self.EVPAerr_src =[]
        self.phases=[]
        self.I=[]
        self.Ie=[]
        self.PD_bkg =[]
        self.PDerr_bkg =[]
        self.EVPA_bkg = []
        self.EVPAerr_bkg =[]
        self.MDP=[]
        self.MDP_bkg =[]
        self.ct_spec_src=[]
        self.E_spec_src=[]
        self.N_evt = []
        if 1 in det_use:
            keep_det1=1
            self.det1=[]
        else:
            keep_det1=0
            self.det1=None
        if 2 in det_use:
            keep_det2=1
            self.det2=[]
        else:
            keep_det2=0
            self.det2=None
        if 3 in det_use:
            keep_det3=1
            self.det3=[]
        else:
            keep_det3=0
            self.det3=None
        for k, direct in enumerate(direct_list):
            pattern =direct+'det1*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d1 = glob.glob(pattern)
            pattern =direct+'det2*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d2 = glob.glob(pattern)
            pattern =direct+'det3*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d3 = glob.glob(pattern)
            #print('detector 1 file ')
            #print(matching_files_d1)
            pattern = hk_direct_list[k]+'det1_att_v*.fits' 
            matching_hk_d1 = glob.glob(pattern)
            pattern = hk_direct_list[k]+'det2_att_v*.fits' 
            matching_hk_d2 = glob.glob(pattern)
            pattern = hk_direct_list[k]+'det3_att_v*.fits' 
            matching_hk_d3 = glob.glob(pattern)
            
            detector1=standard_pipeline(matching_files_d1[0], matching_hk_d1[0], 1, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], backgroundthresh=bg_thresh_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector2=standard_pipeline(matching_files_d2[0], matching_hk_d2[0], 2, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], backgroundthresh=bg_thresh_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector3=standard_pipeline(matching_files_d3[0], matching_hk_d3[0], 3, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], backgroundthresh=bg_thresh_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            
            if combine_all:
                if keep_det1:
                    self.det1.append(detector1)
                if keep_det2:
                    self.det2.append(detector2)
                if keep_det3:
                    self.det3.append(detector3)
            
            I_alldetectors=[keep_det1*detector1.detector.flux_orb_LC_src, keep_det2*detector2.detector.flux_orb_LC_src, keep_det3*detector3.detector.flux_orb_LC_src]
            Ie_alldetectors=[keep_det1*detector1.detector.error_orb_LC_src, keep_det2*detector2.detector.error_orb_LC_src, keep_det3*detector3.detector.error_orb_LC_src]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            I_combined, Ierr_combined = add_datasets(I_alldetectors, Ie_alldetectors)
            Q_combined, Qerr_combined = add_datasets(Q_alldetectors, Qe_alldetectors)#, weights=I_alldetectors)
            U_combined, Uerr_combined = add_datasets(U_alldetectors, Ue_alldetectors)#, weights=I_alldetectors)
            N_total = detector1.detector.N_perbin+detector2.detector.N_perbin+detector3.detector.N_perbin
            self.N_evt.append(N_total)
            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined, Ierr_combined**2)
            exposure_time=keep_det1*detector1.detector.exposure_time+keep_det2*detector2.detector.exposure_time+keep_det3*detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time #total exposure time in a bin
            #print('Total exposure time {}'.format(np.sum(exposure_time/3)))
            exposure_time=1.
            self.I_src.append(I_combined/exposure_time)
            self.Ie_src.append(Ierr_combined/exposure_time)
            self.Q_src.append(Q_combined/exposure_time)
            self.Qe_src.append(Qerr_combined/exposure_time)
            self.U_src.append(U_combined/exposure_time)
            self.Ue_src.append(Uerr_combined/exposure_time)

            
            I_bkg_alldetectors=[detector1.detector.flux_orb_LC_bkg, detector2.detector.flux_orb_LC_bkg, detector3.detector.flux_orb_LC_bkg]
            Ie_bkg_alldetectors=[detector1.detector.error_orb_LC_bkg, detector2.detector.error_orb_LC_bkg, detector3.detector.error_orb_LC_bkg]
            Q_bkg_alldetectors = [detector1.detector.Q_LC_orb_bkg, detector2.detector.Q_LC_orb_bkg, detector3.detector.Q_LC_orb_bkg]
            Qe_bkg_alldetectors = [detector1.detector.Qerr_LC_orb_bkg, detector2.detector.Qerr_LC_orb_bkg, detector3.detector.Qerr_LC_orb_bkg]
            U_bkg_alldetectors = [detector1.detector.U_LC_orb_bkg, detector2.detector.U_LC_orb_bkg, detector3.detector.U_LC_orb_bkg]
            Ue_bkg_alldetectors = [detector1.detector.Uerr_LC_orb_bkg, detector2.detector.Uerr_LC_orb_bkg, detector3.detector.Uerr_LC_orb_bkg]
                
            I_bkg_combined, Ierr_bkg_combined = add_datasets(I_bkg_alldetectors, Ie_bkg_alldetectors)
            Q_bkg_combined, Qerr_bkg_combined = add_datasets(Q_bkg_alldetectors, Qe_bkg_alldetectors)
            U_bkg_combined, Uerr_bkg_combined = add_datasets(U_bkg_alldetectors, Ue_bkg_alldetectors)

                

            PD_combined_bkg, PDerr_combined_bkg =PD_calc(Q_bkg_combined, U_bkg_combined, I_bkg_combined, Qerr_combined, Uerr_bkg_combined, Ierr_bkg_combined) 
            EVPA_combined_bkg, EVPAerr_combined_bkg =EVPA_calc(Q_bkg_combined, U_bkg_combined,  Qerr_bkg_combined, Uerr_bkg_combined)
            self.I_bkg.append(I_bkg_combined/exposure_time)
            self.Ie_bkg.append(Ierr_bkg_combined/exposure_time)
            self.PD_bkg.append(PD_combined_bkg)
            self.PDerr_bkg.append(PDerr_combined_bkg)
            self.EVPA_bkg.append(EVPA_combined_bkg)
            self.EVPAerr_bkg.append(EVPAerr_combined_bkg)
            if bkg_sub_list[k]==1:
                #print('Yes subtracted background the old way')
                I_combined_sub, Ierr_combined_sub  = subtract_bkg(I_combined, Ierr_combined, I_bkg_combined, Ierr_bkg_combined, srcd1_r_list[k],bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                Q_combined_sub,  Qerr_combined_sub = subtract_bkg(Q_combined, Qerr_combined, Q_bkg_combined, Qerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                U_combined_sub,  Uerr_combined_sub = subtract_bkg(U_combined, Uerr_combined, U_bkg_combined, Uerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k]) 
                #print('Printing Q background subtracted')
                #print(Q_combined_sub)
            else:
                #print('No background subtraction background the old way')
                I_combined_sub, Ierr_combined_sub= I_combined, Ierr_combined
                Q_combined_sub, Qerr_combined_sub= Q_combined, Qerr_combined
                U_combined_sub, Uerr_combined_sub= U_combined, Uerr_combined
                #print('Printing Q no background subtracted')
                #print(Q_combined_sub)


            MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined_sub, Ierr_combined_sub**2)
            MDP_bkg_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_bkg_combined, Ierr_bkg_combined**2)
            self.I.append(I_combined_sub/exposure_time)
            self.Ie.append(Ierr_combined_sub/exposure_time)
            self.MDP.append(MDP_combined)
            self.MDP_bkg.append(MDP_bkg_combined)
            if compute_spectrum_list[k]:
                self.ct_spec_src.append(detector1.detector.spec_counts+detector2.detector.spec_counts+detector3.detector.spec_counts)
                self.E_spec_src.append(detector1.detector.spec_Energy[:len(detector1.detector.spec_Energy)-1])
            #self.phases.append()

            PD_combined, PDerr_combined =PD_calc(Q_combined_sub, U_combined_sub, I_combined_sub, Qerr_combined_sub, Uerr_combined_sub, Ierr_combined_sub)
            EVPA_combined, EVPAerr_combined =EVPA_calc(Q_combined_sub, U_combined_sub,  Qerr_combined_sub, Uerr_combined_sub)
            self.phases.append(detector1.detector.phases_orb_LC)
            self.PD_src.append(PD_combined)
            self.PDerr_src.append(PDerr_combined)
            self.EVPA_src.append(EVPA_combined)
            self.EVPAerr_src.append(EVPAerr_combined)
            #print('number of photons in dataset {} is {}'.format(k, N_total))


        if combine_all:
            self.I_combined_total, self.Ierr_combined_total = add_datasets(self.I, self.Ie)
            self.Q_combined_total, self.Qerr_combined_total = add_datasets(self.Q_src, self.Qe_src)#, self.I)
            self.U_combined_total, self.Uerr_combined_total = add_datasets(self.U_src, self.Ue_src)#, self.I)
            self.PD_src_total, self.PDerr_src_total =PD_calc(self.Q_combined_total, self.U_combined_total, self.I_combined_total, self.Qerr_combined_total, self.Uerr_combined_total, self.Ierr_combined_total)
            self.EVPA_combined_total, self.EVPAerr_combined_total =EVPA_calc(self.Q_combined_total, self.U_combined_total,  self.Qerr_combined_total, self.Uerr_combined_total)


    def likelihood(self, theta):
        Q0=theta[0]
        U0=theta[1]
        params={'Q0':Q0, 'U0':U0}
        likelihood=0
        if self.det1 is not None:
            for src in self.det1:
                likelihood+=src.fitter(params)
        if self.det2 is not None:
            for src in self.det2:
                likelihood+=src.fitter(params)
        if self.det3 is not None:
            for src in self.det3:
                likelihood += src.fitter(params)
                
        return -likelihood
    
    def fitter(self, initial=[1.0, 1.0], mcmc=False):
        initial_use=np.array(initial)
        if not mcmc:
            fit = minimize(self.likelihood, initial_use)
        
        return fit.x
        
        
                
class run_pipeline_leakage(object):
    def __init__(self, direct_list, hk_direct_list, srcd1_x_list, srcd1_y_list, srcd1_r_list, bkgd1_x_list, bkgd1_y_list, bkgd1_r_list, Pb_list, Ephemeris_IXPE_list, nbins = 5, timebin=1000, Emin=2, Emax=8, bkg_sub=0, annulus=False, bkgd1_rin=0, weights=1, use_weights=True, compute_spectrum=False, specbins=10, bg_thresh =5, bg_prob=False, bg_prob_thresh=0.5, combine_all=True):
        if check_array(nbins):
            nbins_list=np.full(len(direct_list), nbins)
        else:
            nbins_list=nbins
        if check_array(timebin):
            timebin_list=np.full(len(direct_list), timebin)
        else:
            timebin_list=timebin
        if check_array(Emin):
            Emin_list=np.full(len(direct_list), Emin)
        else:
            Emin_list=Emin
        if check_array(Emax):
            Emax_list=np.full(len(direct_list), Emax)
        else:
            Emax_list=Emax

        if check_array(bkg_sub):
            bkg_sub_list=np.full(len(direct_list), bkg_sub)
        else:
            bkg_sub_list=bkg_sub
        if isinstance(annulus, bool):
            annulus_list=[annulus]*len(direct_list)
        else:
            annulus_list=annulus
        if check_array(bkgd1_rin):
            bkgd1_rin_list=np.full(len(direct_list), bkgd1_rin)
        else:
            bkgd1_rin_list=bkgd1_rin
        if check_array(weights):
            weights_list=np.full(len(direct_list), weights)
        else:
            weights_list=weights
        if isinstance(compute_spectrum, bool):
            compute_spectrum_list=[compute_spectrum]*len(direct_list)
        else:
            compute_spectrum_list=compute_spectrum
        if check_array(specbins):
            specbins_list=np.full(len(direct_list), specbins)
        else:
            specbins_list=specbins

        if isinstance(use_weights, bool):
            use_weights_list=[use_weights]*len(direct_list)
        else:
            use_weights_list=use_weights
        if isinstance(bg_prob, bool):
            bg_prob_list=[bg_prob]*len(direct_list)
        else:
            bg_prob_list=bg_prob           
        if check_array(bg_prob_thresh):
            bg_prob_thresh_list=np.full(len(direct_list), bg_prob_thresh)
        else:
            bg_prob_thresh_list=bg_prob_thresh
        if check_array(bg_thresh):
            bg_thresh_list=np.full(len(direct_list), bg_thresh)
        else:
            bg_thresh_list=bg_thresh
        self.I_src = []
        self.I_bkg = [] 
        self.Ie_src = []
        self.Ie_bkg = []
        self.Q_src = []
        self.Q_bkg = []
        self.Qe_src = []
        self.Qe_bkg = []            
        self.U_src =[]
        self.U_bkg = []
        self.Ue_src = []
        self.Ue_bkg = []
        self.PD_src =[]
        self.PDerr_src =[]
        self.EVPA_src = []
        self.EVPAerr_src =[]
        self.phases=[]
        self.I=[]
        self.Ie=[]
        self.PD_bkg =[]
        self.PDerr_bkg =[]
        self.EVPA_bkg = []
        self.EVPAerr_bkg =[]
        self.MDP=[]
        self.MDP_bkg =[]
        self.ct_spec_src=[]
        self.E_spec_src=[]
        self.N_evt = []
        keep_det1=1
        keep_det2=1
        keep_det3=1
        for k, direct in enumerate(direct_list):
            pattern =direct+'det1*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d1 = glob.glob(pattern)
            pattern =direct+'det2*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d2 = glob.glob(pattern)
            pattern =direct+'det3*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d3 = glob.glob(pattern)
            
            pattern = hk_direct_list[k]+'det1_att_v*.fits' 
            matching_hk_d1 = glob.glob(pattern)
            pattern = hk_direct_list[k]+'det2_att_v*.fits' 
            matching_hk_d2 = glob.glob(pattern)
            pattern = hk_direct_list[k]+'det3_att_v*.fits' 
            matching_hk_d3 = glob.glob(pattern)
            #print('detector 1 file ')
            #print(matching_files_d1)
            detector1=standard_pipeline_leakage(matching_files_d1[0], matching_hk_d1[0], 1, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector2=standard_pipeline_leakage(matching_files_d2[0], matching_hk_d2[0], 2, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector3=standard_pipeline_leakage(matching_files_d3[0], matching_hk_d3[0], 3, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])

            I_alldetectors=[keep_det1*detector1.detector.flux_orb_LC_src, keep_det2*detector2.detector.flux_orb_LC_src, keep_det3*detector3.detector.flux_orb_LC_src]
            Ie_alldetectors=[keep_det1*detector1.detector.error_orb_LC_src, keep_det2*detector2.detector.error_orb_LC_src, keep_det3*detector3.detector.error_orb_LC_src]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            
            I_alldetectors_flux=[keep_det1*detector1.detector.flux_orb_LC_src/(1.+detector1.detector.flux_orb_LC_src)*detector1.detector.N_perbin, keep_det2*detector2.detector.flux_orb_LC_src/(1.+detector2.detector.flux_orb_LC_src)*detector2.detector.N_perbin, keep_det3*detector3.detector.flux_orb_LC_src/(1.+detector3.detector.flux_orb_LC_src)*detector3.detector.N_perbin]
            Ie_alldetectors_flux=[keep_det1*detector1.detector.error_orb_LC_src/(1.+detector1.detector.flux_orb_LC_src)*detector1.detector.N_perbin, keep_det2*detector2.detector.error_orb_LC_src/(1.+detector2.detector.flux_orb_LC_src)*detector2.detector.N_perbin, keep_det3*detector3.detector.error_orb_LC_src/(1.+detector3.detector.flux_orb_LC_src)*detector3.detector.N_perbin]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            
            Q_alldetectors_flux = [x * y for x, y in zip( Q_alldetectors, I_alldetectors_flux)]
            Qe_alldetectors_flux = [x * y for x, y in zip( Qe_alldetectors, I_alldetectors_flux)]
            
            U_alldetectors_flux = [x * y for x, y in zip( U_alldetectors, I_alldetectors_flux)]
            Ue_alldetectors_flux = [x * y for x, y in zip( Ue_alldetectors, I_alldetectors_flux)]
            
            I_combined, Ierr_combined = add_datasets(I_alldetectors_flux, Ie_alldetectors_flux)
            Q_combined, Qerr_combined = add_datasets(Q_alldetectors_flux, Qe_alldetectors_flux)#,  weights=I_alldetectors)
            U_combined, Uerr_combined = add_datasets(U_alldetectors_flux, Ue_alldetectors_flux)#,  weights=I_alldetectors)
            
   
            N_total = detector1.detector.N_perbin+detector2.detector.N_perbin+detector3.detector.N_perbin
            self.N_evt.append(N_total)
            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined, Ierr_combined**2)
            exposure_time=keep_det1*detector1.detector.exposure_time+keep_det2*detector2.detector.exposure_time+keep_det3*detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time #total exposure time in a bin
            #print('Total exposure time {}'.format(np.sum(exposure_time/3)))
            exposure_time=1.
            
            self.I_src.append(I_combined/exposure_time)
            self.Ie_src.append(Ierr_combined/exposure_time)
            self.Q_src.append(Q_combined/exposure_time)
            self.Qe_src.append(Qerr_combined/exposure_time)
            self.U_src.append(U_combined/exposure_time)
            self.Ue_src.append(Uerr_combined/exposure_time)

            
            I_bkg_alldetectors=[detector1.detector.flux_orb_LC_bkg, detector2.detector.flux_orb_LC_bkg, detector3.detector.flux_orb_LC_bkg]
            Ie_bkg_alldetectors=[detector1.detector.error_orb_LC_bkg, detector2.detector.error_orb_LC_bkg, detector3.detector.error_orb_LC_bkg]
            #Q_bkg_alldetectors = [detector1.detector.Q_LC_orb_bkg, detector2.detector.Q_LC_orb_bkg, detector3.detector.Q_LC_orb_bkg]
            #Qe_bkg_alldetectors = [detector1.detector.Qerr_LC_orb_bkg, detector2.detector.Qerr_LC_orb_bkg, detector3.detector.Qerr_LC_orb_bkg]
            #U_bkg_alldetectors = [detector1.detector.U_LC_orb_bkg, detector2.detector.U_LC_orb_bkg, detector3.detector.U_LC_orb_bkg]
            #Ue_bkg_alldetectors = [detector1.detector.Uerr_LC_orb_bkg, detector2.detector.Uerr_LC_orb_bkg, detector3.detector.Uerr_LC_orb_bkg]
                
            I_bkg_combined, Ierr_bkg_combined = add_datasets(I_bkg_alldetectors, Ie_bkg_alldetectors)
            #Q_bkg_combined, Qerr_bkg_combined = add_datasets(Q_bkg_alldetectors, Qe_bkg_alldetectors)
            #U_bkg_combined, Uerr_bkg_combined = add_datasets(U_bkg_alldetectors, Ue_bkg_alldetectors)

                

            #PD_combined_bkg, PDerr_combined_bkg =PD_calc(Q_bkg_combined, U_bkg_combined, I_bkg_combined, Qerr_combined, Uerr_bkg_combined, Ierr_bkg_combined) 
            #EVPA_combined_bkg, EVPAerr_combined_bkg =EVPA_calc(Q_bkg_combined, U_bkg_combined,  Qerr_bkg_combined, Uerr_bkg_combined)
            self.I_bkg.append(I_bkg_combined/exposure_time)
            self.Ie_bkg.append(Ierr_bkg_combined/exposure_time)
            #self.PD_bkg.append(PD_combined_bkg)
            #self.PDerr_bkg.append(PDerr_combined_bkg)
            #self.EVPA_bkg.append(EVPA_combined_bkg)
            #self.EVPAerr_bkg.append(EVPAerr_combined_bkg)
            if bkg_sub_list[k]==1:
                #print('Yes subtracted background the old way')
                I_combined_sub, Ierr_combined_sub  = subtract_bkg(I_combined, Ierr_combined, I_bkg_combined, Ierr_bkg_combined, srcd1_r_list[k],bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                #Q_combined_sub,  Qerr_combined_sub = subtract_bkg(Q_combined, Qerr_combined, Q_bkg_combined, Qerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                #U_combined_sub,  Uerr_combined_sub = subtract_bkg(U_combined, Uerr_combined, U_bkg_combined, Uerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k]) 
                #print('Printing Q background subtracted')
                #print(Q_combined_sub)
            else:
                #print('No background subtraction background the old way')
                I_combined_sub, Ierr_combined_sub= I_combined, Ierr_combined
                Q_combined_sub, Qerr_combined_sub= Q_combined, Qerr_combined
                U_combined_sub, Uerr_combined_sub= U_combined, Uerr_combined
                #print('Printing Q no background subtracted')
                #print(Q_combined_sub)


            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined_sub, Ierr_combined_sub**2)
            #MDP_bkg_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_bkg_combined, Ierr_bkg_combined**2)
            
            self.I.append(I_combined_sub/exposure_time)
            self.Ie.append(Ierr_combined_sub/exposure_time)
            #self.MDP.append(MDP_combined)
            #self.MDP_bkg.append(MDP_bkg_combined)
            if compute_spectrum_list[k]:
                self.ct_spec_src.append(detector1.detector.spec_counts+detector2.detector.spec_counts+detector3.detector.spec_counts)
                self.E_spec_src.append(detector1.detector.spec_Energy[:len(detector1.detector.spec_Energy)-1])
            #self.phases.append()

            PD_combined, PDerr_combined =PD_calc(Q_combined_sub, U_combined_sub, I_combined_sub, Qerr_combined_sub, Uerr_combined_sub, Ierr_combined_sub)
            EVPA_combined, EVPAerr_combined =EVPA_calc(Q_combined_sub, U_combined_sub,  Qerr_combined_sub, Uerr_combined_sub)
            self.phases.append(detector1.detector.phases_orb_LC)
            self.PD_src.append(PD_combined)
            self.PDerr_src.append(PDerr_combined)
            self.EVPA_src.append(EVPA_combined)
            self.EVPAerr_src.append(EVPAerr_combined)
            
        if combine_all:
            self.I_combined_total, self.Ierr_combined_total = add_datasets(self.I, self.Ie)
            self.Q_combined_total, self.Qerr_combined_total = add_datasets(self.Q_src, self.Qe_src)#, self.I)
            self.U_combined_total, self.Uerr_combined_total = add_datasets(self.U_src, self.Ue_src)#, self.I)
            self.PD_src_total, self.PDerr_src_total =PD_calc(self.Q_combined_total, self.U_combined_total, self.I_combined_total, self.Qerr_combined_total, self.Uerr_combined_total, self.Ierr_combined_total)
            self.EVPA_combined_total, self.EVPAerr_combined_total =EVPA_calc(self.Q_combined_total, self.U_combined_total,  self.Qerr_combined_total, self.Uerr_combined_total)
            





class run_pipeline_leakage_v2(object):
    def __init__(self, direct_list, hk_direct_list, srcd1_x_list, srcd1_y_list, srcd1_r_list, bkgd1_x_list, bkgd1_y_list, bkgd1_r_list, Pb_list, Ephemeris_IXPE_list, det_use = [1, 2, 3], nbins = 5, timebin=1000, Emin=2, Emax=8, bkg_sub=0, annulus=False, bkgd1_rin=0, weights=1, use_weights=True,  compute_spectrum=False, specbins=10, bg_thresh =5, bg_prob=False, bg_prob_thresh=0.5, combine_all=True, combine_detectors_leakage=True, model_fit = False, model_i=None, model_q=None, model_u=None, model_prior=[0.6, 0.03], model_unpolarized=False, fit_const=False, fit_const_qu=False, q_model_const=None, u_model_const=None,const_qu_prior=[0.0, 0.0], bg_weights=True, bg_cut=False, fit_bg_qu=False, fit_bg_det=False, fit_bg_det_sep=True, save_bkg=True, bkg_name='main', bkg_dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/', src_remove=20, make_lc=True, flare_cut=True, show_hist=False, cut_amount=35, fit_prtl_bg=True, fit_prtl_qu=True, do_fit=True, hist_xlim=[0, 100], hist_ylim=[0, 100], hist_bins=100, rotation_dir_model=1, rotation_dir_const=1, re_fit=False, load_bkg=True, load_bkg_name="main", load_bkg_dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/', fix_det23=True, exclude_r_input=False, do_pcube=True, bkgd1_outerradius=100):
        '''combine detectors leakage determines whether to combine the detectors when doing the leakage fitting or separately by summing in quadrature after doing individual detector fits'''
        if check_array(nbins):
            nbins_list=np.full(len(direct_list), nbins)
        else:
            nbins_list=nbins
        if check_array(timebin):
            timebin_list=np.full(len(direct_list), timebin)
        else:
            timebin_list=timebin
        if check_array(Emin):
            Emin_list=np.full(len(direct_list), Emin)
        else:
            Emin_list=Emin
        if check_array(Emax):
            Emax_list=np.full(len(direct_list), Emax)
        else:
            Emax_list=Emax

        if check_array(bkg_sub):
            bkg_sub_list=np.full(len(direct_list), bkg_sub)
        else:
            bkg_sub_list=bkg_sub
        if isinstance(annulus, bool):
            annulus_list=[annulus]*len(direct_list)
        else:
            annulus_list=annulus
        if check_array(bkgd1_rin):
            bkgd1_rin_list=np.full(len(direct_list), bkgd1_rin)
        else:
            bkgd1_rin_list=bkgd1_rin
        if check_array(weights):
            weights_list=np.full(len(direct_list), weights)
        else:
            weights_list=weights
        if isinstance(compute_spectrum, bool):
            compute_spectrum_list=[compute_spectrum]*len(direct_list)
        else:
            compute_spectrum_list=compute_spectrum
        if check_array(specbins):
            specbins_list=np.full(len(direct_list), specbins)
        else:
            specbins_list=specbins

        if isinstance(use_weights, bool):
            use_weights_list=[use_weights]*len(direct_list)
        else:
            use_weights_list=use_weights
        if isinstance(bg_prob, bool):
            bg_prob_list=[bg_prob]*len(direct_list)
        else:
            bg_prob_list=bg_prob           
        if check_array(bg_prob_thresh):
            bg_prob_thresh_list=np.full(len(direct_list), bg_prob_thresh)
        else:
            bg_prob_thresh_list=bg_prob_thresh
        if check_array(bg_thresh):
            bg_thresh_list=np.full(len(direct_list), bg_thresh)
        else:
            bg_thresh_list=bg_thresh
        self.I_src = []
        self.I_bkg = [] 
        self.Ie_src = []
        self.Ie_bkg = []
        self.Q_src = []
        self.Q_bkg = []
        self.Qe_src = []
        self.Qe_bkg = []            
        self.U_src =[]
        self.U_bkg = []
        self.Ue_src = []
        self.Ue_bkg = []
        self.PD_src =[]
        self.PDerr_src =[]
        self.EVPA_src = []
        self.EVPAerr_src =[]
        self.phases=[]
        self.I=[]
        self.Ie=[]
        self.PD_bkg =[]
        self.PDerr_bkg =[]
        self.EVPA_bkg = []
        self.EVPAerr_bkg =[]
        self.MDP=[]
        self.MDP_bkg =[]
        self.ct_spec_src=[]
        self.E_spec_src=[]
        self.N_evt = []
        keep_det1=1
        keep_det2=1
        keep_det3=1
        
        list_dataset_unpacked=[]
        
        list_datasets_unpacked_du1=[]
        list_datasets_unpacked_du2=[]
        list_datasets_unpacked_du3=[]
        
        self.combine_detectors_leakage=combine_detectors_leakage
        self.direct_list=direct_list
        self.du_id_list=[]
        
        for k, direct in enumerate(direct_list):
            #pattern =direct+'det1*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            #matching_files_d1 = glob.glob(pattern)
            #pattern =direct+'det2*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            #matching_files_d2 = glob.glob(pattern)
            #pattern =direct+'det3*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            #matching_files_d3 = glob.glob(pattern)
            
            #pattern = hk_direct_list[k]+'det1_att_v*.fits' 
            #matching_hk_d1 = glob.glob(pattern)
            #pattern = hk_direct_list[k]+'det2_att_v*.fits' 
            #matching_hk_d2 = glob.glob(pattern)
            #pattern = hk_direct_list[k]+'det3_att_v*.fits' 
            #matching_hk_d3 = glob.glob(pattern)
            #print('detector 1 file ')
            #print(matching_files_d1)
            for du_id in det_use:
                pattern =direct+'det'+str(du_id)+'*.fits'
                matching_files_d1 = glob.glob(pattern)
                pattern = hk_direct_list[k]+'det'+str(du_id)+'_att_v*.fits' 
                matching_hk_d1 = glob.glob(pattern)
                if weights_list[k] ==2:
                    NN=True
                else:
                    NN=False
                print(matching_files_d1)
                du_use = IXPE_fits(matching_files_d1[0], matching_hk_d1[0] , du_id, NN=NN, plot_modf=False, save_modf_plot=False, bg_prob=bg_prob_list[k])
                du_use.src_region(srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k])
                if  combine_detectors_leakage:
                    
                    
                    list_dataset_unpacked.append(du_use)
                    self.du_id_list.append(du_id)
                else:
                    if du_id == 1:
                        list_datasets_unpacked_du1.append(du_use)
                    if du_id == 2:
                        list_datasets_unpacked_du2.append(du_use)
                    if du_id == 3:
                        list_datasets_unpacked_du3.append(du_use)
        
        
        if combine_detectors_leakage:        
            if flare_cut:
                if exclude_r_input:
                    exclude_r=None
                else:
                    exclude_r=src_remove
                self.flare_cut(list_dataset_unpacked, timebin, Pb_list[0], Ephemeris_IXPE_list[0], Emin=Emin_list[0], Emax =Emax_list[0],  weights=weights_list[0], use_weights=True, bkg_sub=use_weights_list[0], show_hist=show_hist, cut_amount=cut_amount, xlim=hist_xlim, ylim=hist_ylim, hist_bins=hist_bins, exclude_r=src_remove)
            
            if fit_bg_det and fit_bg_det_sep and not load_bkg:
                self.IXPE_bg_fit(list_dataset_unpacked, src_remove, nbins, Pb_list[0], Ephemeris_IXPE_list[0], Emin=Emin_list[0], Emax=Emax_list[0], weights=weights_list[0],  bg_weights=bg_weights, bg_cut=bg_cut, save_bkg=save_bkg, background_name=bkg_name, background_dir_name=bkg_dir_name,flare_cut=flare_cut, fit_prtl_bg=fit_prtl_bg, fit_prtl_qu=fit_prtl_qu, fix_det23=fix_det23, do_pcube=do_pcube)
            
            if load_bkg:
                self.background_load = ixpe_background()
                self.background_load.load_bkg(load_bkg_name, load_bkg_dir_name)
                self.IXPE_pcube_bg(list_dataset_unpacked, src_remove, bkgd1_outerradius, nbins, Pb_list[0], Ephemeris_IXPE_list[0], Emin=Emin_list[0], Emax=Emax_list[0], weights=weights_list[0],  bg_weights=bg_weights, bg_cut=bg_cut, save_bkg=save_bkg, background_name=bkg_name, background_dir_name=bkg_dir_name,flare_cut=flare_cut, fit_prtl_bg=fit_prtl_bg, fit_prtl_qu=fit_prtl_qu, fix_det23=fix_det23, do_pcube=do_pcube)
            else:
                self.background_load = None
            
            if model_fit:
                
                self.IXPE_lc_fit(list_dataset_unpacked, model_i, model_q, model_u, nbins, Pb_list[0], Ephemeris_IXPE_list[0], Emin=Emin_list[0], Emax=Emax_list[0], weights=weights_list[0], use_weights=use_weights_list[0], model_prior=model_prior, model_unpolarized=model_unpolarized, fit_const=fit_const, fit_const_qu=fit_const_qu, const_qu_prior=const_qu_prior, q_model_const=q_model_const, u_model_const=u_model_const, bg_weights=bg_weights, bg_cut=bg_cut,  fit_bg_qu=fit_bg_qu, fit_bg_det=fit_bg_det, fit_bg_det_sep=fit_bg_det_sep, flare_cut=flare_cut, fit_prtl_bg=fit_prtl_bg, fit_prtl_qu=fit_prtl_qu, do_fit=do_fit, rotation_dir_model=rotation_dir_model, rotation_dir_const=rotation_dir_const, re_fit=re_fit, provide_background=self.background_load, do_pcube=do_pcube)
            self.list_dataset_unpacked = list_dataset_unpacked
            #self.nbins = nbins
            self.Pb_use=  Pb_list[0]
            self.Ephemeris = Ephemeris_IXPE_list[0]
            self.Emin = Emin_list[0]
            self.Emax = Emax_list[0]
            self.weights = weights_list[0]
            self.use_weights = use_weights_list[0]
            self.fit_bg_qu =  fit_bg_qu
            self.fit_bg_det=fit_bg_det
            self.fit_bg_det_sep=fit_bg_det_sep
            self.model_fit=model_fit
            self.flare_cut=flare_cut
            self.fit_prtl_bg=fit_prtl_bg
            self.fit_prtl_qu=fit_prtl_qu
            
            if make_lc:
                self.make_lc_func(nbins)
        else:
            self.I_src =[]
            self.Ie_src =[]
            self.Q_src =[]
            self.Qe_src =[]
            self.U_src =[]
            self.Ue_src =[]
            for du_id in det_use:
                if du_id == 1:
                    list_dataset_unpacked_use=list_datasets_unpacked_du1
                if du_id == 2:
                    list_dataset_unpacked_use=list_datasets_unpacked_du2
                if du_id == 3:
                    list_dataset_unpacked_use=list_datasets_unpacked_du3
                    
                
                
                self.make_orb_LC(list_dataset_unpacked_use, nbins, Pb_list[0], Ephemeris_IXPE_list[0], Emin=Emin_list[0], Emax=Emax_list[0], weights=weights_list[0], use_weights=use_weights_list[0])
                
                I_use= self.flux_orb_LC_src*self.N_perbin/(1+self.flux_orb_LC_src)
                I_err_use = self.error_orb_LC_src*self.N_perbin/(1+self.flux_orb_LC_src)
                
                self.I_src.append(I_use)
                self.Ie_src.append(I_err_use)
                
                self.Q_src.append(self.Q_LC_orb_src*I_use)
                self.Qe_src.append(self.Qerr_LC_orb_src*I_use)
                
                self.U_src.append(self.U_LC_orb_src*I_use)
                self.Ue_src.append(self.Uerr_LC_orb_src*I_use)
            self.phases=[self.phases_orb_LC]*len(det_use)   
            self.I_combined_total, self.Ierr_combined_total  = add_datasets(self.I_src, self.Ie_src)
            self.Q_combined_total, self.Qerr_combined_total  = add_datasets(self.Q_src, self.Qe_src)#, weights=self.I_src)
            self.U_combined_total, self.Uerr_combined_total  = add_datasets(self.U_src, self.Ue_src)#, weights=self.I_src)
            
            self.compute_chi2()
            
            self.PD_src_total, self.PDerr_src_total =PD_calc(self.Q_combined_total, self.U_combined_total, self.I_combined_total, self.Qerr_combined_total, self.Uerr_combined_total, self.Ierr_combined_total)
            self.EVPA_combined_total, self.EVPAerr_combined_total =EVPA_calc(self.Q_combined_total, self.U_combined_total,  self.Qerr_combined_total, self.Uerr_combined_total)
            
        '''    
            detector1=standard_pipeline_leakage(matching_files_d1[0], matching_hk_d1[0], 1, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector2=standard_pipeline_leakage(matching_files_d2[0], matching_hk_d2[0], 2, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector3=standard_pipeline_leakage(matching_files_d3[0], matching_hk_d3[0], 3, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], backgroundthresh=bg_thresh_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])

            I_alldetectors=[keep_det1*detector1.detector.flux_orb_LC_src, keep_det2*detector2.detector.flux_orb_LC_src, keep_det3*detector3.detector.flux_orb_LC_src]
            Ie_alldetectors=[keep_det1*detector1.detector.error_orb_LC_src, keep_det2*detector2.detector.error_orb_LC_src, keep_det3*detector3.detector.error_orb_LC_src]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            
            I_alldetectors_flux=[keep_det1*detector1.detector.flux_orb_LC_src/(1.+detector1.detector.flux_orb_LC_src)*detector1.detector.N_perbin, keep_det2*detector2.detector.flux_orb_LC_src/(1.+detector2.detector.flux_orb_LC_src)*detector2.detector.N_perbin, keep_det3*detector3.detector.flux_orb_LC_src/(1.+detector3.detector.flux_orb_LC_src)*detector3.detector.N_perbin]
            Ie_alldetectors_flux=[keep_det1*detector1.detector.error_orb_LC_src/(1.+detector1.detector.flux_orb_LC_src)*detector1.detector.N_perbin, keep_det2*detector2.detector.error_orb_LC_src/(1.+detector2.detector.flux_orb_LC_src)*detector2.detector.N_perbin, keep_det3*detector3.detector.error_orb_LC_src/(1.+detector3.detector.flux_orb_LC_src)*detector3.detector.N_perbin]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            
            Q_alldetectors_flux = [x * y for x, y in zip( Q_alldetectors, I_alldetectors_flux)]
            Qe_alldetectors_flux = [x * y for x, y in zip( Qe_alldetectors, I_alldetectors_flux)]
            
            U_alldetectors_flux = [x * y for x, y in zip( U_alldetectors, I_alldetectors_flux)]
            Ue_alldetectors_flux = [x * y for x, y in zip( Ue_alldetectors, I_alldetectors_flux)]
            
            I_combined, Ierr_combined = add_datasets(I_alldetectors_flux, Ie_alldetectors_flux)
            Q_combined, Qerr_combined = add_datasets(Q_alldetectors_flux, Qe_alldetectors_flux)#,  weights=I_alldetectors)
            U_combined, Uerr_combined = add_datasets(U_alldetectors_flux, Ue_alldetectors_flux)#,  weights=I_alldetectors)
            
   
            N_total = detector1.detector.N_perbin+detector2.detector.N_perbin+detector3.detector.N_perbin
            self.N_evt.append(N_total)
            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined, Ierr_combined**2)
            exposure_time=keep_det1*detector1.detector.exposure_time+keep_det2*detector2.detector.exposure_time+keep_det3*detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time #total exposure time in a bin
            #print('Total exposure time {}'.format(np.sum(exposure_time/3)))
            exposure_time=1.
            
            self.I_src.append(I_combined/exposure_time)
            self.Ie_src.append(Ierr_combined/exposure_time)
            self.Q_src.append(Q_combined/exposure_time)
            self.Qe_src.append(Qerr_combined/exposure_time)
            self.U_src.append(U_combined/exposure_time)
            self.Ue_src.append(Uerr_combined/exposure_time)

            
            I_bkg_alldetectors=[detector1.detector.flux_orb_LC_bkg, detector2.detector.flux_orb_LC_bkg, detector3.detector.flux_orb_LC_bkg]
            Ie_bkg_alldetectors=[detector1.detector.error_orb_LC_bkg, detector2.detector.error_orb_LC_bkg, detector3.detector.error_orb_LC_bkg]
            #Q_bkg_alldetectors = [detector1.detector.Q_LC_orb_bkg, detector2.detector.Q_LC_orb_bkg, detector3.detector.Q_LC_orb_bkg]
            #Qe_bkg_alldetectors = [detector1.detector.Qerr_LC_orb_bkg, detector2.detector.Qerr_LC_orb_bkg, detector3.detector.Qerr_LC_orb_bkg]
            #U_bkg_alldetectors = [detector1.detector.U_LC_orb_bkg, detector2.detector.U_LC_orb_bkg, detector3.detector.U_LC_orb_bkg]
            #Ue_bkg_alldetectors = [detector1.detector.Uerr_LC_orb_bkg, detector2.detector.Uerr_LC_orb_bkg, detector3.detector.Uerr_LC_orb_bkg]
                
            I_bkg_combined, Ierr_bkg_combined = add_datasets(I_bkg_alldetectors, Ie_bkg_alldetectors)
            #Q_bkg_combined, Qerr_bkg_combined = add_datasets(Q_bkg_alldetectors, Qe_bkg_alldetectors)
            #U_bkg_combined, Uerr_bkg_combined = add_datasets(U_bkg_alldetectors, Ue_bkg_alldetectors)

                

            #PD_combined_bkg, PDerr_combined_bkg =PD_calc(Q_bkg_combined, U_bkg_combined, I_bkg_combined, Qerr_combined, Uerr_bkg_combined, Ierr_bkg_combined) 
            #EVPA_combined_bkg, EVPAerr_combined_bkg =EVPA_calc(Q_bkg_combined, U_bkg_combined,  Qerr_bkg_combined, Uerr_bkg_combined)
            self.I_bkg.append(I_bkg_combined/exposure_time)
            self.Ie_bkg.append(Ierr_bkg_combined/exposure_time)
            #self.PD_bkg.append(PD_combined_bkg)
            #self.PDerr_bkg.append(PDerr_combined_bkg)
            #self.EVPA_bkg.append(EVPA_combined_bkg)
            #self.EVPAerr_bkg.append(EVPAerr_combined_bkg)
            if bkg_sub_list[k]==1:
                #print('Yes subtracted background the old way')
                I_combined_sub, Ierr_combined_sub  = subtract_bkg(I_combined, Ierr_combined, I_bkg_combined, Ierr_bkg_combined, srcd1_r_list[k],bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                #Q_combined_sub,  Qerr_combined_sub = subtract_bkg(Q_combined, Qerr_combined, Q_bkg_combined, Qerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k])
                #U_combined_sub,  Uerr_combined_sub = subtract_bkg(U_combined, Uerr_combined, U_bkg_combined, Uerr_bkg_combined, srcd1_r_list[k], bkgd1_r_list[k], annulus=annulus_list[k], r_bkg_in=bkgd1_rin_list[k]) 
                #print('Printing Q background subtracted')
                #print(Q_combined_sub)
            else:
                #print('No background subtraction background the old way')
                I_combined_sub, Ierr_combined_sub= I_combined, Ierr_combined
                Q_combined_sub, Qerr_combined_sub= Q_combined, Qerr_combined
                U_combined_sub, Uerr_combined_sub= U_combined, Uerr_combined
                #print('Printing Q no background subtracted')
                #print(Q_combined_sub)


            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined_sub, Ierr_combined_sub**2)
            #MDP_bkg_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_bkg_combined, Ierr_bkg_combined**2)
            
            self.I.append(I_combined_sub/exposure_time)
            self.Ie.append(Ierr_combined_sub/exposure_time)
            #self.MDP.append(MDP_combined)
            #self.MDP_bkg.append(MDP_bkg_combined)
            if compute_spectrum_list[k]:
                self.ct_spec_src.append(detector1.detector.spec_counts+detector2.detector.spec_counts+detector3.detector.spec_counts)
                self.E_spec_src.append(detector1.detector.spec_Energy[:len(detector1.detector.spec_Energy)-1])
            #self.phases.append()

            PD_combined, PDerr_combined =PD_calc(Q_combined_sub, U_combined_sub, I_combined_sub, Qerr_combined_sub, Uerr_combined_sub, Ierr_combined_sub)
            EVPA_combined, EVPAerr_combined =EVPA_calc(Q_combined_sub, U_combined_sub,  Qerr_combined_sub, Uerr_combined_sub)
            self.phases.append(detector1.detector.phases_orb_LC)
            self.PD_src.append(PD_combined)
            self.PDerr_src.append(PDerr_combined)
            self.EVPA_src.append(EVPA_combined)
            self.EVPAerr_src.append(EVPAerr_combined)
            
        if combine_all:
            self.I_combined_total, self.Ierr_combined_total = add_datasets(self.I, self.Ie)
            self.Q_combined_total, self.Qerr_combined_total = add_datasets(self.Q_src, self.Qe_src)#, self.I)
            self.U_combined_total, self.Uerr_combined_total = add_datasets(self.U_src, self.Ue_src)#, self.I)
            self.PD_src_total, self.PDerr_src_total =PD_calc(self.Q_combined_total, self.U_combined_total, self.I_combined_total, self.Qerr_combined_total, self.Uerr_combined_total, self.Ierr_combined_total)
            self.EVPA_combined_total, self.EVPAerr_combined_total =EVPA_calc(self.Q_combined_total, self.U_combined_total,  self.Qerr_combined_total, self.Uerr_combined_total)
        '''
        
    def make_lc_func(self, nbins, calc_nonGaussian_errors=True, manual_bins=None):
        self.make_orb_LC(self.list_dataset_unpacked, nbins, self.Pb_use, self.Ephemeris, Emin=self.Emin, Emax=self.Emax, weights=self.weights, use_weights=self.use_weights,  fit_bg_qu=self.fit_bg_qu, fit_bg_det=self.fit_bg_det, fit_bg_det_sep=self.fit_bg_det_sep, model_fit=self.model_fit, flare_cut=self.flare_cut, fit_prtl_bg=self.fit_prtl_bg, fit_prtl_qu=self.fit_prtl_qu, provide_background=self.background_load, manual_bins=manual_bins)
    
        self.phases=[self.phases_orb_LC]*len(self.direct_list)
        if self.combine_detectors_leakage:
            self.I_combined_total = self.flux_orb_LC_src/np.max(self.flux_orb_LC_src)#*self.N_perbin/(1+self.flux_orb_LC_src)
            self.Ierr_combined_total = self.error_orb_LC_src/np.max(self.flux_orb_LC_src)#*self.N_perbin/(1+self.flux_orb_LC_src)
    
    
            self.Q_combined_total = self.Q_LC_orb_src*self.I_combined_total
            self.Qerr_combined_total = self.Qerr_LC_orb_src*self.I_combined_total
            self.U_combined_total = self.U_LC_orb_src*self.I_combined_total
            self.Uerr_combined_total = self.Uerr_LC_orb_src*self.I_combined_total
    
        self.I_src = [self.I_combined_total]*len(self.direct_list)
        self.Ie_src = [self.Ierr_combined_total]*len(self.direct_list)
        self.Q_src = [self.Q_combined_total]*len(self.direct_list)
        self.Qe_src = [self.Qerr_combined_total]*len(self.direct_list)
        self.U_src = [self.U_combined_total]*len(self.direct_list)
        self.Ue_src = [self.Uerr_combined_total]*len(self.direct_list)
        if calc_nonGaussian_errors:
            self.PD_src_total,  self.PDerr_src_total_upper,  self.PDerr_src_total_lower,  self.EVPA_combined_total, self.EVPAerr_combined_total_upper, self.EVPAerr_combined_total_lower= polarization_lc(self.Q_LC_orb_src, self.U_LC_orb_src,  self.Qerr_LC_orb_src, self.Uerr_LC_orb_src)
        else:
            self.PD_src_total, self.PDerr_src_total =PD_calc(self.Q_LC_orb_src, self.U_LC_orb_src, 1, self.Qerr_LC_orb_src, self.Uerr_LC_orb_src, 0.0)
            self.EVPA_combined_total, self.EVPAerr_combined_total =EVPA_calc(self.Q_LC_orb_src, self.U_LC_orb_src,  self.Qerr_LC_orb_src, self.Uerr_LC_orb_src)    
        if manual_bins is not None:
            if calc_nonGaussian_errors:
                self.PD_src_total_manual,  self.PDerr_src_total_upper_manual,  self.PDerr_src_total_lower_manual,  self.EVPA_combined_total_manual, self.EVPAerr_combined_total_upper_manual, self.EVPAerr_combined_total_lower_manual = polarization_lc(self.Q_LC_orb_src_manual, self.U_LC_orb_src_manual,  self.Qerr_LC_orb_src_manual, self.Uerr_LC_orb_src_manual)
            else:
                self.PD_src_total_manual, self.PDerr_src_total_manual =PD_calc(self.Q_LC_orb_src_manual, self.U_LC_orb_src_manual, 1, self.Qerr_LC_orb_src_manual, self.Uerr_LC_orb_src_manual, 0.0)
                self.EVPA_combined_total_manual, self.EVPAerr_combined_total_manual =EVPA_calc(self.Q_LC_orb_src_manual, self.U_LC_orb_src_manual,  self.Qerr_LC_orb_src_manual, self.Uerr_LC_orb_src_manual)    
            
        
    def flare_cut(self, list_dataset_unpacked, timebin, Porb, TASC, Emin=2, Emax =8,  weights=1, use_weights=True, bkg_sub=0, fit_bg_qu=False, fit_bg_det=False, fit_bg_det_sep=True, model_fit=False, show_hist=True, cut_amount=35, xlim=[0, 100], ylim=[0, 100], hist_bins=100, exclude_r=None):
           
      
            if weights == 1:
                NN=False
            elif weights ==2:
                NN=True
            source = leakagelib_v9.source.Source.no_image(NN)
            
            
           
            #self.phases_LC_tot = Convert_Phase_normalized(self.t_LC, TASC, Porb)
            
            # overreite evt_times wityh phases
            # set_sweep(namne, (q_func, u_func))
            # set_lightcurve(name, i_func)
            # these functions will be evaluated on evt times but since I convert to phase they will evaluate based on phase
            self.mask_flares=[]
            total_counts_initial=0
            total_counts_final=0
            timebins_removed=0
            timebins_total=0
            hist_cts_arr=[]
            for n, dataset in enumerate(list_dataset_unpacked):
               
               E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
               dataset.make_LC_v2(timebin, Emin, Emax, exclude_r=exclude_r)
               if self.du_id_list[n]==self.du_id_list[0]:    
                   hist_cts=np.zeros(len(dataset.total_counts_LC))
               src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, dataset.src_r, mask=E_mask)
               src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
               hist_cts+=np.array(dataset.total_counts_LC)
               print("The length of the light curve for dataset {}: {}".format(n, len(dataset.total_counts_LC)))
                
               if n == len(list_dataset_unpacked)-1:
                   hist_cts_arr.append(hist_cts)
               elif self.du_id_list[n+1]==self.du_id_list[0]:
                   hist_cts_arr.append(hist_cts)
            
            if show_hist:
                
                for histcts in hist_cts_arr:
                   fig, ax = plt.subplots(1)
                   ax.hist(histcts, bins =hist_bins)
                   ax.set_xlim(xlim[0], xlim[1])
                   ax.set_ylim(ylim[0], ylim[1])
                   
                   fig, ax = plt.subplots(1)
                   ax.plot(histcts)
                   ax.set_ylim(xlim[0], xlim[1])
                   
                   #ax.set_ylim(ylim[0], ylim[1])
            
                print("The length of the histogram list: {}".format(len(hist_cts_arr)))
            
            m=0
            for n, dataset in enumerate(list_dataset_unpacked):
               if self.du_id_list[n]==self.du_id_list[0]:    
                   hist_cts_use=hist_cts_arr[m]
                   m+=1
               keep_counts = np.where(hist_cts_use>cut_amount, False, True)
               remaining_bins = dataset.bin_list[keep_counts]
               #print("Remaining bins after cut: {}".format(len(remaining_bins)))
               counts_bins=dataset.time_index
               print('Original counts: {}'.format(len(counts_bins)))
               total_counts_initial+=len(counts_bins)
               self.counts_bins=counts_bins
               self.total_counts_LC=dataset.total_counts_LC
               timebins_removed+=len(dataset.bin_list)-len(remaining_bins)
               timebins_total+=len(dataset.bin_list)
               mask = np.isin(counts_bins, remaining_bins)
               print('Remaining counts: {}'.format(len(counts_bins[mask])))
               total_counts_final+=len(counts_bins[mask])
               self.mask_flares.append(mask)
            
            print("Total counts initial {}".format(total_counts_initial))
            print("Total counts removed {}".format(total_counts_initial-total_counts_final))
            print("Fraction removed: {}".format((total_counts_initial-total_counts_final)/total_counts_initial))
            print("Exposure time removed {} s".format(timebin*(timebins_removed/3)))
            print("Exposure time removed {} s".format((timebins_removed)/timebins_total))
            
               
               
    def make_orb_LC(self, list_dataset_unpacked, nbins, Porb, TASC, Emin=2, Emax =8,  weights=1, use_weights=True, bkg_sub=0, fit_bg_qu=False, fit_bg_det=False, fit_bg_det_sep=True, model_fit=False, flare_cut=True, fit_prtl_bg=True, fit_prtl_qu=True, provide_background=None, manual_bins=None):
        binwidth=1./(nbins)
        N_perbin=np.empty(nbins)
        self.phases_orb_LC=np.linspace(0.0, 1.0-binwidth, nbins)
        self.flux_orb_LC_src = np.empty(nbins)
        self.error_orb_LC_src = np.empty(nbins)
        self.flux_orb_LC_bkg = np.empty(nbins)
        self.error_orb_LC_bkg = np.empty(nbins)

        self.Q_LC_orb_src = np.empty(nbins)
        self.U_LC_orb_src = np.empty(nbins)
        self.Qerr_LC_orb_src = np.empty(nbins)
        self.Uerr_LC_orb_src = np.empty(nbins)
        if weights == 1:
            NN=False
        elif weights ==2:
            NN=True
        source = leakagelib_v9.source.Source.no_image(NN)
        
        
       
        #self.phases_LC_tot = Convert_Phase_normalized(self.t_LC, TASC, Porb)
        
        # overreite evt_times wityh phases
        # set_sweep(namne, (q_func, u_func))
        # set_lightcurve(name, i_func)
        # these functions will be evaluated on evt times but since I convert to phase they will evaluate based on phase
        self.result=[]
        self.exposure_time=np.empty(nbins)
        for tbin in range(0, nbins):
            N_perbin=0
            IXPE_full_dataset=[]
            if flare_cut:
                flare_cut_counter=0
            for dataset in list_dataset_unpacked:
                E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
                src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, dataset.src_r, mask=E_mask)
                src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
                dataset.total_image.phase = Convert_Phase_normalized(dataset.total_image.time, TASC, Porb)
                
                IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (dataset.filename, dataset.hkname), energy_cut=(Emin, Emax))
                if flare_cut:
                    combined_src_flares_mask=np.where((src_reg_Elim) & (self.mask_flares[flare_cut_counter]), True, False)
                    IXPE_data.retain(combined_src_flares_mask)
                    flare_cut_counter+=1
                else:
                    IXPE_data.retain(src_reg_Elim)
                #IXPE_data.cut(E_mask)
                IXPE_data.explicit_center(dataset.src_x, dataset.src_y)

                
                
                #IXPE_data_copy = copy.deepcopy(IXPE_data)
       
                if flare_cut:
                    time_mask = np.where((tbin*binwidth<dataset.total_image.phase[E_mask][combined_src_flares_mask]) & ((tbin+1)*binwidth>dataset.total_image.phase[E_mask][combined_src_flares_mask]), True, False)
                else:
                    time_mask = np.where((tbin*binwidth<src_photons_Elim.phase) & ((tbin+1)*binwidth>src_photons_Elim.phase), True, False)
                    
                
                IXPE_data.retain(time_mask)
                if NN:
                    IXPE_data.weight_nn()
                N_perbin += len(IXPE_data.evt_xs)
                IXPE_full_dataset.append(IXPE_data)
                
                
            settings = leakagelib_v9.ps_fit.FitSettings(IXPE_full_dataset)
            settings.add_point_source() # Point source component
            #settings.add_background() # Background component
            #settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized
            
            if fit_bg_det:
                
                
                    settings.add_background("bkg1", det=(1,)) # Background component from detector 1
                    settings.add_background("bkg2", det=(2,)) # Background component from detector 2
                    settings.add_background("bkg3", det=(3,))
                    
                    if fit_bg_qu:
                        if fit_bg_det_sep:
                            if provide_background is None:
                                settings.fix_qu("bkg1", (self.q_det1_bkg, self.u_det1_bkg))
                                settings.fix_qu("bkg2", (self.q_det2_bkg, self.u_det2_bkg))
                                settings.fix_qu("bkg3", (self.q_det3_bkg, self.u_det3_bkg))
                            else:
                                settings.fix_qu("bkg1", (provide_background.q_det1_bkg, provide_background.u_det1_bkg))
                                settings.fix_qu("bkg2", (provide_background.q_det2_bkg, provide_background.u_det2_bkg))
                                settings.fix_qu("bkg3", (provide_background.q_det3_bkg, provide_background.u_det3_bkg))
                                
                        else:
                            settings.set_initial_qu("bkg1", (0., 0.))
                            settings.set_initial_qu("bkg2", (0., 0.))
                            settings.set_initial_qu("bkg3", (0., 0.))
                    else:
                        settings.fix_qu("bkg1", (0, 0))
                        settings.fix_qu("bkg2", (0, 0))
                        settings.fix_qu("bkg3", (0, 0))
                    
                    if fit_prtl_bg:
                        settings.add_particle_source("prtl1", det=(1,))
                        settings.add_particle_source("prtl2", det=(2,))
                        settings.add_particle_source("prtl3", det=(3,))
                        
                        if fit_prtl_qu:
                            if fit_bg_det_sep:
                                if provide_background is None:
                                    settings.fix_qu("bkg1", (self.q_det1_pbkg, self.u_det1_pbkg))
                                    settings.fix_qu("bkg2", (self.q_det2_pbkg, self.u_det2_pbkg))
                                    settings.fix_qu("bkg3", (self.q_det3_pbkg, self.u_det3_pbkg))
                                else:
                                    settings.fix_qu("prtl1", (provide_background.q_det1_pbkg, provide_background.u_det1_pbkg))
                                    settings.fix_qu("prtl2", (provide_background.q_det2_pbkg, provide_background.u_det2_pbkg))
                                    settings.fix_qu("prtl3", (provide_background.q_det3_pbkg, provide_background.u_det3_pbkg))
                            else:
                                settings.set_initial_qu("prtl1", (0., 0.))
                                settings.set_initial_qu("prtl2", (0., 0.))
                                settings.set_initial_qu("prtl3", (0., 0.))
                        else:   
                            settings.fix_qu("prtl1", (0., 0.))
                            settings.fix_qu("prtl2", (0., 0.))
                            settings.fix_qu("prtl3", (0., 0.))
                    
                    settings.fix_flux("bkg1", 1)
                    if fit_bg_det_sep:
                        if provide_background is None:
                            settings.fix_flux("bkg1", self.f_det1_bkg)
                            settings.fix_flux("bkg2", self.f_det2_bkg)
                            settings.fix_flux("bkg3", self.f_det3_bkg)
                            if fit_prtl_bg:
                                settings.fix_flux("prtl1", self.f_det1_pbkg)
                                settings.fix_flux("prtl2", self.f_det2_pbkg)
                                settings.fix_flux("prtl3", self.f_det3_pbkg)
                        else:
                            settings.fix_flux("bkg1", provide_background.f_det1_bkg)
                            settings.fix_flux("bkg2", provide_background.f_det2_bkg)
                            settings.fix_flux("bkg3", provide_background.f_det3_bkg)
                            if fit_prtl_bg:
                                settings.fix_flux("prtl1", provide_background.f_det1_pbkg)
                                settings.fix_flux("prtl2", provide_background.f_det2_pbkg)
                                settings.fix_flux("prtl3", provide_background.f_det3_pbkg)
            else:
                settings.add_background()
                settings.fix_flux("bkg", 1)
                
            
               
                if not fit_bg_qu:
                    settings.fix_qu("bkg", (0.0, 0.0)) # Set the background to be unpolarized
            
            
           
            settings.set_initial_flux("src", 0.1)
            #settings.set_initial_qu("src", (0.3, 0.3))
            
            settings.apply_circular_roi(dataset.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
            fitter = leakagelib_v9.ps_fit.Fitter(IXPE_full_dataset, settings)
            result = fitter.fit()
            #phases_exposure=np.where((self.phases_orb_LC[tbin]<self.phases_LC_tot)& (self.phases_orb_LC[tbin]+binwidth>self.phases_LC_tot))
            
            #exposure= np.sum(self.exposure_clean[phases_exposure])
            print(result)
            #print('exposure time: {}'.format(exposure ))
            #self.exposure_time[tbin]=self.timebin*exposure 
            self.flux_orb_LC_src[tbin]=result.params[('f', 'src')]#/self.exposure_time[tbin]
            self.Q_LC_orb_src[tbin]=result.params[('q', 'src')]#/self.exposure_time[tbin]
            self.U_LC_orb_src[tbin]=result.params[('u', 'src')]#/self.exposure_time[tbin]
            self.error_orb_LC_src[tbin]=result.sigmas[('f', 'src')]#/self.exposure_time[tbin]
            self.Qerr_LC_orb_src[tbin]=result.sigmas[('q', 'src')]#/self.exposure_time[tbin]
            self.Uerr_LC_orb_src[tbin]=result.sigmas[('u', 'src')]#/self.exposure_time[tbin]
            self.result.append(result)  
            self.flux_orb_LC_bkg[tbin]=1.#result.params[('f', 'bkg')]
            #self.Q_LC_orb_bkg[tbin]=0#result.params[('q', 'bkg')]
            #self.U_LC_orb_bkg[tbin]=0#esult.params[('u', 'bkg')]
            
            self.error_orb_LC_bkg[tbin] = 0.
            #print(result)
            #print(result.params[('f', 'src')])
            #print(result[('q', 'src')])
            #print(result[('u', 'src')])
            
            #len(self.src_photons_Elim.phase[time_mask])
        if manual_bins is not None:
            self.flux_orb_LC_src_manual = np.empty(len(manual_bins))
            self.error_orb_LC_src_manual = np.empty(len(manual_bins))


            self.Q_LC_orb_src_manual = np.empty(len(manual_bins))
            self.U_LC_orb_src_manual = np.empty(len(manual_bins))
            self.Qerr_LC_orb_src_manual = np.empty(len(manual_bins))
            self.Uerr_LC_orb_src_manual = np.empty(len(manual_bins))
            bins_right=np.roll(manual_bins, -1)
            bins_right[len(manual_bins)-1]+=1.
            self.phases_orb_LC_manual=0.5*(manual_bins+bins_right)
            
            for n, time in enumerate(manual_bins):
                N_perbin=0
                IXPE_full_dataset=[]
                
                if n+1<len(manual_bins):
                    upper_bound=manual_bins[n+1]
                else:
                    upper_bound=manual_bins[0]
                print("manual bin range {}-{}".format(manual_bins[n], upper_bound))    
                if flare_cut:
                    flare_cut_counter=0
                for dataset in list_dataset_unpacked:
                    E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
                    src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, dataset.src_r, mask=E_mask)
                    src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
                    dataset.total_image.phase = Convert_Phase_normalized(dataset.total_image.time, TASC, Porb)
                    
                    IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (dataset.filename, dataset.hkname), energy_cut=(Emin, Emax))
                    if flare_cut:
                        combined_src_flares_mask=np.where((src_reg_Elim) & (self.mask_flares[flare_cut_counter]), True, False)
                        IXPE_data.retain(combined_src_flares_mask)
                        flare_cut_counter+=1
                    else:
                        IXPE_data.retain(src_reg_Elim)
                    #IXPE_data.cut(E_mask)
                    IXPE_data.explicit_center(dataset.src_x, dataset.src_y)
    
                    
                    
                    #IXPE_data_copy = copy.deepcopy(IXPE_data)
                    
                    if flare_cut:
                        if upper_bound<manual_bins[n]:
                            time_mask = np.where((manual_bins[n]<dataset.total_image.phase[E_mask][combined_src_flares_mask]) | (upper_bound>dataset.total_image.phase[E_mask][combined_src_flares_mask]), True, False)
                        else:
                            time_mask = np.where((manual_bins[n]<dataset.total_image.phase[E_mask][combined_src_flares_mask]) & (upper_bound>dataset.total_image.phase[E_mask][combined_src_flares_mask]), True, False)
                    else:
                        time_mask = np.where((manual_bins[n]<src_photons_Elim.phase) & (upper_bound>src_photons_Elim.phase), True, False)
                        
                    
                    IXPE_data.retain(time_mask)
                    if NN:
                        IXPE_data.weight_nn()
                    N_perbin += len(IXPE_data.evt_xs)
                    IXPE_full_dataset.append(IXPE_data)
                    
                    
                settings = leakagelib_v9.ps_fit.FitSettings(IXPE_full_dataset)
                settings.add_point_source() # Point source component
                #settings.add_background() # Background component
                #settings.fix_qu("bkg", (0, 0)) # Set the background to be unpolarized
                
                if fit_bg_det:
                    
                    
                        settings.add_background("bkg1", det=(1,)) # Background component from detector 1
                        settings.add_background("bkg2", det=(2,)) # Background component from detector 2
                        settings.add_background("bkg3", det=(3,))
                        
                        if fit_bg_qu:
                            if fit_bg_det_sep:
                                if provide_background is None:
                                    settings.fix_qu("bkg1", (self.q_det1_bkg, self.u_det1_bkg))
                                    settings.fix_qu("bkg2", (self.q_det2_bkg, self.u_det2_bkg))
                                    settings.fix_qu("bkg3", (self.q_det3_bkg, self.u_det3_bkg))
                                else:
                                    settings.fix_qu("bkg1", (provide_background.q_det1_bkg, provide_background.u_det1_bkg))
                                    settings.fix_qu("bkg2", (provide_background.q_det2_bkg, provide_background.u_det2_bkg))
                                    settings.fix_qu("bkg3", (provide_background.q_det3_bkg, provide_background.u_det3_bkg))
                                    
                            else:
                                settings.set_initial_qu("bkg1", (0., 0.))
                                settings.set_initial_qu("bkg2", (0., 0.))
                                settings.set_initial_qu("bkg3", (0., 0.))
                        else:
                            settings.fix_qu("bkg1", (0, 0))
                            settings.fix_qu("bkg2", (0, 0))
                            settings.fix_qu("bkg3", (0, 0))
                        
                        if fit_prtl_bg:
                            settings.add_particle_source("prtl1", det=(1,))
                            settings.add_particle_source("prtl2", det=(2,))
                            settings.add_particle_source("prtl3", det=(3,))
                            
                            if fit_prtl_qu:
                                if fit_bg_det_sep:
                                    if provide_background is None:
                                        settings.fix_qu("bkg1", (self.q_det1_pbkg, self.u_det1_pbkg))
                                        settings.fix_qu("bkg2", (self.q_det2_pbkg, self.u_det2_pbkg))
                                        settings.fix_qu("bkg3", (self.q_det3_pbkg, self.u_det3_pbkg))
                                    else:
                                        settings.fix_qu("prtl1", (provide_background.q_det1_pbkg, provide_background.u_det1_pbkg))
                                        settings.fix_qu("prtl2", (provide_background.q_det2_pbkg, provide_background.u_det2_pbkg))
                                        settings.fix_qu("prtl3", (provide_background.q_det3_pbkg, provide_background.u_det3_pbkg))
                                else:
                                    settings.set_initial_qu("prtl1", (0., 0.))
                                    settings.set_initial_qu("prtl2", (0., 0.))
                                    settings.set_initial_qu("prtl3", (0., 0.))
                            else:   
                                settings.fix_qu("prtl1", (0., 0.))
                                settings.fix_qu("prtl2", (0., 0.))
                                settings.fix_qu("prtl3", (0., 0.))
                        
                        settings.fix_flux("bkg1", 1)
                        if fit_bg_det_sep:
                            if provide_background is None:
                                settings.fix_flux("bkg1", self.f_det1_bkg)
                                settings.fix_flux("bkg2", self.f_det2_bkg)
                                settings.fix_flux("bkg3", self.f_det3_bkg)
                                if fit_prtl_bg:
                                    settings.fix_flux("prtl1", self.f_det1_pbkg)
                                    settings.fix_flux("prtl2", self.f_det2_pbkg)
                                    settings.fix_flux("prtl3", self.f_det3_pbkg)
                            else:
                                settings.fix_flux("bkg1", provide_background.f_det1_bkg)
                                settings.fix_flux("bkg2", provide_background.f_det2_bkg)
                                settings.fix_flux("bkg3", provide_background.f_det3_bkg)
                                if fit_prtl_bg:
                                    settings.fix_flux("prtl1", provide_background.f_det1_pbkg)
                                    settings.fix_flux("prtl2", provide_background.f_det2_pbkg)
                                    settings.fix_flux("prtl3", provide_background.f_det3_pbkg)
                else:
                    settings.add_background()
                    settings.fix_flux("bkg", 1)
                    
                
                   
                    if not fit_bg_qu:
                        settings.fix_qu("bkg", (0.0, 0.0)) # Set the background to be unpolarized
                
                
               
                settings.set_initial_flux("src", 0.1)
                #settings.set_initial_qu("src", (0.3, 0.3))
                
                settings.apply_circular_roi(dataset.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
                fitter = leakagelib_v9.ps_fit.Fitter(IXPE_full_dataset, settings)
                result = fitter.fit()
                #phases_exposure=np.where((self.phases_orb_LC[tbin]<self.phases_LC_tot)& (self.phases_orb_LC[tbin]+binwidth>self.phases_LC_tot))
                
                #exposure= np.sum(self.exposure_clean[phases_exposure])
                print(result)
                #print('exposure time: {}'.format(exposure ))
                #self.exposure_time[tbin]=self.timebin*exposure 
                self.flux_orb_LC_src_manual[n]=result.params[('f', 'src')]#/self.exposure_time[tbin]
                self.Q_LC_orb_src_manual[n]=result.params[('q', 'src')]#/self.exposure_time[tbin]
                self.U_LC_orb_src_manual[n]=result.params[('u', 'src')]#/self.exposure_time[tbin]
                self.error_orb_LC_src_manual[n]=result.sigmas[('f', 'src')]#/self.exposure_time[tbin]
                self.Qerr_LC_orb_src_manual[n]=result.sigmas[('q', 'src')]#/self.exposure_time[tbin]
                self.Uerr_LC_orb_src_manual[n]=result.sigmas[('u', 'src')]#/self.exposure_time[tbin]
                #self.result_manual.append(result)  
                #self.flux_orb_LC_bkg_manual[tbin]=1.#result.params[('f', 'bkg')]
                #self.Q_LC_orb_bkg[tbin]=0#result.params[('q', 'bkg')]
                #self.U_LC_orb_bkg[tbin]=0#esult.params[('u', 'bkg')]
                
                #self.error_orb_LC_bkg[tbin] = 0.
        
        else:
            self.phases_orb_LC_manual=None
            
        self.N_perbin=N_perbin

        self.phases_orb_LC+=binwidth/2.
        
        
        
    def compute_chi2(self):
        chi2_q_val = self.Q_combined_total**2/self.Qerr_combined_total**2
        chi2_u_val = self.U_combined_total**2/self.Uerr_combined_total**2
        
        chi2 = np.sum(chi2_q_val+chi2_u_val)
        self.chi2_perdof=chi2/(len(chi2_q_val)*2)
        
        p_value=1.-scipy.stats.chi2.cdf(chi2, df=len(chi2_q_val))
        self.sigma = norm.isf(p_value)

    def IXPE_bg_fit(self, list_dataset_unpacked, src_remove, nbins, Porb, TASC, Emin=2, Emax =8,  weights=1, use_weights=True, bkg_sub=0, fit_sep_det=True, bg_weights=True, bg_weights_thresh=0.99, bg_cut=True, flare_cut=True, fit_prtl_bg=True, fit_prtl_qu=True, save_bkg=True, background_name="main_", background_dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/', do_pcube=True, fix_det23=True):
        '''
        src_remove is the radius in pixels to remove to exclude the source
        '''
        
        if weights == 1:
            NN=False
        elif weights ==2:
            NN=True
        source = leakagelib_v9.source.Source.no_image(NN)
        N_perbin=0
        IXPE_full_dataset=[]
        IXPE_det1_dataset=[]
        IXPE_det2_dataset=[]
        IXPE_det3_dataset=[]
        if flare_cut:
            flare_cut_counter=0
        n_evt_counter=0
        self.det1_N_bkg=0
        self.det2_N_bkg=0
        self.det3_N_bkg=0
        for dataset in list_dataset_unpacked:
            E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
            src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, dataset.src_r, mask=E_mask)
            src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
            dataset.total_image.phase = Convert_Phase_normalized(dataset.total_image.time, TASC, Porb)
            IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (dataset.filename, dataset.hkname), energy_cut=(Emin, Emax))
            #IXPE_data.retain(src_reg_Elim)
            #IXPE_data.evt_times=src_photons_Elim.phase
            #IXPE_data.cut(E_mask)
            
            
            
            if flare_cut:
                print("Flare mask length {}".format(len(dataset.total_image.phase[E_mask][self.mask_flares[flare_cut_counter]])))
                combined_src_flares_mask=np.where((src_reg_Elim) & (self.mask_flares[flare_cut_counter]), True, False)
                IXPE_data.retain(combined_src_flares_mask)
                flare_cut_counter+=1
                IXPE_data.evt_times=dataset.total_image.phase[E_mask][combined_src_flares_mask]
                
            else:
                IXPE_data.retain(src_reg_Elim)
                IXPE_data.evt_times=src_photons_Elim.phase
            
            IXPE_data.explicit_center(dataset.src_x, dataset.src_y)
            
            self.src_remove_mask = np.where(IXPE_data.evt_xs**2+ IXPE_data.evt_ys**2 < src_remove**2, False, True)
            print("Number of photons {}".format(len(IXPE_data.evt_xs)))
            IXPE_data.retain(self.src_remove_mask)
            print("Number of photons after cutting {}".format(len(IXPE_data.evt_xs)))
            print("removing: {}".format( src_remove))
            #IXPE_data_copy = copy.deepcopy(IXPE_data)
            
            
            
            
            
            
            
            
            if not fit_prtl_bg:
                IXPE_data.evt_bg_chars=np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, IXPE_data.evt_bg_chars, 0)
                #IXPE_data
                #IXPE_data.cut()
            elif bg_cut:
                evts_cut = np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, 0, 1)
                IXPE_data.retain(evts_cut)
            if NN:
                IXPE_data.weight_nn()
            N_perbin += len(IXPE_data.evt_xs)
            if do_pcube:
                IXPE_data.evt_ws = np.ones( len(IXPE_data.evt_ws ))
            IXPE_full_dataset.append(IXPE_data)
            
            print("Max distance from center {}".format(np.max(np.sqrt(IXPE_data.evt_ys**2+IXPE_data.evt_xs**2))))
            
            if self.du_id_list[n_evt_counter]==1:
                self.det1_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det1_dataset.append(IXPE_data)
            if self.du_id_list[n_evt_counter]==2:
                self.det2_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det2_dataset.append(IXPE_data)
            if self.du_id_list[n_evt_counter]==3:
                self.det3_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det3_dataset.append(IXPE_data)
            n_evt_counter+=1
            
            
            
                
        if do_pcube:
            self.pcube_full=leakagelib_v9.ps_fit.get_pcube(IXPE_full_dataset)     
            self.pcube_det1=leakagelib_v9.ps_fit.get_pcube(IXPE_det1_dataset)   
            self.pcube_det2=leakagelib_v9.ps_fit.get_pcube(IXPE_det2_dataset)   
            self.pcube_det3=leakagelib_v9.ps_fit.get_pcube(IXPE_det3_dataset)   
  
        settings = leakagelib_v9.ps_fit.FitSettings(IXPE_full_dataset)
        #settings.add_point_source() # Point source component
        
        settings.add_background("bkg1", det=(1,)) # Background component from detector 1
        settings.add_background("bkg2", det=(2,)) # Background component from detector 2
        settings.add_background("bkg3", det=(3,)) # Background component from detector 3
        
        settings.set_initial_qu("bkg1", (0., 0.))
        settings.set_initial_qu("bkg2", (0., 0.))
        settings.set_initial_qu("bkg3", (0., 0.))
        
        if fit_prtl_bg:
            print("fit particle background")
            settings.add_particle_source("prtl1", det=(1,))
            settings.add_particle_source("prtl2", det=(2,))
            settings.add_particle_source("prtl3", det=(3,))
            
            settings.set_initial_flux("prtl1", 1.   )
            settings.set_initial_flux("prtl2", 1.   )
            settings.set_initial_flux("prtl3", 1.   )
            
            if fit_prtl_qu:
                settings.set_initial_qu("prtl1", (0., 0.))
                settings.set_initial_qu("prtl2", (0., 0.))
                settings.set_initial_qu("prtl3", (0., 0.))
            else:   
                settings.fix_qu("prtl1", (0., 0.))
                settings.fix_qu("prtl2", (0., 0.))
                settings.fix_qu("prtl3", (0., 0.))

        settings.fix_flux("bkg1", 1.   )
        if fix_det23:
            settings.fix_flux("bkg2", 1.   )
            settings.fix_flux("bkg3", 1.   )
        
        
        
        source=settings.sources[0]
        pixel_centers=source.pixel_centers 
        
        map_coords_2D_X, map_coords_2D_Y=np.meshgrid(pixel_centers, pixel_centers)
        
        mask_roi=np.where((map_coords_2D_X**2+map_coords_2D_Y**2<(dataset.src_r * 2.6)**2 )& (map_coords_2D_X**2+map_coords_2D_Y**2>(src_remove* 2.6)**2 ), 1, 0)
        
        self.pixel_centers=pixel_centers
        self.map_coords_2D_X = map_coords_2D_X
        self.map_coords_2D_y= map_coords_2D_Y
        
        self.mask_roi=mask_roi
        # source object
        # source.source will be square image with right dimensions
        #source.pixelcenters will be coordinates of each pixel, this is 1D array since source image 
        # 
        settings.apply_roi(mask_roi)
        #settings.apply_circular_roi(dataset.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
        fitter = leakagelib_v9.ps_fit.Fitter(IXPE_full_dataset, settings)
        result = fitter.fit()
        print("fit_bkg")
        #fitter.log_prob()
        
        self.model_background=ixpe_background()
        
        self.f_det1_bkg = 1. 
        self.f_det1_bkg_error = 0.
        if fix_det23:
            self.f_det2_bkg = self.det2_N_bkg/self.det1_N_bkg #result.params[('f', 'bkg2')]
            self.f_det2_bkg_error =0# result.sigmas[('f', 'bkg2')]
            self.f_det3_bkg = self.det3_N_bkg/self.det1_N_bkg#result.params[('f', 'bkg3')]
            self.f_det3_bkg_error = 0#result.sigmas[('f', 'bkg3')]
        else:
            self.f_det2_bkg = result.params[('f', 'bkg2')]
            self.f_det2_bkg_error =result.sigmas[('f', 'bkg2')]
            self.f_det3_bkg = result.params[('f', 'bkg3')]
            self.f_det3_bkg_error = result.sigmas[('f', 'bkg3')]
        
        self.q_det1_bkg = result.params[('q', 'bkg1')]
        self.q_det1_bkg_error = result.sigmas[('q', 'bkg1')]
        self.q_det2_bkg = result.params[('q', 'bkg2')]
        self.q_det2_bkg_error = result.sigmas[('q', 'bkg2')]
        self.q_det3_bkg = result.params[('q', 'bkg3')]
        self.q_det3_bkg_error = result.sigmas[('q', 'bkg3')]
        
        self.u_det1_bkg = result.params[('u', 'bkg1')]
        self.u_det1_bkg_error = result.sigmas[('u', 'bkg1')]
        self.u_det2_bkg = result.params[('u', 'bkg2')]
        self.u_det2_bkg_error = result.sigmas[('u', 'bkg2')]
        self.u_det3_bkg = result.params[('u', 'bkg3')]
        self.u_det3_bkg_error = result.sigmas[('u', 'bkg3')]
        
        self.model_background.det_photon_bkg(1, self.f_det1_bkg, self.f_det1_bkg_error, self.q_det1_bkg, self.q_det1_bkg_error, self.u_det1_bkg, self.u_det1_bkg_error)
        
        
        
        self.model_background.det_photon_bkg(2, self.f_det2_bkg, self.f_det2_bkg_error, self.q_det2_bkg, self.q_det2_bkg_error, self.u_det2_bkg, self.u_det2_bkg_error)
        
        
        self.model_background.det_photon_bkg(3, self.f_det3_bkg, self.f_det3_bkg_error, self.q_det3_bkg, self.q_det3_bkg_error, self.u_det3_bkg, self.u_det3_bkg_error)
        
        
        
        if fit_prtl_bg:
            self.f_det1_pbkg = result.params[('f', 'prtl1')]
            self.f_det1_pbkg_error = result.sigmas[('f', 'prtl1')]
            if fix_det23:
                self.f_det2_pbkg = result.params[('f', 'prtl2')]*self.det2_N_bkg/self.det1_N_bkg
                self.f_det2_pbkg_error = result.sigmas[('f', 'prtl2')]*self.det2_N_bkg/self.det1_N_bkg
                self.f_det3_pbkg = result.params[('f', 'prtl3')]*self.det3_N_bkg/self.det1_N_bkg
                self.f_det3_pbkg_error = result.sigmas[('f', 'prtl3')]*self.det3_N_bkg/self.det1_N_bkg
            else:
                self.f_det2_pbkg = result.params[('f', 'prtl2')]#*self.det2_N_bkg/self.det1_N_bkg
                self.f_det2_pbkg_error = result.sigmas[('f', 'prtl2')]#*self.det2_N_bkg/self.det1_N_bkg
                self.f_det3_pbkg = result.params[('f', 'prtl3')]#*self.det3_N_bkg/self.det1_N_bkg
                self.f_det3_pbkg_error = result.sigmas[('f', 'prtl3')]#*self.det3_N_bkg/self.det1_N_bkg
            if fit_prtl_qu:
                
                self.q_det1_pbkg = result.params[('q', 'prtl1')]
                self.q_det1_pbkg_error = result.sigmas[('q', 'prtl1')]
                self.q_det2_pbkg = result.params[('q', 'prtl2')]
                self.q_det2_pbkg_error = result.sigmas[('q', 'prtl2')]
                self.q_det3_pbkg = result.params[('q', 'prtl3')]
                self.q_det3_pbkg_error = result.sigmas[('q', 'prtl3')]
                
                self.u_det1_pbkg = result.params[('u', 'prtl1')]
                self.u_det1_pbkg_error = result.sigmas[('u', 'prtl1')]
                self.u_det2_pbkg = result.params[('u', 'prtl2')]
                self.u_det2_pbkg_error = result.sigmas[('u', 'prtl2')]
                self.u_det3_pbkg = result.params[('u', 'prtl3')]
                self.u_det3_pbkg_error = result.sigmas[('u', 'prtl3')]
                
                self.model_background.det_particle_bkg(1, self.f_det1_pbkg, self.f_det1_pbkg_error, self.q_det1_pbkg, self.q_det1_pbkg_error, self.u_det1_pbkg, self.u_det1_pbkg_error)
                self.model_background.det_particle_bkg(2, self.f_det2_pbkg, self.f_det2_pbkg_error, self.q_det2_pbkg, self.q_det2_pbkg_error, self.u_det2_pbkg, self.u_det2_pbkg_error)
                self.model_background.det_particle_bkg(3, self.f_det3_pbkg, self.f_det3_pbkg_error, self.q_det3_pbkg, self.q_det3_pbkg_error, self.u_det3_pbkg, self.u_det3_pbkg_error)
        
        if save_bkg:
            self.model_background.save_bkg(background_name, background_dir_name)
        
        print("fit_bkg success")
        self.likelihod_bkg_final = result.fun
        
        
        
    def IXPE_pcube_bg(self, list_dataset_unpacked, src_remove, bkg_out_r, nbins, Porb, TASC, Emin=2, Emax =8,  weights=1, use_weights=True, bkg_sub=0, fit_sep_det=True, bg_weights=True, bg_weights_thresh=0.99, bg_cut=True, flare_cut=True, fit_prtl_bg=True, fit_prtl_qu=True, save_bkg=True, background_name="main_", background_dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/', do_pcube=True, fix_det23=True):
        '''
        src_remove is the radius in pixels to remove to exclude the source
        '''
        
        if weights == 1:
            NN=False
        elif weights ==2:
            NN=True
        source = leakagelib_v9.source.Source.no_image(NN)
        N_perbin=0
        IXPE_full_dataset=[]
        IXPE_det1_dataset=[]
        IXPE_det2_dataset=[]
        IXPE_det3_dataset=[]
        if flare_cut:
            flare_cut_counter=0
        n_evt_counter=0
        self.det1_N_bkg=0
        self.det2_N_bkg=0
        self.det3_N_bkg=0
        for dataset in list_dataset_unpacked:
            E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
            src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, bkg_out_r, mask=E_mask)
            src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
            dataset.total_image.phase = Convert_Phase_normalized(dataset.total_image.time, TASC, Porb)
            IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (dataset.filename, dataset.hkname), energy_cut=(Emin, Emax))
            #IXPE_data.retain(src_reg_Elim)
            #IXPE_data.evt_times=src_photons_Elim.phase
            #IXPE_data.cut(E_mask)
            
            
            
            if flare_cut:
                print("Flare mask length {}".format(len(dataset.total_image.phase[E_mask][self.mask_flares[flare_cut_counter]])))
                combined_src_flares_mask=np.where((src_reg_Elim) & (self.mask_flares[flare_cut_counter]), True, False)
                IXPE_data.retain(combined_src_flares_mask)
                flare_cut_counter+=1
                IXPE_data.evt_times=dataset.total_image.phase[E_mask][combined_src_flares_mask]
                
            else:
                IXPE_data.retain(src_reg_Elim)
                IXPE_data.evt_times=src_photons_Elim.phase
            
            IXPE_data.explicit_center(dataset.src_x, dataset.src_y)
            
            self.src_remove_mask = np.where(IXPE_data.evt_xs**2+ IXPE_data.evt_ys**2 < src_remove**2, False, True)
            print("Number of photons {}".format(len(IXPE_data.evt_xs)))
            IXPE_data.retain(self.src_remove_mask)
            print("Number of photons after cutting {}".format(len(IXPE_data.evt_xs)))
            print("removing: {}".format( src_remove))
            #IXPE_data_copy = copy.deepcopy(IXPE_data)
            
            
            
            
            
            
            
            
            if not fit_prtl_bg:
                IXPE_data.evt_bg_chars=np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, IXPE_data.evt_bg_chars, 0)
                #IXPE_data
                #IXPE_data.cut()
            elif bg_cut:
                evts_cut = np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, 0, 1)
                IXPE_data.retain(evts_cut)
            if NN:
                IXPE_data.weight_nn()
            N_perbin += len(IXPE_data.evt_xs)
            if do_pcube:
                IXPE_data.evt_ws = np.ones( len(IXPE_data.evt_ws ))
            IXPE_full_dataset.append(IXPE_data)
            
            print("Max distance from center {}".format(np.max(np.sqrt(IXPE_data.evt_ys**2+IXPE_data.evt_xs**2))))
            
            if self.du_id_list[n_evt_counter]==1:
                self.det1_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det1_dataset.append(IXPE_data)
            if self.du_id_list[n_evt_counter]==2:
                self.det2_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det2_dataset.append(IXPE_data)
            if self.du_id_list[n_evt_counter]==3:
                self.det3_N_bkg+=len(IXPE_data.evt_xs)
                IXPE_det3_dataset.append(IXPE_data)
            n_evt_counter+=1
            
            
            
                
        if do_pcube:
            self.pcube_bkg_full=leakagelib_v9.ps_fit.get_pcube(IXPE_full_dataset)     
            

    def IXPE_lc_fit(self, list_dataset_unpacked, lc_model, q_model, u_model, nbins, Porb, TASC, Emin=2, Emax =8,  weights=1, use_weights=True, bkg_sub=0, rotation_dir_model=1, model_unpolarized=False, model_prior=[0.3, 0.3], fit_const=False, q_model_const=None, u_model_const=None, rotation_dir_const=1, fit_const_qu=False, const_qu_prior=[0, 0], fit_bg_qu=False, fit_bg_det=False, fit_bg_det_sep=True, bg_weights=True, bg_weights_thresh=0.99, bg_cut=True, flare_cut=True, fit_prtl_bg=True, fit_prtl_qu=True, do_fit=True, re_fit=False, provide_background=None, do_pcube=True):
        
        
        
        if weights == 1:
            NN=False
        elif weights ==2:
            NN=True
        source = leakagelib_v9.source.Source.no_image(NN)
        N_perbin=0
        IXPE_full_dataset=[]
        if flare_cut:
            flare_cut_counter=0
        for dataset in list_dataset_unpacked:
            E_mask=np.where((dataset.total_image.E>Emin) & (dataset.total_image.E<Emax), True, False)
            src_photons_Elim, src_reg_Elim = dataset.select_region(dataset.src_x, dataset.src_y, dataset.src_r, mask=E_mask)
            src_photons_Elim.phase = Convert_Phase_normalized(src_photons_Elim.time, TASC, Porb)
            dataset.total_image.phase = Convert_Phase_normalized(dataset.total_image.time, TASC, Porb)
        
            IXPE_data=leakagelib_v9.ixpe_data.IXPEData(source, (dataset.filename, dataset.hkname), energy_cut=(Emin, Emax))
            
            if flare_cut:
                print("Flare mask length {}".format(len(dataset.total_image.phase[E_mask][self.mask_flares[flare_cut_counter]])))
                combined_src_flares_mask=np.where((src_reg_Elim) & (self.mask_flares[flare_cut_counter]), True, False)
                IXPE_data.retain(combined_src_flares_mask)
                flare_cut_counter+=1
                IXPE_data.evt_times=dataset.total_image.phase[E_mask][combined_src_flares_mask]
            else:
                IXPE_data.retain(src_reg_Elim)
                IXPE_data.evt_times=src_photons_Elim.phase
            
            
            
            #IXPE_data.cut(E_mask)
            IXPE_data.explicit_center(dataset.src_x, dataset.src_y)
            #IXPE_data_copy = copy.deepcopy(IXPE_data)
            if not bg_weights:
                IXPE_data.evt_bg_chars=np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, IXPE_data.evt_bg_chars, 0)
                #IXPE_data
                #IXPE_data.cut()
            elif bg_cut:
                evts_cut = np.where(IXPE_data.evt_bg_chars>bg_weights_thresh, 0, 1)
                IXPE_data.retain(evts_cut)
            if NN:
                IXPE_data.weight_nn()
            N_perbin += len(IXPE_data.evt_xs)
            if do_pcube:
                IXPE_data.evt_ws = np.ones( len(IXPE_data.evt_ws ))
            IXPE_full_dataset.append(IXPE_data)
        print("Number of events remaining {}".format(N_perbin))
        self.N_evts=N_perbin
        self.IXPE_full_dataset=IXPE_full_dataset
        self.fit_const=fit_const
        self.fit_const_qu=fit_const_qu
        self.fit_bg_det=fit_bg_det
        self.fit_bg_qu = fit_bg_qu
        self.fit_prtl_bg=fit_prtl_bg
        self.fit_prtl_qu=fit_prtl_qu
        
        if do_pcube:
            self.pcube_src_full=leakagelib_v9.ps_fit.get_pcube(IXPE_full_dataset) 
        
        if do_fit:
            settings = leakagelib_v9.ps_fit.FitSettings(IXPE_full_dataset)
            settings.add_point_source() # Point source component
            if fit_const:
                settings.add_point_source("const")
                if not fit_const_qu:
                    settings.fix_qu("const", (0, 0))
                else:
                    settings.set_initial_qu("const", (const_qu_prior[0], const_qu_prior[1]))
                    if q_model_const is not None and u_model_const is not None:
                        
                        def qu_model_const(phase):
                            return (q_model_const(phase), u_model_const(phase))
                        
                        settings.set_sweep("const", qu_model_const, rotation_dir=rotation_dir_const)
             # Background component
            if fit_bg_det:
                
                
                    settings.add_background("bkg1", det=(1,)) # Background component from detector 1
                    settings.add_background("bkg2", det=(2,)) # Background component from detector 2
                    settings.add_background("bkg3", det=(3,))
                    
                    if fit_bg_qu:
                        if fit_bg_det_sep:
                            if provide_background is None:
                                settings.fix_qu("bkg1", (self.q_det1_bkg, self.u_det1_bkg))
                                settings.fix_qu("bkg2", (self.q_det2_bkg, self.u_det2_bkg))
                                settings.fix_qu("bkg3", (self.q_det3_bkg, self.u_det3_bkg))
                            else:
                                settings.fix_qu("bkg1", (provide_background.q_det1_bkg, provide_background.u_det1_bkg))
                                settings.fix_qu("bkg2", (provide_background.q_det2_bkg, provide_background.u_det2_bkg))
                                settings.fix_qu("bkg3", (provide_background.q_det3_bkg, provide_background.u_det3_bkg))
                                
                        else:
                            settings.set_initial_qu("bkg1", (0., 0.))
                            settings.set_initial_qu("bkg2", (0., 0.))
                            settings.set_initial_qu("bkg3", (0., 0.))
                    else:
                        settings.fix_qu("bkg1", (0, 0))
                        settings.fix_qu("bkg2", (0, 0))
                        settings.fix_qu("bkg3", (0, 0))
                    
                    if fit_prtl_bg:
                        settings.add_particle_source("prtl1", det=(1,))
                        settings.add_particle_source("prtl2", det=(2,))
                        settings.add_particle_source("prtl3", det=(3,))
                        
                        if fit_prtl_qu:
                            if fit_bg_det_sep:
                                if provide_background is None:
                                    settings.fix_qu("bkg1", (self.q_det1_pbkg, self.u_det1_pbkg))
                                    settings.fix_qu("bkg2", (self.q_det2_pbkg, self.u_det2_pbkg))
                                    settings.fix_qu("bkg3", (self.q_det3_pbkg, self.u_det3_pbkg))
                                else:
                                    settings.fix_qu("prtl1", (provide_background.q_det1_pbkg, provide_background.u_det1_pbkg))
                                    settings.fix_qu("prtl2", (provide_background.q_det2_pbkg, provide_background.u_det2_pbkg))
                                    settings.fix_qu("prtl3", (provide_background.q_det3_pbkg, provide_background.u_det3_pbkg))
                            else:
                                settings.set_initial_qu("prtl1", (0., 0.))
                                settings.set_initial_qu("prtl2", (0., 0.))
                                settings.set_initial_qu("prtl3", (0., 0.))
                        else:   
                            settings.fix_qu("prtl1", (0., 0.))
                            settings.fix_qu("prtl2", (0., 0.))
                            settings.fix_qu("prtl3", (0., 0.))
                    
                    settings.fix_flux("bkg1", 1)
                    if fit_bg_det_sep:
                        if provide_background is None:
                            settings.fix_flux("bkg1", self.f_det1_bkg)
                            settings.fix_flux("bkg2", self.f_det2_bkg)
                            settings.fix_flux("bkg3", self.f_det3_bkg)
                            if fit_prtl_bg:
                                settings.fix_flux("prtl1", self.f_det1_pbkg)
                                settings.fix_flux("prtl2", self.f_det2_pbkg)
                                settings.fix_flux("prtl3", self.f_det3_pbkg)
                        else:
                            settings.fix_flux("bkg1", provide_background.f_det1_bkg)
                            settings.fix_flux("bkg2", provide_background.f_det2_bkg)
                            settings.fix_flux("bkg3", provide_background.f_det3_bkg)
                            if fit_prtl_bg:
                                settings.fix_flux("prtl1", provide_background.f_det1_pbkg)
                                settings.fix_flux("prtl2", provide_background.f_det2_pbkg)
                                settings.fix_flux("prtl3", provide_background.f_det3_pbkg)
                                
            else:
                settings.add_background()
                settings.fix_flux("bkg", 1)
                
            
               
                if not fit_bg_qu:
                    settings.fix_qu("bkg", (0.0, 0.0)) # Set the background to be unpolarized
                
            
            settings.set_initial_flux("src", 0.02)
            if model_unpolarized:
                settings.fix_qu("src", (0.0, 0.0))
            else:
                settings.set_initial_qu("src", (model_prior[0], model_prior[1]))
                def qu_model(phase):
                    return (q_model(phase), u_model(phase))
                settings.set_sweep("src", qu_model, rotation_dir=rotation_dir_model)
            settings.set_lightcurve("src", lc_model)
            self.src_r_use=dataset.src_r
            self.lc_model=lc_model
            settings.apply_circular_roi(dataset.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
            fitter = leakagelib_v9.ps_fit.Fitter(IXPE_full_dataset, settings)
            result = fitter.fit()
            self.result=result
            if fit_bg_qu:
                if fit_bg_det:
                    if not fit_bg_det_sep:
                        self.f_det1_bkg = 1. 
                        self.f_det1_bkg_error = 0.
                        self.f_det2_bkg = result.params[('f', 'bkg2')]
                        self.f_det2_bkg_error = result.sigmas[('f', 'bkg2')]
                        self.f_det3_bkg = result.params[('f', 'bkg3')]
                        self.f_det3_bkg_error = result.sigmas[('f', 'bkg3')]
                        
                        self.q_det1_bkg = result.params[('q', 'bkg1')]
                        self.q_det1_bkg_error = result.sigmas[('q', 'bkg1')]
                        self.q_det2_bkg = result.params[('q', 'bkg2')]
                        self.q_det2_bkg_error = result.sigmas[('q', 'bkg2')]
                        self.q_det3_bkg = result.params[('q', 'bkg3')]
                        self.q_det3_bkg_error = result.sigmas[('q', 'bkg3')]
                        
                        self.u_det1_bkg = result.params[('u', 'bkg1')]
                        self.u_det1_bkg_error = result.sigmas[('u', 'bkg1')]
                        self.u_det2_bkg = result.params[('u', 'bkg2')]
                        self.u_det2_bkg_error = result.sigmas[('u', 'bkg2')]
                        self.u_det3_bkg = result.params[('u', 'bkg3')]
                        self.u_det3_bkg_error = result.sigmas[('u', 'bkg3')]
                        if fit_prtl_bg:
                            self.f_det1_pbkg = result.params[('f', 'prtl1')]
                            self.f_det1_pbkg_error = result.sigmas[('f', 'prtl1')]
                            self.f_det2_pbkg = result.params[('f', 'prtl2')]
                            self.f_det2_pbkg_error = result.sigmas[('f', 'prtl2')]
                            self.f_det3_pbkg = result.params[('f', 'prtl3')]
                            self.f_det3_pbkg_error = result.sigmas[('f', 'prtl3')]
                            if fit_prtl_qu:
                                self.q_det1_pbkg = result.params[('q', 'prtl1')]
                                self.q_det1_pbkg_error = result.sigmas[('q', 'prtl1')]
                                self.q_det2_pbkg = result.params[('q', 'prtl2')]
                                self.q_det2_pbkg_error = result.sigmas[('q', 'prtl2')]
                                self.q_det3_pbkg = result.params[('q', 'prtl3')]
                                self.q_det3_pbkg_error = result.sigmas[('q', 'prtl3')]
                                
                                self.u_det1_pbkg = result.params[('u', 'prtl1')]
                                self.u_det1_pbkg_error = result.sigmas[('u', 'prtl1')]
                                self.u_det2_pbkg = result.params[('u', 'prtl2')]
                                self.u_det2_pbkg_error = result.sigmas[('u', 'prtl2')]
                                self.u_det3_pbkg = result.params[('u', 'prtl3')]
                                self.u_det3_pbkg_error = result.sigmas[('u', 'prtl3')]
                        
                        
            #fitter.log_prob()
            self.f_fit = result.params[('f', 'src')]
            self.f_error_fit = result.sigmas[('f', 'src')]
            if not model_unpolarized:
                self.q_fit = result.params[('q', 'src')]
                self.q_error_fit = result.sigmas[('q', 'src')]
                self.u_fit = result.params[('u', 'src')]
                self.u_error_fit = result.sigmas[('u', 'src')]
            self.likelihod_final = result.fun
            if fit_const:
                self.f_const_fit = result.params[('f', 'const')]
                self.f_const_error_fit = result.sigmas[('f', 'const')]
            if fit_const_qu:
                self.q_const_fit = result.params[('q', 'const')]
                self.q_const_error_fit = result.sigmas[('q', 'const')] 
           
                self.u_const_fit = result.params[('u', 'const')]
                self.u_const_error_fit = result.sigmas[('u', 'const')]
                
                
            self.model_unpolarized=model_unpolarized        
            if not model_unpolarized:
                null_params = []
                model_params = []
                for name in result.parameter_names:
                    model_params.append(result.params[name])
                    if name == ("q", "src") or name == ("u", "src"):
                        null_params.append(0)
                    else:
                        null_params.append(result.params[name])
                model_like = self.likelihod_final #fitter.log_prob(model_params)
                null_like = fitter.log_prob(null_params)
                self.null_like=null_like
                self.model_likes_all_pts = fitter.log_prob(model_params, return_array=True)
                self.null_likes = fitter.log_prob(model_params, return_array=True)
                like_std = np.std(self.model_likes_all_pts - self.null_likes)
                root_counts = np.sqrt(len(self.model_likes_all_pts))
                self.sig_vuong = (model_like - null_like) / (root_counts * like_std)
            if re_fit:
                settings = leakagelib_v9.ps_fit.FitSettings(IXPE_full_dataset)
                settings.add_point_source() # Point source component
                if fit_const:
                    settings.add_point_source("const")
                    if not fit_const_qu:
                        settings.fix_qu("const", (0, 0))
                    else:
                        settings.set_initial_qu("const", (const_qu_prior[0], const_qu_prior[1]))
                        if q_model_const is not None and u_model_const is not None:
                            
                            def qu_model_const(phase):
                                return (q_model_const(phase), u_model_const(phase))
                            
                            settings.set_sweep("const", qu_model_const, rotation_dir=rotation_dir_const)
                 # Background component
                if fit_bg_det:
                    
                    
                        settings.add_background("bkg1", det=(1,)) # Background component from detector 1
                        settings.add_background("bkg2", det=(2,)) # Background component from detector 2
                        settings.add_background("bkg3", det=(3,))
                        if fit_bg_qu:
                            settings.fix_qu("bkg1", (self.q_det1_bkg, self.u_det1_bkg))
                            settings.fix_qu("bkg2", (self.q_det2_bkg, self.u_det2_bkg))
                            settings.fix_qu("bkg3", (self.q_det3_bkg, self.u_det3_bkg))
                                
                            
                        
                        if fit_prtl_bg:
                            settings.add_particle_source("prtl1", det=(1,))
                            settings.add_particle_source("prtl2", det=(2,))
                            settings.add_particle_source("prtl3", det=(3,))
                            settings.fix_flux("prtl1", self.f_det1_pbkg)
                            settings.fix_flux("prtl2", self.f_det2_pbkg)
                            settings.fix_flux("prtl3", self.f_det3_pbkg)
                            
                            if fit_prtl_qu:
                                settings.fix_qu("prtl1", (self.q_det1_pbkg , self.u_det1_pbkg ))
                                settings.fix_qu("prtl2", (self.q_det1_pbkg , self.u_det1_pbkg ))
                                settings.fix_qu("prtl3", (self.q_det1_pbkg , self.u_det1_pbkg ))
                            else:   
                                settings.fix_qu("prtl1", (0., 0.))
                                settings.fix_qu("prtl2", (0., 0.))
                                settings.fix_qu("prtl3", (0., 0.))
                        
                        settings.fix_flux("bkg1", 1)
                        settings.fix_flux("bkg2", self.f_det2_bkg)
                        settings.fix_flux("bkg3", self.f_det3_bkg)
                else:
                    settings.add_background()
                    settings.fix_flux("bkg", 1)
                    
                
                   
                    if not fit_bg_qu:
                        settings.fix_qu("bkg", (0.0, 0.0)) # Set the background to be unpolarized
                    
                
                settings.set_initial_flux("src", self.f_fit)
                settings.set_initial_flux("const", self.f_const_fit)
               
                #settings
                
                if model_unpolarized:
                    settings.fix_qu("src", (0.0, 0.0))
                else:
                    settings.set_initial_qu("src", (model_prior[0], model_prior[1]))
                    def qu_model(phase):
                        return (q_model(phase), u_model(phase))
                    settings.set_sweep("src", qu_model, rotation_dir=rotation_dir_model)
                    settings.set_initial_qu("src", (self.q_fit, self.u_fit))
                settings.set_lightcurve("src", lc_model)
                settings.apply_circular_roi(dataset.src_r * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
                fitter = leakagelib_v9.ps_fit.Fitter(IXPE_full_dataset, settings)
                result = fitter.fit()
                self.result = result
                self.f_fit = result.params[('f', 'src')]
                self.f_error_fit = result.sigmas[('f', 'src')]
                if not model_unpolarized:
                    self.q_fit = result.params[('q', 'src')]
                    self.q_error_fit = result.sigmas[('q', 'src')]
                    self.u_fit = result.params[('u', 'src')]
                    self.u_error_fit = result.sigmas[('u', 'src')]
                self.likelihod_final = result.fun
                if fit_const:
                    self.f_const_fit = result.params[('f', 'const')]
                    self.f_const_error_fit = result.sigmas[('f', 'const')]
                if fit_const_qu:
                    self.q_const_fit = result.params[('q', 'const')]
                    self.q_const_error_fit = result.sigmas[('q', 'const')] 
               
                    self.u_const_fit = result.params[('u', 'const')]
                    self.u_const_error_fit = result.sigmas[('u', 'const')]
                    
                    
                        
                if not model_unpolarized:
                    null_params = []
                    model_params = []
                    for name in result.parameter_names:
                        model_params.append(result.params[name])
                        if name == ("q", "src") or name == ("u", "src"):
                            null_params.append(0)
                        else:
                            null_params.append(result.params[name])
                    model_like = self.likelihod_final #fitter.log_prob(model_params)
                    null_like = fitter.log_prob(null_params)
                    self.null_like=null_like
                    self.model_likes_all_pts = fitter.log_prob(model_params, return_array=True)
                    self.null_likes = fitter.log_prob(model_params, return_array=True)
                    like_std = np.std(self.model_likes_all_pts - self.null_likes)
                    root_counts = np.sqrt(len(self.model_likes_all_pts))
                    self.sig_vuong = (model_like - null_like) / (root_counts * like_std)
                    

    def re_fit_newmodel(self, q_model=None, u_model=None, q_model_const=None, u_model_const=None, model_unpolarized=False, fit_const_qu=True, const_qu_prior=[0, 0], model_prior=[0.4, -0.05], rotation_dir_const=1, rotation_dir_model=1):
        settings = leakagelib_v9.ps_fit.FitSettings(self.IXPE_full_dataset)
        settings.add_point_source() # Point source component
        if self.fit_const:
            settings.add_point_source("const")
            if not fit_const_qu:
                settings.fix_qu("const", (0, 0))
            else:
                settings.set_initial_qu("const", (const_qu_prior[0], const_qu_prior[1]))
                if q_model_const is not None and u_model_const is not None:
                    
                    def qu_model_const(phase):
                        return (q_model_const(phase), u_model_const(phase))
                    
                    settings.set_sweep("const", qu_model_const, rotation_dir=rotation_dir_const)
         # Background component
        if self.fit_bg_det:
            
            
                settings.add_background("bkg1", det=(1,)) # Background component from detector 1
                settings.add_background("bkg2", det=(2,)) # Background component from detector 2
                settings.add_background("bkg3", det=(3,))
                if self.fit_bg_qu:
                    settings.fix_qu("bkg1", (self.q_det1_bkg, self.u_det1_bkg))
                    settings.fix_qu("bkg2", (self.q_det2_bkg, self.u_det2_bkg))
                    settings.fix_qu("bkg3", (self.q_det3_bkg, self.u_det3_bkg))
                        
                    
                
                if self.fit_prtl_bg:
                    settings.add_particle_source("prtl1", det=(1,))
                    settings.add_particle_source("prtl2", det=(2,))
                    settings.add_particle_source("prtl3", det=(3,))
                    settings.fix_flux("prtl1", self.f_det1_pbkg)
                    settings.fix_flux("prtl2", self.f_det2_pbkg)
                    settings.fix_flux("prtl3", self.f_det3_pbkg)
                    
                    if self.fit_prtl_qu:
                        settings.fix_qu("prtl1", (self.q_det1_pbkg , self.u_det1_pbkg ))
                        settings.fix_qu("prtl2", (self.q_det1_pbkg , self.u_det1_pbkg ))
                        settings.fix_qu("prtl3", (self.q_det1_pbkg , self.u_det1_pbkg ))
                    else:   
                        settings.fix_qu("prtl1", (0., 0.))
                        settings.fix_qu("prtl2", (0., 0.))
                        settings.fix_qu("prtl3", (0., 0.))
                
                settings.fix_flux("bkg1", 1)
                settings.fix_flux("bkg2", self.f_det2_bkg)
                settings.fix_flux("bkg3", self.f_det3_bkg)
        else:
            settings.add_background()
            settings.fix_flux("bkg", 1)
            
        
           
            if not self.fit_bg_qu:
                settings.fix_qu("bkg", (0.0, 0.0)) # Set the background to be unpolarized
            
        
        settings.set_initial_flux("src", self.f_fit)
        settings.set_initial_flux("const", self.f_const_fit)
       
        #settings
        
        if model_unpolarized:
            settings.fix_qu("src", (0.0, 0.0))
        else:
            #settings.set_initial_qu("src", (model_prior[0], model_prior[1]))
            def qu_model(phase):
                return (q_model(phase), u_model(phase))
            settings.set_sweep("src", qu_model, rotation_dir=rotation_dir_model)
            if not self.model_unpolarized:
                settings.set_initial_qu("src", (self.q_fit, self.u_fit))
            else:
                settings.set_initial_qu("src", (model_prior[0], model_prior[1]))
        settings.set_lightcurve("src", self.lc_model)
        settings.apply_circular_roi(self.src_r_use * 2.6) # Tell the fitter how big the region is, so that it can normalize the background PDF. This number must be the ROI size in arcsec
        fitter = leakagelib_v9.ps_fit.Fitter(self.IXPE_full_dataset, settings)
        result = fitter.fit()
        self.result_refit = result
        self.f_fit_refit = result.params[('f', 'src')]
        self.f_error_fit_refit = result.sigmas[('f', 'src')]
        if not model_unpolarized:
            self.q_fit_refit = result.params[('q', 'src')]
            self.q_error_fit_refit = result.sigmas[('q', 'src')]
            self.u_fit_refit = result.params[('u', 'src')]
            self.u_error_fit_refit = result.sigmas[('u', 'src')]
        self.likelihod_final_refit = result.fun
        if self.fit_const:
            self.f_const_fit_refit = result.params[('f', 'const')]
            self.f_const_error_fit_refit = result.sigmas[('f', 'const')]
        if fit_const_qu:
            self.q_const_fit_refit = result.params[('q', 'const')]
            self.q_const_error_fit_refit = result.sigmas[('q', 'const')] 
       
            self.u_const_fit_refit = result.params[('u', 'const')]
            self.u_const_error_fit_refit = result.sigmas[('u', 'const')]
            
            
                
        if not model_unpolarized:
            null_params = []
            model_params = []
            for name in result.parameter_names:
                model_params.append(result.params[name])
                if name == ("q", "src") or name == ("u", "src"):
                    null_params.append(0)
                else:
                    null_params.append(result.params[name])
            model_like = self.likelihod_final #fitter.log_prob(model_params)
            null_like = fitter.log_prob(null_params)
            self.null_like_refit=null_like
            self.model_likes_all_pts_refit = fitter.log_prob(model_params, return_array=True)
            self.null_likes_refit = fitter.log_prob(model_params, return_array=True)
            like_std_refit = np.std(self.model_likes_all_pts_refit - self.null_likes_refit)
            root_counts = np.sqrt(len(self.model_likes_all_pts_refit))
            self.sig_vuong_refit = (model_like - null_like) / (root_counts * like_std_refit)



def cut_bkg():
    reject = np.where(abs((self.flux_LC_bkg-np.mean(self.flux_LC_bkg)))/np.std(self.flux_LC_bkg)> threshold)
    self.flux_LC_src_clean=   self.flux_LC_src
    self.error_LC_src_clean=   self.error_LC_src
    self.flux_LC_src_clean[reject] = 0
    self.error_LC_src_clean[reject] = 0
    self.flux_LC_bkg[reject] = 0
    #print(self.flux_LC_src)
    self.exposure_clean = np.where(self.flux_LC_bkg>0, 1, 0)
    excluded_parts=find_zero_blocks(self.exposure_clean)
    time_excluded = self.t_LC[excluded_parts]
    #print(time_excluded)
    photon_times_keep, mask = remove_between_values(self.total_image.time, time_excluded)


def plot_polarization(list_pipeline, list_labels, list_colors, shift=True, norm_flux = True, plot2orb=False, pd_ylim=[0, 0.4], desiredcapsize=3, save_fig=False, savename='polarization_plot', dpi=300, desiredfontsize=16, plot_bkg=False, plot_MDP=True, combine_all=False, no_show=False, combine_color='black', combine_color2='darkmagenta', list_pipeline2=None, list_labels2=None, list_colors2=None, combine_title='Total', combine_title2='Total2', model1 = None, model1color='blue', model_offset=None, model2= None, model2color='orange', i_ibs_model=None, i_const_model=None, q_ibs_model1=0., u_ibs_model1=0., q_const_model1=0, u_const_model1=0, q_ibs_model2=0., u_ibs_model2=0., q_const_model2=0, u_const_model2=0, rotation_dir_model1=1, rotation_dir_const1=1, rotation_dir_model2=1, rotation_dir_const2=1):
    if plot2orb:
        ticknum = 9
        if shift:
            xlim = [0., 2.]
        else:
            xlim = [-0.25, 1.75]
        
    else:
        ticknum = 5
        if shift:
            xlim = [0., 1.]
        else:
            xlim = [-0.25, 0.75]
    xticks = np.linspace(xlim[0], xlim[1], ticknum)
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    #if inclnum == 2:
    if not no_show:
        if plot_bkg:
            for n, I in enumerate(list_pipeline.I_bkg):
                #print(I_src)
                #print(list_pipeline.phases[n])
                ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                ax[1].errorbar(list_pipeline.phases[n], list_pipeline.PD_bkg[n], yerr=list_pipeline.PDerr_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_bkg[n], yerr=list_pipeline.EVPAerr_bkg[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                if plot_MDP:
                    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP_bkg[n],color=list_colors[n],  linestyle='None', marker='x',)
        else:
            for n, I in enumerate(list_pipeline.I):
                #print(I_src)
                #print(list_pipeline.phases[n])
                ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                ax[1].errorbar(list_pipeline.phases[n], list_pipeline.PD_src[n], yerr=list_pipeline.PDerr_src[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_src[n], yerr=list_pipeline.EVPAerr_src[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
                if plot_MDP:
                    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP[n],color=list_colors[n],  linestyle='None', marker='x',)
    else:
        n=0

    #ax[0].plot(pol60.phase, pol60.I_LC[inclnum]/np.max(pol60.I_LC[inclnum]), color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')
    #ax[1].plot(pol60.phase, pol60.PD[inclnum], color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')
    #ax[2].plot(pol60.phase, pol60.PA[inclnum], color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')

    #ax[0].plot(pol_no_cooling.phase, pol_no_cooling.I_LC[inclnum]/np.max(pol_no_cooling.I_LC[inclnum]), color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[1].plot(pol_no_cooling.phase, pol_no_cooling.PD[inclnum], color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[2].plot(pol_no_cooling.phase, pol_no_cooling.PA[inclnum], color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[1].plot(LC_60[0], LC_60[1]/LC_norm60, color='green', label = r'$\gamma_{rad}=60$')
    #ax[2].plot(LC_nocool[0], LC_nocool[1]/LC_norm_nocool, color='blue', label = r'No Cooling')

    if combine_all:
        if norm_flux:
            norm=np.max(list_pipeline.I_combined_total)
        else:
            norm=1.
            
        ax[0].errorbar(list_pipeline.phases[n], list_pipeline.I_combined_total/norm, yerr=list_pipeline.Ierr_combined_total/norm, color=combine_color, label = combine_title, linestyle='None', marker='o', capsize=desiredcapsize)
        ax[1].errorbar(list_pipeline.phases[n], list_pipeline.PD_src_total, yerr=list_pipeline.PDerr_src_total, color=combine_color, label = 'Total', linestyle='None', marker='o', capsize=desiredcapsize)
        ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_combined_total, yerr=list_pipeline.EVPAerr_combined_total ,  color=combine_color,  label = 'Total', linestyle='None', marker='o', capsize=desiredcapsize)
        
    
    
    if model1 is not None:
        if norm_flux:
            norm = 1.
        else:
            norm= np.max(list_pipeline.I_combined_total)
        #if model_offset is not None:
            # need to work on this
        if i_ibs_model is None:
            ax[0].plot(model1[:, 0], model1[:, 1]*norm, color=model1color)
            ax[1].plot(model1[:, 0], model1[:, 2], color=model1color)
        else:
            ax[0].plot(model1[:, 0], (model1[:, 1]*i_ibs_model+i_const_model)*norm/(i_ibs_model+i_const_model), color=model1color)
            #ax[1].plot(model1[:, 0], (model1[:, 2]**2*i_ibs_model**2*(q_ibs_model1**2+u_ibs_model1**2)+(i_const_model**2*(q_const_model1**2+u_const_model1**2)))/(i_ibs_model+i_const_model), color=model1color)
        threshold = 90.0  # set based on what "too big" means in your data
        
        if i_ibs_model is not None:
            q_ibs= q_calculate(model1[:, 2], model1[:, 3])*q_ibs_model1 - u_calculate(model1[:, 2], model1[:, 3])*u_ibs_model1
            u_ibs= rotation_dir_model1*u_calculate(model1[:, 2], model1[:, 3])*q_ibs_model1+ rotation_dir_model1* q_calculate(model1[:, 2], model1[:, 3])*u_ibs_model1 
            
            q_tot = q_ibs*i_ibs_model + q_const_model1*i_const_model
            u_tot = u_ibs*i_ibs_model + u_const_model1*i_const_model
            evpa_model1= evpa_calc(q_tot, u_tot)
            ax[1].plot(model1[:, 0], np.sqrt((q_tot**2+u_tot**2))/(i_ibs_model+i_const_model), color=model1color)
        else:
            evpa_model1=model1[:, 3]
        phase_PA_model1, PA_model1 = plot_with_asymptotic_behavior(model1[:, 0], evpa_model1, threshold, ylim=[-90, 90])
        
        ax[2].plot(phase_PA_model1, PA_model1, color=model1color)
        
    if model2 is not None:
        #ax[0].plot(model2[:, 0], model2[:, 1], color=model2color)
       
        
        if i_ibs_model is not None:
            q_ibs= q_calculate(model2[:, 2], model2[:, 3])*q_ibs_model2 - u_calculate(model2[:, 2], model2[:, 3])*u_ibs_model2
            u_ibs= rotation_dir_model2*u_calculate(model2[:, 2], model2[:, 3])*q_ibs_model2+rotation_dir_model2* q_calculate(model2[:, 2], model2[:, 3])*u_ibs_model2
            
            q_tot = q_ibs*i_ibs_model + q_const_model2*i_const_model
            u_tot = u_ibs*i_ibs_model + u_const_model2*i_const_model
            evpa_model2= evpa_calc(q_tot, u_tot)
            ax[1].plot(model2[:, 0], np.sqrt((q_tot**2+u_tot**2))/(i_ibs_model+i_const_model), color=model2color)
        else:
            ax[1].plot(model2[:, 0], model2[:, 2], color=model2color)
            evpa_model2=model2[:, 3]
        
        phase_PA_model2, PA_model2 = plot_with_asymptotic_behavior(model2[:, 0], evpa_model2, threshold, ylim=[-90, 90])
        
        ax[2].plot(phase_PA_model2, PA_model2, color=model2color)
    
    if list_pipeline2 is not None:
        if not no_show:
            if plot_bkg:
                for n, I in enumerate(list_pipeline2.I_bkg):
                    #print(I_src)
                    #print(list_pipeline.phases[n])
                    ax[0].errorbar(list_pipeline2.phases[n], I, yerr=list_pipeline2.Ie_bkg[n], color=list_colors2[n], label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    ax[1].errorbar(list_pipeline2.phases[n], list_pipeline2.PD_bkg[n], yerr=list_pipeline2.PDerr_bkg[n], color=list_colors2[n], label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    ax[2].errorbar(list_pipeline2.phases[n], list_pipeline2.EVPA_bkg[n], yerr=list_pipeline2.EVPAerr_bkg[n],  color=list_colors2[n],  label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    if plot_MDP:
                        ax[1].plot(list_pipeline2.phases[n]+0.1/len(list_pipeline2.phases[n]), list_pipeline2.MDP_bkg[n],color=list_colors2[n],  linestyle='None', marker='x',)
            else:
                for n, I in enumerate(list_pipeline2.I):
                    #print(I_src)
                    #print(list_pipeline.phases[n])
                    ax[0].errorbar(list_pipeline2.phases[n], I, yerr=list_pipeline2.Ie[n], color=list_colors2[n], label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    ax[1].errorbar(list_pipeline2.phases[n], list_pipeline2.PD_src[n], yerr=list_pipeline2.PDerr_src[n], color=list_colors2[n], label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    ax[2].errorbar(list_pipeline2.phases[n], list_pipeline2.EVPA_src[n], yerr=list_pipeline2.EVPAerr_src[n],  color=list_colors2[n],  label = list_labels2[n], linestyle='None', marker='o', capsize=desiredcapsize)
                    if plot_MDP:
                        ax[1].plot(list_pipeline2.phases[n]+0.1/len(list_pipeline2.phases[n]), list_pipeline2.MDP[n],color=list_colors[n],  linestyle='None', marker='x',)
                    
        if combine_all:
            norm_all=np.max(list_pipeline.I_combined_total)/np.max(list_pipeline2.I_combined_total)
            ax[0].errorbar(list_pipeline2.phases[n], list_pipeline2.I_combined_total*norm_all, yerr=list_pipeline2.Ierr_combined_total*norm_all, color=combine_color2, label = combine_title2, linestyle='None', marker='o', capsize=desiredcapsize)
            ax[1].errorbar(list_pipeline2.phases[n], list_pipeline2.PD_src_total, yerr=list_pipeline2.PDerr_src_total, color=combine_color2, label = 'Total 2', linestyle='None', marker='o', capsize=desiredcapsize)
            ax[2].errorbar(list_pipeline2.phases[n], list_pipeline2.EVPA_combined_total, yerr=list_pipeline2.EVPAerr_combined_total ,  color=combine_color2,  label = 'Total 2', linestyle='None', marker='o', capsize=desiredcapsize)
    
    ax[0].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[0].set_ylabel('Flux', fontsize=desiredfontsize)
    ax[0].legend(loc ='upper left', fontsize=desiredfontsize)
    ax[0].set_xlim(xlim)
    ax[0].set_xticks(xticks)

    ax[1].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[1].set_ylabel('Polarization Degree', fontsize=desiredfontsize)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(xticks)
    ax[1].set_ylim(pd_ylim)
    ax[2].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[2].set_ylabel('Polarization Angle (deg)', fontsize=desiredfontsize)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    ax[2].set_xlim(xlim)
    ax[2].set_ylim([-90, 90])
    ax[2].set_xticks(xticks)
    if save_fig:
        fig.savefig(savename+'.pdf', dpi=dpi)
        fig.savefig(savename+'.png', dpi=dpi)
    plt.show()


def q_calculate(pd, evpa):
    return pd*np.cos(2.*np.pi*evpa/180.)

def u_calculate(pd, evpa):
    return pd*np.sin(2.*np.pi*evpa/180.)
def evpa_calc(q, u):
    return 180./2.*np.arctan2(u, q)/np.pi




def plot_spectrum(list_pipeline, list_labels, list_colors, linestyle_input = None, xlim=[2, 8], ylim=[1, 1000.],  save=False, savename='ixpespec_plot', dpi=300, desiredfontsize=16):    
    if linestyle_input is None:
        linestyle='solid'
        linestyle_list  = [linestyle] * len(list_pipeline.ct_spec_src)
    else:
        linestyle_list = linestyle_input



    fig, ax = plt.subplots(1, figsize=(8, 6))
    for n, ct_spec in enumerate(list_pipeline.ct_spec_src):

        ax.loglog(list_pipeline.E_spec_src[n], ct_spec, color = list_colors[n], label = list_labels[n], linestyle=linestyle_list[n])
 
    #syncticks = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), np.log10(xlim[1])-np.log10(xlim[0])+1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
#ax.plot(10**-5)
    ax.set_xlabel(r'E (keV)', fontsize=desiredfontsize)
    ax.set_ylabel(r'cts keV$^{-1}$', fontsize=desiredfontsize)
    ax.legend(loc ='upper right', fontsize=desiredfontsize)
    #ax.set_xticks(syncticks)
    if save:
        fig.savefig(savename+'.pdf', dpi=dpi)
        fig.savefig(savename+'.png', dpi=dpi) 
    plt.show()
    
    
def plot_QU(list_pipeline, list_labels, list_colors, shift=True, plot2orb=False, desiredcapsize=3, save_fig=False, savename='Q_U_plot', dpi=300, desiredfontsize=16, plot_bkg=False, plot_MDP=True):    
    if plot2orb:
        ticknum = 9
        if shift:
            xlim = [0., 2.]
        else:
            xlim = [-0.25, 1.75]
        
    else:
        ticknum = 5
        if shift:
            xlim = [0., 1.]
        else:
            xlim = [-0.25, 0.75]
    xticks = np.linspace(xlim[0], xlim[1], ticknum)
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    #if inclnum == 2:

    if plot_bkg:
        for n, I in enumerate(list_pipeline.I_bkg):
            #print(I_src)
            #print(list_pipeline.phases[n])
            ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[1].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[0].errorbar(list_pipeline.phases[n], list_pipeline.Q_bkg[n], yerr=list_pipeline.Qe_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            ax[1].errorbar(list_pipeline.phases[n], list_pipeline.U_bkg[n], yerr=list_pipeline.Ue_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_bkg[n], yerr=list_pipeline.EVPAerr_bkg[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #if plot_MDP:
            #    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP_bkg[n],color=list_colors[n],  linestyle='None', marker='x',)
    else:
        for n, I in enumerate(list_pipeline.I):
            #print(I_src)
            #print(list_pipeline.phases[n])
            ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[1].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[0].errorbar(list_pipeline.phases[n], list_pipeline.Q_src[n], yerr=list_pipeline.Qe_src[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            ax[1].errorbar(list_pipeline.phases[n], list_pipeline.U_src[n], yerr=list_pipeline.Ue_src[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_src[n], yerr=list_pipeline.EVPAerr_src[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #if plot_MDP:
            #    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP[n],color=list_colors[n],  linestyle='None', marker='x',)

    ax[0].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[0].set_ylabel('Q', fontsize=desiredfontsize)
    ax[0].legend(loc ='upper left', fontsize=desiredfontsize)
    ax[0].set_xlim(xlim)
    ax[0].set_xticks(xticks)

    ax[1].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[1].set_ylabel('U', fontsize=desiredfontsize)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(xticks)
    #ax[1].set_ylim(pd_ylim)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    #ax[2].set_xlim(xlim)
    #ax[2].set_ylim([-90, 90])
    #ax[2].set_xticks(xticks)
    if save_fig:
        fig.savefig(savename+'.pdf', dpi=dpi)
        fig.savefig(savename+'.png', dpi=dpi)
    plt.show()

def plot_qu(list_pipeline, list_labels, list_colors, shift=True, plot2orb=False, desiredcapsize=3, save_fig=False, savename='q_u_plot', dpi=300, desiredfontsize=16, plot_bkg=False, plot_MDP=True, ylim=[-1., 1.]):    
    if plot2orb:
        ticknum = 9
        if shift:
            xlim = [0., 2.]
        else:
            xlim = [-0.25, 1.75]
        
    else:
        ticknum = 5
        if shift:
            xlim = [0., 1.]
        else:
            xlim = [-0.25, 0.75]
    xticks = np.linspace(xlim[0], xlim[1], ticknum)
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    #if inclnum == 2:

    if plot_bkg:
        for n, I in enumerate(list_pipeline.I_bkg):
            #print(I_src)
            #print(list_pipeline.phases[n])
            #ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            #ax[1].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie_bkg[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[0].errorbar(list_pipeline.phases[n], list_pipeline.Q_bkg[n]/I, yerr=list_pipeline.Qe_bkg[n]/I, color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            ax[1].errorbar(list_pipeline.phases[n], list_pipeline.U_bkg[n]/I, yerr=list_pipeline.Ue_bkg[n]/I, color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_bkg[n], yerr=list_pipeline.EVPAerr_bkg[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #if plot_MDP:
            #    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP_bkg[n],color=list_colors[n],  linestyle='None', marker='x',)
    else:
        for n, I in enumerate(list_pipeline.I):
            #print(I_src)
            #print(list_pipeline.phases[n])
            #ax[0].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            #ax[1].errorbar(list_pipeline.phases[n], I, yerr=list_pipeline.Ie[n], color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize, alpha=0.5)
            ax[0].errorbar(list_pipeline.phases[n], list_pipeline.Q_src[n]/I, yerr=list_pipeline.Qe_src[n]/I, color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            ax[1].errorbar(list_pipeline.phases[n], list_pipeline.U_src[n]/I, yerr=list_pipeline.Ue_src[n]/I, color=list_colors[n], label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_src[n], yerr=list_pipeline.EVPAerr_src[n],  color=list_colors[n],  label = list_labels[n], linestyle='None', marker='o', capsize=desiredcapsize)
            #if plot_MDP:
            #    ax[1].plot(list_pipeline.phases[n]+0.1/len(list_pipeline.phases[n]), list_pipeline.MDP[n],color=list_colors[n],  linestyle='None', marker='x',)

    ax[0].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[0].set_ylabel('q', fontsize=desiredfontsize)
    ax[0].legend(loc ='upper left', fontsize=desiredfontsize)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xticks(xticks)

    ax[1].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[1].set_ylabel('u', fontsize=desiredfontsize)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(xticks)
    ax[1].set_ylim(ylim)
    #ax[1].set_ylim(pd_ylim)
    #ax[0].legend(loc ='upper right', fontsize=desiredfontsize)
    #ax[2].set_xlim(xlim)
    #ax[2].set_ylim([-90, 90])
    #ax[2].set_xticks(xticks)
    if save_fig:
        fig.savefig(savename+'.pdf', dpi=dpi)
        fig.savefig(savename+'.png', dpi=dpi)
    plt.show()

def merge_files(fits_files, combined_name):
    '''This is an array of files'''
    tables = [Table.read(file) for file in fits_files]
    combined_table = vstack(tables)

    # Save the combined table to a new FITS file
    combined_table.write(combined_name, overwrite=True)

def show_orbit(time, XMMT, Ps):
    phase=(time-XMMT)/Ps
    ph=phase-int(np.min(phase))
    ph1=np.array(ph, dtype=int)
    num=np.where(np.abs(ph1-np.roll(ph1, 1))>0)
    numb=len(ph[num])-1
    stat=np.empty(numb)
    e=np.empty(numb)
    sp=ph1[num]
    h=np.linspace(0, numb, numb+1)
    #print(ph1)
    #print(numb)
    #print(h)
    o_p=ph-ph1
    #print(o_p)
    for n in range(0, numb+1):
        i=np.where((ph>sp[n])&(ph<sp[n]+1))
        o_p[i]+=h[n]
    return o_p






def plot_with_asymptotic_behavior(x, y, jump_thresh, ylim=[-90, 90]):
    x = np.asarray(x)
    y = np.asarray(y)

    x_plot = []
    y_plot = []

    dy = np.abs(np.diff(y))
    jumps = np.where(dy > jump_thresh)[0]

    last_idx = 0

    for j in jumps:
        # Left side segment
        x_seg = x[last_idx:j+1].tolist()
        y_seg = y[last_idx:j+1].tolist()

        # Add a point that goes to plot edge
        direction = np.sign(y[j+1] - y[j])
        y_edge = ylim[1]+1 if direction > 0 else ylim[0]-1
        if y[j+1] > y[j]:
            # Function is going up, go to the top of the plot
            y_edge2 = ylim[1]+1
            y_edge1=ylim[0]-1
        else:
            # Function is going down, go to the bottom of the plot
            y_edge1 = ylim[1]+1
            y_edge2=ylim[0]-1
        x_seg.append(x[j])
        y_seg.append(y_edge1)

        # Insert NaN to break line
        x_seg.append(np.nan)
        y_seg.append(np.nan)

        # Add pre-jump segment
        x_plot.extend(x_seg)
        y_plot.extend(y_seg)

        # Add post-jump asymptotic entry
        x_plot.extend([np.nan, x[j+1], x[j+1]])
        y_plot.extend([np.nan, y_edge2, y[j+1]])

        last_idx = j + 1

    # Add final segment
    x_plot.extend(x[last_idx:].tolist())
    y_plot.extend(y[last_idx:].tolist())
    
    return x_plot, y_plot