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
    def __init__(self, filename, du_id, NN=False, plot_modf=False, save_modf_plot=False, bg_prob=False):
        self.filename = filename
        file = fits.open(filename)
        data=file[1].data
        self.NN=NN
        self.bg_prob=bg_prob
        self.set_irf = load_irf_set(du_id =du_id)
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
                reg=np.where(((self.total_image.X-x)**2+(self.total_image.Y-y)**2< r**2) & ((self.total_image.X-x)**2+(self.total_image.Y-y)**2> rin**2))
            else:
                reg=np.where((self.total_image.X-x)**2+(self.total_image.Y-y)**2< r**2)
            if self.NN:
                return photon_region(self.total_image.time, self.total_image.X, self.total_image.Y, self.total_image.Q, self.total_image.U, self.total_image.PI, self.total_image.W_MOM, NN=self.NN, W_NN=self.total_image.W_NN, src=reg)
            else:
                return photon_region(self.total_image.time, self.total_image.X, self.total_image.Y, self.total_image.Q, self.total_image.U, self.total_image.PI, self.total_image.W_MOM,  src=reg)
        else:
            if annulus:
                reg=np.where(((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2< r**2) & ((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2> rin**2))
            else:
                reg=np.where((self.total_image.X[mask]-x)**2+(self.total_image.Y[mask]-y)**2< r**2)
            if self.NN:    
                return photon_region(self.total_image.time[mask], self.total_image.X[mask], self.total_image.Y[mask], self.total_image.Q[mask], self.total_image.U[mask], self.total_image.PI[mask], self.total_image.W_MOM[mask], NN=self.NN, W_NN=self.total_image.W_NN[mask], src=reg)
            else:
                return photon_region(self.total_image.time[mask], self.total_image.X[mask], self.total_image.Y[mask], self.total_image.Q[mask], self.total_image.U[mask], self.total_image.PI[mask], self.total_image.W_MOM[mask], src=reg)
    def src_region(self, x, y, r):
        self.src_r = r
        self.src_x = x
        self.src_y = y
        self.src_photons = self.select_region(x, y, r)
    def bkg_region(self, x, y, r, annulus=False, rin=0):
        self.bkg_x = x
        self.bkg_y = y        
        self.bkg_r = r
        self.annulus=annulus
        if annulus:
            self.bkg_rin=rin
        else:
            self.bkg_rin=0
        self.bkg_photons = self.select_region(x, y, r, annulus=annulus, rin=rin)  

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
    def make_orbital_LC(self, nbins, Emin, Emax, Porb, TASC, weights=1, use_weights=True, bkg_sub=0):
        '''weights can be one of 0 (meaning no weights), 1 (meaning moments), and 2 (meaning neural net weights)'''
            
        Echoose_src=np.where((self.src_photons_clean.E>Emin) & ( self.src_photons_clean.E<Emax))
        Echoose_bkg = np.where((self.bkg_photons_clean.E>Emin) & ( self.bkg_photons_clean.E<Emax))
        
        if weights>0:
            # calculate the effective areas and modulation factors for every photon
            src_energies=self.src_photons_clean.E[Echoose_src]
            bkg_energies=self.bkg_photons_clean.E[Echoose_bkg]

            src_aeffs = self.set_irf.aeff(src_energies)/self.set_irf.aeff(src_energies)
            bkg_aeffs = self.set_irf.aeff(bkg_energies)/self.set_irf.aeff(bkg_energies)
            if weights == 1:
                src_modfs = self.set_irf.modf(src_energies)
                bkg_modfs = self.set_irf.modf(bkg_energies)
            elif weights ==2:
                src_modfs = self.NN_Modf(src_energies)
                bkg_modfs = self.NN_Modf(bkg_energies)
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
            print('exposure time: {}'.format(exposure ))
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
                MOM_src = self.src_photons_clean.W_MOM[Echoose_src]
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
        if self.bg_prob:
            mask_src=np.where((self.total_image.BG_PROB<bg_prob_thresh) & (mask))
        else:
            mask_src=mask
        if self.bg_prob:
            mask_bkg=np.where((self.total_image.BG_PROB>bg_prob_thresh) & (mask))
        else:
            mask_bkg=mask
        self.bkg_photons_clean = self.select_region(self.bkg_x, self.bkg_y, self.bkg_r,annulus=self.annulus, rin=self.bkg_rin, mask=mask_bkg) 

        self.src_photons_clean = self.select_region(self.src_x, self.src_y, self.src_r, mask=mask_src) 
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
     return np.arctan2(U, Q)*180./np.pi/2

def EVPA_error(Q, U, Qe, Ue):
    dEVPAdu = 1/(Q*(1+(U/Q)**2))
    dEVPAdq = -U/Q**2/(1+(U/Q)**2)
    return np.sqrt(dEVPAdu**2*Ue**2+dEVPAdq**2*Qe**2)*180./np.pi/2


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

def add_datasets(fluxes, errors):
    flux_sum = sum(fluxes)
    error_sum=0
    for error in errors:
        error_sum = error**2
    error_sum=error_sum**(0.5)
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
    def __init__(self, filename,  du_id, srcd1_x, srcd1_y, srcd1_r, bkgd1_x, bkgd1_y, bkgd1_r, Pb, Ephemeris_IXPE, nbins = 5, timebin=1000, Emin=2, Emax=8, weights=1, use_weights=True, excise_bkg_flares=False, Emin_flare=2, Emax_flare=8, backgroundthresh=5, annulus=False, rin=0, compute_spectrum=False, specbins=10, bkg_sub=0, bg_prob=False, bg_prob_thresh=0.75):
        if weights ==2:
            NN=True
        else:
            NN=False
        self.detector =IXPE_fits(filename, du_id=du_id, NN=NN, bg_prob=bg_prob)
        self.detector.src_region(srcd1_x, srcd1_y, srcd1_r)
        self.detector.bkg_region(bkgd1_x, bkgd1_y, bkgd1_r, annulus=annulus, rin=rin)
        self.detector.make_LC(timebin, Emin_flare, Emax_flare, weights=weights)
        self.detector.excise_background_flares(threshold=backgroundthresh, bg_prob_thresh=bg_prob_thresh)
        self.detector.make_orbital_LC(nbins, Emin, Emax, Pb, Ephemeris_IXPE, weights=weights, use_weights=use_weights, bkg_sub=bkg_sub)
        if compute_spectrum:
            self.detector.compute_spectrum(specbins, Emin=Emin, Emax=Emax)

def check_array(variable):
    if isinstance(variable, numbers.Number):
        return True
    elif isinstance(variable, (list, np.ndarray)):
        return False

class run_pipeline(object):
    def __init__(self, direct_list, srcd1_x_list, srcd1_y_list, srcd1_r_list, bkgd1_x_list, bkgd1_y_list, bkgd1_r_list, Pb_list, Ephemeris_IXPE_list, nbins = 5, timebin=1000, Emin=2, Emax=8, bkg_sub=0, annulus=False, bkgd1_rin=0, weights=1, use_weights=True, compute_spectrum=False, specbins=10, bg_prob=False, bg_prob_thresh=0.5):
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
        keep_det1=1
        keep_det2=1
        keep_det3=1
        for k, direct in enumerate(direct_list):
            pattern =direct+'det1*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d1 = glob.glob(pattern)
            pattern =direct+'det2*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d2 = glob.glob(pattern)
            pattern =direct+'det2*.fits' #"event_nn_l2/03006799/ixpe030067_tot_det1*.fits"
            matching_files_d3 = glob.glob(pattern)
            #print('detector 1 file ')
            #print(matching_files_d1)
            detector1=standard_pipeline(matching_files_d1[0], 1, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector2=standard_pipeline(matching_files_d2[0], 2, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])
            detector3=standard_pipeline(matching_files_d3[0], 3, srcd1_x_list[k], srcd1_y_list[k], srcd1_r_list[k], bkgd1_x_list[k], bkgd1_y_list[k], bkgd1_r_list[k], Pb_list[k], Ephemeris_IXPE_list[k], nbins = nbins_list[k], timebin=timebin_list[k], Emin=Emin_list[k], Emax=Emax_list[k], annulus=annulus_list[k], rin=bkgd1_rin_list[k], weights=weights_list[k], compute_spectrum=compute_spectrum_list[k], specbins=specbins_list[k], use_weights=use_weights_list[k], bkg_sub=bkg_sub_list[k], bg_prob=bg_prob_list[k], bg_prob_thresh=bg_prob_thresh_list[k])

            I_alldetectors=[keep_det1*detector1.detector.flux_orb_LC_src, keep_det2*detector2.detector.flux_orb_LC_src, keep_det3*detector3.detector.flux_orb_LC_src]
            Ie_alldetectors=[keep_det1*detector1.detector.error_orb_LC_src, keep_det2*detector2.detector.error_orb_LC_src, keep_det3*detector3.detector.error_orb_LC_src]
            Q_alldetectors = [keep_det1*detector1.detector.Q_LC_orb_src, keep_det2*detector2.detector.Q_LC_orb_src, keep_det3*detector3.detector.Q_LC_orb_src]
            Qe_alldetectors = [keep_det1*detector1.detector.Qerr_LC_orb_src, keep_det2*detector2.detector.Qerr_LC_orb_src, keep_det3*detector3.detector.Qerr_LC_orb_src]
            U_alldetectors = [keep_det1*detector1.detector.U_LC_orb_src, keep_det2*detector2.detector.U_LC_orb_src, keep_det3*detector3.detector.U_LC_orb_src]
            Ue_alldetectors = [keep_det1*detector1.detector.Uerr_LC_orb_src, keep_det2*detector2.detector.Uerr_LC_orb_src, keep_det3*detector3.detector.Uerr_LC_orb_src]
            I_combined, Ierr_combined = add_datasets(I_alldetectors, Ie_alldetectors)
            Q_combined, Qerr_combined = add_datasets(Q_alldetectors, Qe_alldetectors)
            U_combined, Uerr_combined = add_datasets(U_alldetectors, Ue_alldetectors)
            N_total = detector1.detector.N_perbin+detector2.detector.N_perbin+detector3.detector.N_perbin
            #MDP_combined = MDP_mom((detector1.detector.mu+detector2.detector.mu+detector3.detector.mu)/3, I_combined, Ierr_combined**2)
            exposure_time=keep_det1*detector1.detector.exposure_time+keep_det2*detector2.detector.exposure_time+keep_det3*detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time#(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)/np.max(detector1.detector.flux_orb_LC_bkg+detector2.detector.flux_orb_LC_bkg+detector3.detector.flux_orb_LC_bkg)#detector1.detector.exposure_time+detector2.detector.exposure_time+detector3.detector.exposure_time #total exposure time in a bin
            #print('Total exposure time {}'.format(np.sum(exposure_time/3)))
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

def plot_polarization(list_pipeline, list_labels, list_colors, shift=True, plot2orb=False, pd_ylim=[0, 0.4], desiredcapsize=3, save_fig=False, savename='polarization_plot', dpi=300, desiredfontsize=16, plot_bkg=False, plot_MDP=True):
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

    #ax[0].plot(pol60.phase, pol60.I_LC[inclnum]/np.max(pol60.I_LC[inclnum]), color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')
    #ax[1].plot(pol60.phase, pol60.PD[inclnum], color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')
    #ax[2].plot(pol60.phase, pol60.PA[inclnum], color=colorgam60, linestyle=linestylegam60, label = r'$\gamma_{rad}=60$')

    #ax[0].plot(pol_no_cooling.phase, pol_no_cooling.I_LC[inclnum]/np.max(pol_no_cooling.I_LC[inclnum]), color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[1].plot(pol_no_cooling.phase, pol_no_cooling.PD[inclnum], color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[2].plot(pol_no_cooling.phase, pol_no_cooling.PA[inclnum], color=colornocool,linestyle=linestylenocool, label = r'No Cooling')
    #ax[1].plot(LC_60[0], LC_60[1]/LC_norm60, color='green', label = r'$\gamma_{rad}=60$')
    #ax[2].plot(LC_nocool[0], LC_nocool[1]/LC_norm_nocool, color='blue', label = r'No Cooling')

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