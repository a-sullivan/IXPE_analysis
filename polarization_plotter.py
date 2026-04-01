#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 05:55:40 2025

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
import leakagelib_v8
import copy
import scipy.stats
from scipy.stats import norm

const = Constants()
IXPE_ref_time_MJD_d = 57754
IXPE_ref_time_MJD_frac = 0.00080074074074

def plot_polarization(list_pipeline, list_labels, list_colors, shift=True, norm_flux = True, plot2orb=False, i_ylim=[0, 1.2], pd_ylim=[0, 0.4], desiredcapsize=3, save_fig=False, savename='polarization_plot', dpi=300, desiredfontsize=16, plot_bkg=False, plot_MDP=True, combine_all=False, no_show=False, combine_color='black', combine_color2='darkmagenta', list_pipeline2=None, list_labels2=None, list_colors2=None, combine_title='Total', combine_title2='Total2', model1 = None, model1color='blue', model1_label=None, model_offset=None, model2= None, model2color='orange', model2_label=None, i_ibs_model=None, i_const_model=None, q_ibs_model1=0., u_ibs_model1=0., q_const_model1=0, u_const_model1=0, q_ibs_model2=0., u_ibs_model2=0., q_const_model2=0, u_const_model2=0, rotation_dir_model1=1, rotation_dir_const1=1, rotation_dir_model2=1, rotation_dir_const2=1, model3= None, model3color='green', model3_label=None, q_ibs_model3=0., u_ibs_model3=0., q_const_model3=0, u_const_model3=0, rotation_dir_model3=1, rotation_dir_const3=1, model3linestyle='dashed', n_take=2, show_legend=False, nonGaussian_errors=True):
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
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
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
            if len(list_pipeline.I_combined_total)>4:
                norm=np.mean(np.sort(list_pipeline.I_combined_total)[-n_take:])
            else:
                norm=np.max(list_pipeline.I_combined_total)
        else:
            norm=1.
            
        ax[0].errorbar(list_pipeline.phases[n], list_pipeline.I_combined_total/norm, yerr=list_pipeline.Ierr_combined_total/norm, color=combine_color, label = combine_title, linestyle='None', marker='o', capsize=desiredcapsize)
        if nonGaussian_errors:
            if list_pipeline.phases_orb_LC_manual is not None:
                ax[1].errorbar(list_pipeline.phases_orb_LC_manual, list_pipeline.PD_src_total_manual*100, yerr=(list_pipeline.PDerr_src_total_lower_manual*100, list_pipeline.PDerr_src_total_upper_manual*100), color=combine_color, linestyle='None', marker='o', capsize=desiredcapsize)
            else:
                ax[1].errorbar(list_pipeline.phases[n], list_pipeline.PD_src_total*100, yerr=(list_pipeline.PDerr_src_total_lower*100, list_pipeline.PDerr_src_total_upper*100), color=combine_color, linestyle='None', marker='o', capsize=desiredcapsize)
            ax[2].errorbar(list_pipeline.phases[n], list_pipeline.EVPA_combined_total, yerr=(list_pipeline.EVPAerr_combined_total_lower, list_pipeline.EVPAerr_combined_total_upper),  color=combine_color, linestyle='None', marker='o', capsize=desiredcapsize)
        
        else:
            if list_pipeline.phases_orb_LC_manual is not None:
                ax[1].errorbar(list_pipeline.phases_orb_LC_manual, list_pipeline.PD_src_total_manual*100,  yerr=list_pipeline.PDerr_src_total_manual*100, color=combine_color, label = 'Total', linestyle='None', marker='o', capsize=desiredcapsize)
            else:   
                ax[1].errorbar(list_pipeline.phases[n], list_pipeline.PD_src_total*100, yerr=list_pipeline.PDerr_src_total*100, color=combine_color, label = 'Total', linestyle='None', marker='o', capsize=desiredcapsize)
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
            
            i_const_line=np.full(len(model1[:, 0]), i_const_model/(i_ibs_model+i_const_model)*norm)
            
            ax[0].plot(model1[:, 0], i_const_line, color='grey', linestyle='dashed')
            #ax[1].plot(model1[:, 0], (model1[:, 2]**2*i_ibs_model**2*(q_ibs_model1**2+u_ibs_model1**2)+(i_const_model**2*(q_const_model1**2+u_const_model1**2)))/(i_ibs_model+i_const_model), color=model1color)
        threshold = 90.0  # set based on what "too big" means in your data
        
        if i_ibs_model is not None:
            q_ibs= q_calculate(model1[:, 2], model1[:, 3])*q_ibs_model1 - u_calculate(model1[:, 2], model1[:, 3])*u_ibs_model1
            u_ibs= rotation_dir_model1*u_calculate(model1[:, 2], model1[:, 3])*q_ibs_model1+ rotation_dir_model1* q_calculate(model1[:, 2], model1[:, 3])*u_ibs_model1 
            
            q_tot = q_ibs*i_ibs_model + q_const_model1*i_const_model
            u_tot = u_ibs*i_ibs_model + u_const_model1*i_const_model
            evpa_model1= evpa_calc(q_tot, u_tot)
            ax[1].plot(model1[:, 0], np.sqrt((q_tot**2+u_tot**2))/(i_ibs_model+i_const_model)*100., color=model1color, label=model1_label)
        else:
            evpa_model1=model1[:, 3]
        phase_PA_model1, PA_model1 = plot_with_asymptotic_behavior(model1[:, 0], evpa_model1, threshold, ylim=[-90, 90])
        
        ax[2].plot(phase_PA_model1, PA_model1, color=model1color, label=model1_label)
        
    if model2 is not None:
        #ax[0].plot(model2[:, 0], model2[:, 1], color=model2color)
       
        
        if i_ibs_model is not None:
            q_ibs= q_calculate(model2[:, 2], model2[:, 3])*q_ibs_model2 - u_calculate(model2[:, 2], model2[:, 3])*u_ibs_model2
            u_ibs= rotation_dir_model2*u_calculate(model2[:, 2], model2[:, 3])*q_ibs_model2+rotation_dir_model2* q_calculate(model2[:, 2], model2[:, 3])*u_ibs_model2
            
            q_tot = q_ibs*i_ibs_model + q_const_model2*i_const_model
            u_tot = u_ibs*i_ibs_model + u_const_model2*i_const_model
            evpa_model2= evpa_calc(q_tot, u_tot)
            ax[1].plot(model2[:, 0], np.sqrt((q_tot**2+u_tot**2))/(i_ibs_model+i_const_model)*100, color=model2color, label=model2_label)
        else:
            ax[1].plot(model2[:, 0], model2[:, 2]*100, color=model2color)
            evpa_model2=model2[:, 3]
        
        phase_PA_model2, PA_model2 = plot_with_asymptotic_behavior(model2[:, 0], evpa_model2, threshold, ylim=[-90, 90])
        
        ax[2].plot(phase_PA_model2, PA_model2, color=model2color, label=model2_label)
    
    if model3 is not None:
        #ax[0].plot(model2[:, 0], model2[:, 1], color=model2color)
       
        
        if i_ibs_model is not None:
            q_ibs= q_calculate(model3[:, 2], model3[:, 3])*q_ibs_model3 - u_calculate(model3[:, 2], model3[:, 3])*u_ibs_model3
            u_ibs= rotation_dir_model3*u_calculate(model3[:, 2], model3[:, 3])*q_ibs_model3+rotation_dir_model3*q_calculate(model3[:, 2], model3[:, 3])*u_ibs_model3
            
            q_tot = q_ibs*i_ibs_model + q_const_model2*i_const_model
            u_tot = u_ibs*i_ibs_model + u_const_model2*i_const_model
            evpa_model3= evpa_calc(q_tot, u_tot)
            ax[1].plot(model3[:, 0], np.sqrt((q_tot**2+u_tot**2))/(i_ibs_model+i_const_model)*100, color=model3color, label=model3_label,  linestyle=model3linestyle)
        else:
            ax[1].plot(model3[:, 0], model3[:, 2]*100, color=model3color,  linestyle=model3linestyle)
            evpa_model3=model3[:, 3]
        
        phase_PA_model3, PA_model3 = plot_with_asymptotic_behavior(model3[:, 0], evpa_model3, threshold, ylim=[-90, 90])
        
        ax[2].plot(phase_PA_model3, PA_model3, color=model3color, label=model3_label, linestyle=model3linestyle)
    
    
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
    if norm_flux:
        ax[0].set_ylabel('Normalized Flux', fontsize=desiredfontsize)
    else:
        ax[0].set_ylabel('Flux', fontsize=desiredfontsize)
    if show_legend:
        #ax[0].legend(loc ='upper left', fontsize=desiredfontsize)
        ax[1].legend(loc ='upper right', fontsize=0.8*desiredfontsize)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(i_ylim)
    ax[0].set_xticks(xticks)

    ax[1].set_xlabel('Orbital Phase', fontsize=desiredfontsize)
    ax[1].set_ylabel('Polarization Degree (\%)', fontsize=desiredfontsize)
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
    ax[0].tick_params( labelsize=0.8*desiredfontsize)
    ax[1].tick_params( labelsize=0.8*desiredfontsize)
    ax[2].tick_params( labelsize=0.8*desiredfontsize)
    if save_fig:
        fig.savefig(savename+'.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(savename+'.png', dpi=dpi, bbox_inches='tight', pad_inches=0.05)
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