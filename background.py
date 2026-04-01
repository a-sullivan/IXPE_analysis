#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 12:36:58 2025

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
import json

const = Constants()
IXPE_ref_time_MJD_d = 57754
IXPE_ref_time_MJD_frac = 0.00080074074074

class ixpe_background(object):
    def __init__(self):
        self.f_det1_bkg = None
        self.f_det1_bkg_error = None
        self.f_det2_bkg = None
        self.f_det2_bkg_error = None
        self.f_det3_bkg = None
        self.f_det3_bkg_error = None
        
        self.q_det1_bkg = None
        self.q_det1_bkg_error = None
        self.q_det2_bkg = None
        self.q_det2_bkg_error = None
        self.q_det3_bkg = None
        self.q_det3_bkg_error = None
        
        self.u_det1_bkg = None
        self.u_det1_bkg_error = None
        self.u_det2_bkg = None
        self.u_det2_bkg_error = None
        self.u_det3_bkg = None
        self.u_det3_bkg_error = None
        
       
        self.f_det1_pbkg = None
        self.f_det1_pbkg_error = None
        self.f_det2_pbkg = None
        self.f_det2_pbkg_error = None
        self.f_det3_pbkg = None
        self.f_det3_pbkg_error = None
            
        self.q_det1_pbkg = None
        self.q_det1_pbkg_error = None
        self.q_det2_pbkg = None
        self.q_det2_pbkg_error = None
        self.q_det3_pbkg = None
        self.q_det3_pbkg_error = None
                
        self.u_det1_pbkg = None
        self.u_det1_pbkg_error = None
        self.u_det2_pbkg = None
        self.u_det2_pbkg_error = None
        self.u_det3_pbkg = None
        self.u_det3_pbkg_error = None
        
    def det_photon_bkg(self, det,  f_bkg, f_bkg_error,  q_bkg, q_bkg_error, u_bkg, u_bkg_error):
        
        if det==1:
            self.f_det1_bkg = f_bkg
            self.f_det1_bkg_error = f_bkg_error
            
            self.q_det1_bkg = q_bkg
            self.q_det1_bkg_error = q_bkg_error
            
            self.u_det1_bkg = u_bkg
            self.u_det1_bkg_error = u_bkg_error
        elif det==2:
            self.f_det2_bkg = f_bkg
            self.f_det2_bkg_error = f_bkg_error
            
            self.q_det2_bkg = q_bkg
            self.q_det2_bkg_error = q_bkg_error
            
            self.u_det2_bkg = u_bkg
            self.u_det2_bkg_error = u_bkg_error
        elif det==3:
            self.f_det3_bkg = f_bkg
            self.f_det3_bkg_error = f_bkg_error
            
            self.q_det3_bkg = q_bkg
            self.q_det3_bkg_error = q_bkg_error
            
            self.u_det3_bkg = u_bkg
            self.u_det3_bkg_error = u_bkg_error
            
    def det_particle_bkg(self, det, f_pbkg, f_pbkg_error,  q_pbkg, q_pbkg_error, u_pbkg, u_pbkg_error):
        if det==1:
            self.f_det1_pbkg = f_pbkg
            self.f_det1_pbkg_error = f_pbkg_error
            
            self.q_det1_pbkg = q_pbkg
            self.q_det1_pbkg_error = q_pbkg_error
            
            self.u_det1_pbkg = u_pbkg
            self.u_det1_pbkg_error = u_pbkg_error
        elif det==2:
            self.f_det2_pbkg = f_pbkg
            self.f_det2_pbkg_error = f_pbkg_error
            
            self.q_det2_pbkg = q_pbkg
            self.q_det2_pbkg_error = q_pbkg_error
            
            self.u_det2_pbkg = u_pbkg
            self.u_det2_pbkg_error = u_pbkg_error
        elif det==3:
            self.f_det3_pbkg = f_pbkg
            self.f_det3_pbkg_error = f_pbkg_error
            
            self.q_det3_pbkg = q_pbkg
            self.q_det3_pbkg_error = q_pbkg_error
            
            self.u_det3_pbkg = u_pbkg
            self.u_det3_pbkg_error = u_pbkg_error
            
    def save_bkg(self, file_name='main', dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/') :
        
        with open(dir_name+file_name+"_det1_photons.json", "w") as f:
            json.dump({"f_det1_bkg": self.f_det1_bkg, "q_det1_bkg": self.q_det1_bkg, "u_det1_bkg": self.u_det1_bkg}, f)
            
        with open(dir_name+file_name+"_det2_photons.json", "w") as f:
            json.dump({"f_det2_bkg": self.f_det2_bkg, "q_det2_bkg": self.q_det2_bkg, "u_det2_bkg": self.u_det2_bkg}, f)
            
        with open(dir_name+file_name+"_det3_photons.json", "w") as f:
            json.dump({"f_det3_bkg": self.f_det3_bkg, "q_det3_bkg": self.q_det3_bkg, "u_det3_bkg": self.u_det3_bkg}, f)
            
        with open(dir_name+file_name+"_det1_particles.json", "w") as f:
            json.dump({"f_det1_pbkg": self.f_det1_pbkg, "q_det1_pbkg": self.q_det1_pbkg, "u_det1_pbkg": self.u_det1_pbkg}, f)
            
        with open(dir_name+file_name+"_det1_particles_error.json", "w") as f:  
            json.dump({"f_det1_pbkg_error": self.f_det1_pbkg_error, "q_det1_pbkg_error": self.q_det1_pbkg_error, "u_det1_pbkg_error": self.u_det1_pbkg_error}, f)
        
        with open(dir_name+file_name+"_det2_particles.json", "w") as f:
            json.dump({"f_det2_pbkg": self.f_det2_pbkg, "q_det2_pbkg": self.q_det2_pbkg, "u_det2_pbkg": self.u_det2_pbkg}, f)
        
        with open(dir_name+file_name+"_det2_particles_error.json", "w") as f:  
            json.dump({"f_det2_pbkg_error": self.f_det2_pbkg_error, "q_det2_pbkg_error": self.q_det2_pbkg_error, "u_det2_pbkg_error": self.u_det2_pbkg_error}, f)
        
        with open(dir_name+file_name+"_det3_particles.json", "w") as f:
            json.dump({"f_det3_pbkg": self.f_det3_pbkg, "q_det3_pbkg": self.q_det3_pbkg, "u_det3_pbkg": self.u_det3_pbkg}, f)
        
        with open(dir_name+file_name+"_det3_particles_error.json", "w") as f:
            json.dump({"f_det3_pbkg_error": self.f_det3_pbkg_error, "q_det3_pbkg_error": self.q_det3_pbkg_error, "u_det3_pbkg_error": self.u_det3_pbkg_error}, f)
    
    def load_bkg(self, file_name='main', dir_name='/Users/asullivan/Stanford/J1723/IXPE/03006799/bkg_models/') :
        with open(dir_name+file_name+"_det1_photons.json", "r") as f:
            data=json.load(f)
            self.f_det1_bkg = data["f_det1_bkg"]
            
            self.q_det1_bkg = data["q_det1_bkg"]
            
            self.u_det1_bkg = data["u_det1_bkg"]
            
        with open(dir_name+file_name+"_det2_photons.json", "r") as f:
            data=json.load(f)
            self.f_det2_bkg = data["f_det2_bkg"]
            
            self.q_det2_bkg = data["q_det2_bkg"]
            
            self.u_det2_bkg = data["u_det2_bkg"]
            
        with open(dir_name+file_name+"_det3_photons.json", "r") as f:
            data=json.load(f)
            self.f_det3_bkg = data["f_det3_bkg"]
            
            self.q_det3_bkg = data["q_det3_bkg"]
            
            self.u_det3_bkg = data["u_det3_bkg"]
            
        with open(dir_name+file_name+"_det1_particles.json", "r") as f:
            data=json.load(f)
            self.f_det1_pbkg = data["f_det1_pbkg"]
            
            self.q_det1_pbkg = data["q_det1_pbkg"]
            
            self.u_det1_pbkg = data["u_det1_pbkg"]
            
        with open(dir_name+file_name+"_det2_particles.json", "r") as f:
            data=json.load(f)
            self.f_det2_pbkg = data["f_det2_pbkg"]
            
            self.q_det2_pbkg = data["q_det2_pbkg"]
            
            self.u_det2_pbkg = data["u_det2_pbkg"]
            
        with open(dir_name+file_name+"_det3_particles.json", "r") as f:
            data=json.load(f)
            self.f_det3_pbkg = data["f_det3_pbkg"]
            
            self.q_det3_pbkg = data["q_det3_pbkg"]
            
            self.u_det3_pbkg = data["u_det3_pbkg"]
            
        with open(dir_name+file_name+"_det1_particles_error.json", "r") as f:
            data=json.load(f)
            self.f_det1_pbkg_error = data["f_det1_pbkg_error"]
            
            self.q_det1_pbkg_error = data["q_det1_pbkg_error"]
            
            self.u_det1_pbkg_error = data["u_det1_pbkg_error"]
            
        with open(dir_name+file_name+"_det2_particles_error.json", "r") as f:
            data=json.load(f)
            self.f_det2_pbkg_error = data["f_det2_pbkg_error"]
            
            self.q_det2_pbkg_error = data["q_det2_pbkg_error"]
            
            self.u_det2_pbkg_error = data["u_det2_pbkg_error"]
            
        with open(dir_name+file_name+"_det3_particles_error.json", "r") as f:
            data=json.load(f)
            self.f_det3_pbkg_error = data["f_det3_pbkg_error"]
            
            self.q_det3_pbkg_error = data["q_det3_pbkg_error"]
            
            self.u_det3_pbkg_error = data["u_det3_pbkg_error"]
                
            
            
            
            
            