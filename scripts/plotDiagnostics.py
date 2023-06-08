#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:01:57 2023

@author: ge56beh
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy import genfromtxt

import readDiagnostics as rd
from Plot import XY,utils

directry = "../data"
filenames = [
               # "vring4by80_euler",
               # "vring5by100_euler",
                "vring6by117_euler",
               # "vring4by80_rk4",
               # "vring5by100_rk4",
                # "vring6by117_rk4",
              ]
Fig = []
data_dir ='/home/HT/ge56beh/Work/pawan_my/validatadata'



for filename in filenames:    
    file = f"{filename}Wake.diagnosis"
    rd = rd.readDiagnostics(f"{directry}/{file}")
    nu = rd.nu
    diagnostics_dictRefA9_5, diagnostics_dictRefAT3 = utils.get_validationdata(data_dir,nu = nu)
    time = np.array(rd.time)
    diagnostics_dict = {#'Time': rd.time,
                        'Total Vorticity':np.array(rd.totalvorticity)[:,2] ,
                        'Normalised Linear Impulse':np.array(rd.linearimpulse)[:,2]/np.array(rd.linearimpulse)[0,2] ,
                        'Angular Impulse':np.array(rd.angularimpulse)[:,2],
                        'Helicity': np.array(rd.helicity), 
                        'Normalised Enstrophy':np.array(rd.enstrophy)/np.array(rd.enstrophy)[0],
                        'Normalised Enstrophy (div-free)':np.array(rd.enstrophyF)/np.array(rd.enstrophyF)[0],
                        'Normalised Kinetic Energy': np.array(rd.kineticenergy)/np.array(rd.kineticenergy)[0],
                        'Normalised Kinetic Energy (div-free)': np.array(rd.kineticenergyF)/np.array(rd.kineticenergyF)[0],
                        'Centroid Pos' : np.array(rd.centroidpos), 
                        'nue': -nu*np.array(rd.enstrophy),
        }
    dkedt = diagnostics_dict['Normalised Kinetic Energy'][1:] - diagnostics_dict['Normalised Kinetic Energy'][:-1]
    Vc = (diagnostics_dict['Centroid Pos'][1:] - diagnostics_dict['Centroid Pos'][:-1])/(time[1:]-time[:-1])
    diagnostics_dict.update({'KE_rate': dkedt,
                             'Vc': Vc})                                               

    limits = {#'Time': [0,5],
            'Total Vorticity':[-1,1],
            'Normalised Linear Impulse':[0.875,1.125],
            'Angular Impulse':[-1,1],
            'Helicity': [-1,1],
            'Normalised Enstrophy':[0,1.25],
            'Normalised Enstrophy (div-free)':[0,1.25],
            'Normalised Kinetic Energy':[0,1.25],
            'Normalised Kinetic Energy (div-free)':[0,1.25],
            'KE_rate':[-2.0,1.0],
            'Centroid Pos':[0,1.0],
            'nue':[-2,1],
            'Vc':[-2,1],
        }
    labels = {'Time':'Time (s)',
              'Total Vorticity':'Total Vorticity',
            'Normalised Linear Impulse':'Normalised Linear Impulse',
            'Angular Impulse':'Angular Impulse',
            'Helicity': 'Helicity',
            'Normalised Enstrophy':'Normalised Enstrophy',
            'Normalised Enstrophy (div-free)':'Normalised Enstrophy (div-free)',
            'Normalised Kinetic Energy':'Normalised Kinetic Energy',
            'Normalised Kinetic Energy (div-free)':'Normalised Kinetic Energy (div-free)',
            'KE_rate': 'KE_rate',
            'Centroid Pos': 'Zc',
            'nue': r'-$\nu$E',
            'Vc':'Vc',
            }
    Fig.append(plt.figure(figsize=(18.5,10)))
    imax, jmax = 7,2      
    Gs=gridspec.GridSpec(imax, jmax, figure=Fig[-1])
    Ax=[]
    
    data_test={}
    for idx, (diag, val) in enumerate(diagnostics_dict.items()):
        data={
            'xlabel':labels['Time'],
            'data':{},
            }
        data['data'].update({'VVPM':np.stack((time[-len(val):],val),axis=1)})#Vc, dkedt not same length as time
        # data_test[diag]    
        case = filename.split('_')[0]
        if diag in diagnostics_dictRefA9_5[case]:
            data['data'].update({'RefA9_5':diagnostics_dictRefA9_5[case][diag]})
        if diag in diagnostics_dictRefAT3[case]:
            data['data'].update({'RefAT3':diagnostics_dictRefAT3[case][diag]})
        data['ylabel'] = labels[diag]
        Ax.append(Fig[-1].add_subplot(imax, jmax, idx+1))
        title = ''
        XY.plotvalidation(Ax[idx],title,data,limits[diag])
        figname = ''.join(diag.split())
        XY.plot_and_savesubplot(figname,data,limits[diag],directry,savetikz=True)



