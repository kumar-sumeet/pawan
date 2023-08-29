#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 10:47:54 2023

@author: ge56beh
"""
from numpy import genfromtxt
import numpy as np

def get_validationdata(data_dir,nu=0.0):
    if nu: #viscous case
        diagnostics_dictRefA9_5 = {
                "vring3by49vring3by49fusion":{},
                "vring2by52vring2by52fissionfusion":{},
                "vring4by80":{},
                "vring5by100":{},
                "vring6by117":{
                                'Total Vorticity': np.array([[0,0],[5.0,0]]),
                                'Normalised Linear Impulse':np.array([[0,1.0],[5.0,1.0]]),
                                'Angular Impulse':np.array([[0,0],[5.0,0]]),
                                'Helicity':np.array([[0,0],[5.0,0]]),
                                'Normalised Enstrophy':genfromtxt(f'{data_dir}/Valentin2022/fig8b_e_ef.csv', delimiter=','),
                                'Normalised Enstrophy (div-free)':genfromtxt(f'{data_dir}/Valentin2022/fig8b_e_ef.csv', delimiter=','),
                                'Normalised Kinetic Energy':genfromtxt(f'{data_dir}/Valentin2022/fig8b_ke_kef.csv', delimiter=','),
                                'Normalised Kinetic Energy (div-free)':genfromtxt(f'{data_dir}/Valentin2022/fig8b_ke_kef.csv', delimiter=','),
                                'Vc': genfromtxt(f'{data_dir}/Valentin2022/fig5b_vi_nondim.csv', delimiter=','),
                                'KE_rate': genfromtxt(f'{data_dir}/Valentin2022/fig5b_dkedt.csv', delimiter=','),
                                'nue':genfromtxt(f'{data_dir}/Valentin2022/fig5b_nue.csv', delimiter=','),
                                },        
        }
        
        
        diagnostics_dictRefAT3 = {
                "vring3by49vring3by49fusion":{
                                'Normalised Linear Impulse':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ43_li.csv', delimiter=','),
                                'Normalised Enstrophy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ43_e.csv', delimiter=','),
                                'Normalised Enstrophy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ43_ef.csv', delimiter=','),
                                'Normalised Kinetic Energy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ43_ke_kef.csv', delimiter=','),
                                'Normalised Kinetic Energy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ43_ke_kef.csv', delimiter=','),
                    
                            },
                "vring2by52vring2by52fissionfusion":{
                                'Normalised Linear Impulse':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ45_li.csv', delimiter=','),
                                'Normalised Kinetic Energy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ45_ke_kef.csv', delimiter=','),
                                'Normalised Kinetic Energy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ45_ke_kef.csv', delimiter=','),                    
                    },
                "vring4by80":{
                            'Vc': genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_dXcdt_nc4.csv', delimiter=','),
                            },
                "vring5by100":{
                                'Vc': genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_dXcdt_nc5.csv', delimiter=','),
                                },
                "vring6by117":{
                                'Normalised Linear Impulse':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_li.csv', delimiter=','),
                                'Normalised Enstrophy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_e_ef.csv', delimiter=','),
                                'Normalised Enstrophy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_e_ef.csv', delimiter=','),
                                'Normalised Kinetic Energy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_ke_kef.csv', delimiter=','),
                                'Normalised Kinetic Energy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_ke_kef.csv', delimiter=','),
                                'Vc': genfromtxt(f'{data_dir}/Winckelmanns1989/figJ36_dXcdt_nc6.csv', delimiter=','),
                                },
        
        }        

    else:#inviscid case

        diagnostics_dictRefA9_5 = {
                "vring4by80":{},
                "vring5by100":{
                                'Total Vorticity': np.array([[0,0],[5,0]]),
                                'Normalised Linear Impulse':np.array([[0,1.0],[5,1.0]]),
                                'Angular Impulse':np.array([[0,0],[5,0]]),
                                'Helicity':np.array([[0,0],[5,0]]),
                                'Normalised Enstrophy':genfromtxt(f'{data_dir}/Valentin2022/fig8a_e.csv', delimiter=','),
                                'Normalised Enstrophy (div-free)':genfromtxt(f'{data_dir}/Valentin2022/fig8a_ef.csv', delimiter=','),
                                'Normalised Kinetic Energy':genfromtxt(f'{data_dir}/Valentin2022/fig8a_ke_kef.csv', delimiter=','),
                                'Normalised Kinetic Energy (div-free)':genfromtxt(f'{data_dir}/Valentin2022/fig8a_ke_kef.csv', delimiter=','),
                                'Vc': genfromtxt(f'{data_dir}/Valentin2022/fig5a_vi_nondim.csv', delimiter=','),
                                'KE_rate': genfromtxt(f'{data_dir}/Valentin2022/fig5a_dkedt.csv', delimiter=','),
                                'nue':np.array([[0,0],[5,0]]),
                                },
                "vring6by117":{},        
        }

        diagnostics_dictRefAT3 = {
                "vring4by80":{
                            'Vc': genfromtxt(f'{data_dir}/Winckelmanns1989/figJ35_dXcdt_nc4.csv', delimiter=','),
                            },
                "vring5by100":{
                                'Normalised Linear Impulse':np.array([[0,1.0],[5,1.0]]),
                                'Normalised Enstrophy':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ35_e.csv', delimiter=','),
                                'Normalised Enstrophy (div-free)':genfromtxt(f'{data_dir}/Winckelmanns1989/figJ35_ef.csv', delimiter=','),
                                'Normalised Kinetic Energy':np.array([[0,1.0],[5,1.0]]),
                                'Normalised Kinetic Energy (div-free)':np.array([[0,1.0],[5,1.0]]),
                                'Vc': genfromtxt(f'{data_dir}/Winckelmanns1989/figJ35_dXcdt_nc5.csv', delimiter=','),
                                },
                "vring6by117":{},
        
        }
        
    return diagnostics_dictRefA9_5, diagnostics_dictRefAT3
        
    