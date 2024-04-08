#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:11:43 2020

@author: SumeetKumar
"""

import pickle
import numpy as np
import tecplot as tp
import sys
import os
import math
from shutil import copy2

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import readWake as rw

def saveplttodirs(dirstosave_plt,pltfilepath):
    for dirtosave_plt in dirstosave_plt:
        try:
            os.makedirs(dirtosave_plt)
        except FileExistsError:
            pass                # directory already exists                   
        copy2(pltfilepath, f'{dirtosave_plt}')
        print(f'wake plt saved to {dirtosave_plt}')
    
def gen_plt(directry,filename,wake):
    nTimesteps = wake.nTimesteps
    time = wake.time
    nParticles = wake.nParticles
    position = wake.position
    vorticity = wake.vorticity
    radius = wake.radius

    with tp.session.suspend():       
        tp.new_layout()
        frametp = tp.active_frame()
        dataset = frametp.dataset
        for var in ['x','y','z','vorticity_x','vorticity_y','vorticity_z',
                    'radius','vorticitymag']:
            dataset.add_variable(var)
        zones={}
            
        for t_idx, t in enumerate(time):
            print('time step-->',t_idx,f' of {nTimesteps}' )
            if t_idx>10: continue    
            position_t = position[t_idx]
            vorticity_t = vorticity[t_idx]
            radius_t = radius[t_idx]
            ord_zone_tup = (1,np.size(position_t,axis=0),1)
            zones[str(t)] = dataset.add_ordered_zone("{:5.4f}".format(t), 
                                                         ord_zone_tup, 
                                                         solution_time=t, 
                                                         strand_id=1)
            pos_x, pos_y, pos_z = position_t[:,0], position_t[:,1], position_t[:,2]
            vor_x, vor_y, vor_z = vorticity_t[:,0], vorticity_t[:,1], vorticity_t[:,2] 
            zones[str(t)].values('x')[:]=pos_x
            zones[str(t)].values('y')[:]=pos_y
            zones[str(t)].values('z')[:]=pos_z
            zones[str(t)].values('vorticity_x')[:]=vor_x
            zones[str(t)].values('vorticity_y')[:]=vor_y
            zones[str(t)].values('vorticity_z')[:]=vor_z
            zones[str(t)].values('radius')[:]=radius_t
            vorticity_mag = np.sqrt(np.power(vor_x,2)+np.power(vor_y,2)+\
                                                            np.power(vor_z,2))
            zones[str(t)].values('vorticitymag')[:]=vorticity_mag
        #                azimuth=(90*blade_no)+(360*t/T)
        #                macr_str=f"""$!AttachText 
        #                  AnchorPos
        #                    {{
        #                    X = 53.08578008059873
        #                    Y = 66.21761658031089
        #                    }}
        #                  TextShape
        #                    {{
        #                    IsBold = No
        #                    }}
        #                  Text = 'Azimuth={azimuth:.1f} deg'"""
        #                tp.macro.execute_command(macr_str)
        tp.data.save_tecplot_plt(f"{directry}/{filename}.plt", dataset=dataset)                                  
        print("...wake plt done")

    
def gen_dat(directry,filename,wake):
    
    nTimesteps = wake.nTimesteps
    time = wake.time
    nParticles = wake.nParticles
    nParticlesMax = wake.nParticlesMax
    position = wake.position
    vorticity = wake.vorticity
    radius = wake.radius
    active = wake.active
    data_dict = {
                 'x':[pos_t[:,0] for pos_t in position],
                 'y':[pos_t[:,1] for pos_t in position],
                 'z':[pos_t[:,2] for pos_t in position],
                 'vor_x':[vor_t[:,0] for vor_t in vorticity],
                 'vor_y':[vor_t[:,1] for vor_t in vorticity],
                 'vor_z':[vor_t[:,2] for vor_t in vorticity],
                 'radius':radius,
                 'active':active,                 
                 'Vor_strength':[np.sqrt(vor_t[:,0]*vor_t[:,0]+vor_t[:,1]*vor_t[:,1]+vor_t[:,2]*vor_t[:,2]) for vor_t in vorticity],
                 }
    variables_lst =  [f"\"{var}\"" for var in data_dict.keys()]
    variables_str = ' '.join(variables_lst)
    
    # datfilename = filename.replace(filename.split('.')[-1],"dat")
    datfilepath = f"{directry}/{filename}.dat"
    with open(datfilepath,'w') as f:
        f.write('''TITLE     = "PAWAN"''')
        f.write("\n")        
        f.write('''VARIABLES = '''+variables_str)
        f.write("\n")
        for tidx,(t, nPlst_t, nPmax_t) in enumerate(zip(time,nParticles,nParticlesMax)):
            # if tidx>100: continue
            # zoneT=f'\"{t}'
            f.write(f'ZONE T=\"{t}\"')
            f.write("\n")
            f.write(f"STRANDID=1 SOLUTIONTIME={t}")
            f.write("\n")
            f.write(f"I={sum(nPlst_t)}, J=1, K=1")
            f.write("\n")
            for nwake,nPi_t in enumerate(nPlst_t):
                for pidx in range(nPi_t):
                    for var in data_dict.keys():
                        f.write(f"{data_dict[var][tidx][pidx + nwake*nPmax_t]:20.05f} ")
                    f.write("\n")    
    print(".dat file created")
    return datfilepath
   
if __name__ == "__main__":     
    directry = "../../data"
    filename = "temp"
    wake = rw.readWake(f"{directry}/{filename}")

    # gen_plt(directry,filename,wake)
