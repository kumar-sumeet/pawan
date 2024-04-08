#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:28:30 2022

@author: Sumeet Kumar
"""

import pickle
import numpy as np
import tecplot as tp
import readWake as rw
from Tecplot import utils as tecutils
from Tecplot import plot
from tecplot.constant import *
import subprocess
from shutil import copy2
import os 

def show_wing1():
    wopwop_datadir = '../../Python/HeliNoise/Data/Diss_runs'
    periodicOraperiodic = 'aperiodic'
    dirstosave_plt = [
            f'{wopwop_datadir}/1_simple_{periodicOraperiodic}/{filename}/{filename}',
            # f'{wopwop_datadir}/2_simple_hemisphere_{periodicOraperiodic}/{filename}/{filename}',
            # f'{wopwop_datadir}/3_simple_hemisphere_{periodicOraperiodic}ARCTIS/{filename}/{filename}',
            # f'{wopwop_datadir}/Plane_50_extended/{filename}/{filename}',
            # f'{wopwop_datadir}/Hemisphere_50/{filename}/{filename}',
            # f'{wopwop_datadir}/SingleObserver_50/below/{filename}/{filename}',
            # f'{wopwop_datadir}/SingleObserver_50/inplane/{filename}/{filename}',
            # f'{wopwop_datadir}/SingleObserver_50/outofplane/{filename}/{filename}'
                      ]
    tecutils.saveplttodirs(dirstosave_plt,pltfilepath)

    dataset_geometry = tp.data.load_tecplot(f"{dirstosave_plt[0]}/{filename}_DymInertial_Blade_1_surfaceload.plt", 
                                            read_data_option = ReadDataOption.Append)
    return dataset_geometry

def show_wing2(data_geom):
    tp.active_frame().plot().fieldmap(1).scatter.show=False
    plot.contour(1).variable = data_geom.variable('Fz')

    #from pytecplot record script
    tp.active_frame().plot().rgb_coloring.red_variable_index=6
    tp.active_frame().plot().rgb_coloring.green_variable_index=3
    tp.active_frame().plot().rgb_coloring.blue_variable_index=3
    tp.active_frame().plot().contour(1).variable_index=3
    tp.active_frame().plot().contour(2).variable_index=4
    tp.active_frame().plot().contour(3).variable_index=5
    tp.active_frame().plot().contour(4).variable_index=6
    tp.active_frame().plot().contour(5).variable_index=7
    tp.active_frame().plot().contour(6).variable_index=9
    tp.active_frame().plot().contour(7).variable_index=10
    tp.active_frame().plot().show_contour=True
    tp.active_frame().plot().contour(1).variable_index=14
    tp.active_frame().plot().fieldmap(1).contour.flood_contour_group_index=1
    tp.macro.execute_command('$!Pick DeselectAll')
    tp.macro.execute_command('''$!Pick AddAllInRect
      SelectText = Yes
      SelectGeoms = Yes
      SelectZones = Yes
      ConsiderStyle = Yes
      X1 = 6.62846347607
      X2 = 8.50251889169
      Y1 = 3.80667506297
      Y2 = 4.95528967254''')
    tp.macro.execute_command('''$!Pick AddAtPosition
      X = 8.97607052897
      Y = 3.89735516373
      ConsiderStyle = Yes''')
    tp.macro.execute_command('''$!Pick Shift
      X = -6.99244332494
      Y = 2.22670025189''')
show_geom = True
show_geom = False

print('running...')
directry = "../data"
runfiles = [
            # "wing_rectangularCOARSE_staticPawan"
            # "wing_rectangularCOARSE_pitchingPawan"
            # "wing_rectangularCOARSE_statictbllkpPawan"
            # "wing_rectangular_staticPawan"            

            # "wing_ellipticalCOARSE_staticPawan"
            # "wing_ellipticalCOARSE_pitchingPawan"
            # "wing_ellipticalCOARSE_statictbllkpPawan"
            # "wing_elliptical_pitchingPawan"

            # "1994_Run15_5_hover_rigidrunupPawan"
                # "1994_Run15_5_hover_rigidshaftrigidrotorrunupPawan"
                # "1994_Run42_7_baselinePetersUnsteady_elasticshaftelasticrotorrunupPawan",
                "1994_Run26_11_baselinePetersUnsteady_elasticshaftelasticrotorrunupPawan",
                # "1994_Run57_7_baselinePetersUnsteady_elasticshaftelasticrotorrunupPawan",
            # "temp",
            # "vring_4by80"
            # "vring_5by100"
            # "vring_6by117"
            # "vring4by80_1and2_fusion"
            # "vring4by80_1and2_fusionrelaxed"
            # "vring4by80_1and2_fissionfusion"
            # "vring5by100_1and2_fusion"
            # "vring5by100_1and2_fusionrelaxed"

            ]
filename = runfiles[0]
preplot_path = '/HTOpt/TecPlot/2019R1/360ex_2019r1/bin/preplot' #path to preplot installation
wakefilename = filename+"Wake"
datfilepath = f"{directry}/{wakefilename}.dat"
wake = rw.readWake(f"{directry}/{wakefilename}.wake")
print('read wake...')
datfilepath = tecutils.gen_dat(directry,wakefilename,wake) 
subprocess.run([preplot_path, datfilepath])
pltfilepath = datfilepath.replace(".dat",".plt")

tp.session.connect()
tp.new_layout()
dataset_wake = tp.data.load_tecplot(pltfilepath)
if show_geom: 
    data_geom = show_wing1()

frame = tp.active_frame()
frame.plot_type = PlotType.Cartesian3D
plot = frame.plot()
plot.contour(0).variable = dataset_wake.variable('Vor_strength')
plot.contour(0).colormap_name='Small Rainbow'
plot.show_scatter = True
plot.scatter.variable = dataset_wake.variable('radius')
# plot.scatter.relative_size_units = RelativeSizeUnits.Grid

# plot.vector.u_variable = dataset_wake.variable('vor_x')
# plot.vector.v_variable = dataset_wake.variable('vor_y')
# plot.vector.w_variable = dataset_wake.variable('vor_z')
# plot.show_vector = True
# plot.vector.use_relative = False
# plot.vector.length = 5


if show_geom:
    show_wing2(data_geom)

for z in dataset_wake.zones():
    scatter = plot.fieldmap(z).scatter
    scatter.symbol_type = SymbolType.Geometry
    scatter.symbol().shape = GeomShape.Sphere
    scatter.fill_mode = FillMode.UseSpecificColor
    scatter.fill_color = plot.contour(0)
    scatter.color = plot.contour(0)
#     scatter.size_by_variable = False
# for z in dataset_wake.zones():
    scatter.size_by_variable = True
plot.scatter.relative_size = 0.25
# plot.scatter.relative_size = 1.0

tp.macro.execute_command('''$!Rotate3DView X
  Angle = 180
  RotateOriginLocation = DefinedOrigin''')
tp.macro.execute_command('''$!Rotate3DView Z
  Angle = 180
  RotateOriginLocation = DefinedOrigin''')
# tp.active_frame().plot().contour(0).levels.reset_levels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])#, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
tp.active_frame().plot().contour(0).levels.reset_levels([0,0.5, 1,1.5, 2, 2.5, 3,])
tp.active_frame().plot().contour(0).levels.reset_to_nice()
tp.active_frame().plot(PlotType.Cartesian3D).vector.u_variable_index=3
tp.active_frame().plot(PlotType.Cartesian3D).vector.v_variable_index=4
tp.active_frame().plot(PlotType.Cartesian3D).vector.w_variable_index=5
tp.active_frame().plot().show_vector=True
tp.active_frame().plot(PlotType.Cartesian3D).vector.relative_length=1

tp.active_frame().plot().axes.z_axis.scale_factor=1