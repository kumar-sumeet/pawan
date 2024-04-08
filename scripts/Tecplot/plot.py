#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:28:18 2023

@author: sk
"""
import tecplot as tp
from tecplot.constant import *

def scatter(pltfilepath):
    dataset = tp.data.load_tecplot(pltfilepath)

    frame = tp.active_frame()
    frame.plot_type = PlotType.Cartesian3D
    plot = frame.plot()
    plot.contour(0).variable = dataset.variable('vor_x')
    plot.show_scatter = True

    plot.scatter.variable = dataset.variable('strength')
    
    for z in dataset.zones():
        scatter = plot.fieldmap(z).scatter
        scatter.symbol_type = SymbolType.Geometry
        scatter.symbol().shape = GeomShape.Sphere
        scatter.fill_mode = FillMode.UseSpecificColor
        scatter.fill_color = plot.contour(0)
        scatter.color = plot.contour(0)
        scatter.size_by_variable = True
    
    frame.add_text('Size of dots indicate relative pressure', (20, 80))
    
    # ensure consistent output between interactive (connected) and batch
    plot.contour(0).levels.reset_to_nice()
    
    # tp.export.save_png('scatter.png')