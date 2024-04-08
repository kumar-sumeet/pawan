"""@plotWake docstring
Documentation for plotWake package
@author Puneet Singh
@date 04/15/2021
"""
import struct
import numpy as np
import readWake as rw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse as ap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

class plotWake:
    """ Documentation for plotWake
    """
    
    def __init__(self,fileName):
        ''" Constructor """
        data = rw.readWake(fileName)
        self.nTimesteps = data.nTimesteps
        self.time = data.time
        self.nParticles = data.nParticles
        self.position = data.position
        self.vorticity = data.vorticity
        self.radius = data.radius
        self.active = data.active
        # self.volume = data.volume
        # self.birthstrength = data.birthstrength
        
        self.position_arr = np.asarray(self.position)        
        multiplier = 1.25
        X = multiplier
        xmax = X*max(np.amax(self.position_arr[:,:,0]),np.amax(self.position_arr[:,:,1]))
        xmin = X*min(np.amin(self.position_arr[:,:,0]),np.amin(self.position_arr[:,:,1]))
        ymax = X*xmax
        ymin = X*xmin
        zmax = X*np.amax(self.position_arr[:,:,2])
        zmin = X*np.amin(self.position_arr[:,:,2])
        self.plot_limits = {'x':[xmax,xmin],'y':[ymax,ymin],'z':[zmax,zmin]}
        
    def getTimeStep(self,n=0):
        """ 
        getTimeStep Loads particle position and vorticity at nth time step 
        
        Parameters
        ----------
        n: int
            Time step index
        """
        #only get 'active' particles
        self.x =  self.position[n][np.nonzero(self.active[n]),0] 
        self.y =  self.position[n][np.nonzero(self.active[n]),1]
        self.z =  self.position[n][np.nonzero(self.active[n]),2]
        self.p = self.vorticity[n][np.nonzero(self.active[n]),0]
        self.q = self.vorticity[n][np.nonzero(self.active[n]),1]
        self.r = self.vorticity[n][np.nonzero(self.active[n]),2]
        self.x=   np.squeeze(self.x.T)
        self.y=   np.squeeze(self.y.T)
        self.z=   np.squeeze(self.z.T)
        self.p=   np.squeeze(self.p.T)
        self.q=   np.squeeze(self.q.T)
        self.r=   np.squeeze(self.r.T)
        self.strength = np.sqrt(self.p**2+self.q**2+self.r**2)
        self.strength = np.where(self.strength>25, 25, self.strength) #to visualise without particles blowing up
        self.maxstrength = max(self.strength)
        self.strengthcolors = self.strength/np.sum(self.strength)
    
    def plot3DQuiver(self,n=0):
        """ 
        plot3DQuiver Plots the vortex particles as colored arrows 
        
        Parameters
        ----------
        n: int
            Time step index
        """
        self.getTimeStep(n)
        y = self.setColors()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x,self.y,self.z,color=y)
        ax.quiver(self.x,self.y,self.z,self.p,self.q,self.r,length=0.1,color=y,normalize=True)
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([-1,3])
        plt.show()

    def plotAnimate3View(self,savefile=""):
        """ 
        plotAnimate3View Plots 3 View of vortex particles as colored spheres 
        """
        fig = plt.figure(figsize=(12,10))
        gs = GridSpec(2,2,figure = fig)
        axTopView = fig.add_subplot(gs[0,0])
        axLeftView = fig.add_subplot(gs[1,0])
        axFrontView = fig.add_subplot(gs[1,1])
        self.getTimeStep(0)
        y = self.setColors()
        parSize = 20
        self.top = axTopView.scatter(self.x,self.y,s=parSize,color=y)
        self.left = axLeftView.scatter(self.x,self.z,s=parSize,color=y)
        self.front = axFrontView.scatter(self.y,self.z,s=parSize,color=y)
        
        def setPlot(ax,x,y): 
            ax.set_xlim([-2,2])
            ax.set_ylim([-2,2])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        setPlot(axTopView,'x','y')
        setPlot(axLeftView,'x','z')
        setPlot(axFrontView,'y','z')

        # time_template = 'Time: %0.2f s'
        # time_text = ax.text2D(0.05,0.95,' ', transform=ax.transAxes)
        
        def update(num):
            self.getTimeStep(n=num)
            y = self.setColors()
            self.top.remove()
            self.left.remove()
            self.front.remove()
            self.top = axTopView.scatter(self.x,self.y,s=parSize,color=y)
            self.left = axLeftView.scatter(self.x,self.z,s=parSize,color=y)
            self.front = axFrontView.scatter(self.y,self.z,s=parSize,color=y)
        ani = animation.FuncAnimation( fig, update, self.nTimesteps, interval = 200)
        plt.show()
        if savefile:
            ani.save(savefile,writer='ffmpeg',fps=2, bitrate = 1000)

    def plotAnimate(self,savefile="",quiver=True):
        """ 
        plotAnimate Plots the vortex particles as colored spheres 
        """
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        self.getTimeStep(0)
        self.scplot = ax.scatter(self.x,self.y,self.z,s=0.1)
        # if quiver:  self.qvplot = ax.quiver(self.x,self.y,self.z,self.p,self.q,self.r,length=0.2)
        # ax.set_xlim(self.plot_limits['x'])
        # ax.set_ylim(self.plot_limits['y'])
        # ax.set_zlim(self.plot_limits['z'])
        ax.set_xlim([-20,20])
        ax.set_ylim([-2,2])
        ax.set_zlim([-20,20])
        # ax.set_xlim([-0.5,2.0])
        # ax.set_ylim([-0.5,2.0])
        # ax.set_zlim([-0.5,2.0])
        ax.view_init(elev=25., azim=55)   
        ax.invert_zaxis()
        ax.invert_xaxis()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        time_template = 'Time: %0.2f s'
        time_text = ax.text2D(0.05,0.95,' ', transform=ax.transAxes)
        
        def update(num):
            print('time step =', num)
            self.getTimeStep(n=num)
            y = self.setColors()
            # if quiver: 
            #     self.qvplot.remove()
            #     self.qvplot = ax.quiver(self.x,self.y,self.z,self.p,self.q,self.r,length=0.1,color=y,normalize=True)
            self.scplot.remove()
            self.scplot = ax.scatter(self.x,self.y,self.z,color=y,s=0.1)
            time_text.set_text(time_template % self.time[num])
        ani = animation.FuncAnimation( fig, update, self.nTimesteps, interval = 200)
        plt.show()
        if savefile:
            ani.save(savefile,writer='ffmpeg',fps=2, bitrate = 1000)

        return ax 
    
    def setColors(self):
        norm = matplotlib.colors.Normalize()
        norm.autoscale(self.strengthcolors)
        cm = matplotlib.cm.viridis
        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        y = cm(norm(self.strengthcolors))
        return y


    def printData(self):
        """ printData prints wake properties
        """
        print(self.nParticles)
        print(self.position)
        print(self.vorticity)
        print(self.radius)
        print(self.active)
        # print(self.volume)
        # print(self.birthstrength)

if __name__ == "__main__":
    # parser = ap.ArgumentParser()
    # parser.add_argument("filename", nargs='?', default="../data/temp.wake", help=".wake file")
    # parser.add_argument("-a","--animate", help="Show animation", action="store_true")
    # parser.add_argument("-a3","--animate3view", help="Show 3 view animation", action="store_true")
    # parser.add_argument("-s","--save",default=1, help="Save animation file", action="store_true")
    # parser.add_argument("-q","--quiver", type=int, default=-1, help="Show quiver plot at time step number ")
    # args = parser.parse_args()
    # pw = plotWake(args.filename)
    # savefile = ""
    # if args.save:
    #     savefile = args.filename.replace(".wake",".mp4")
    # if args.animate:
    #     print('Plotting animation (Isometric 3D view)')
    #     pw.plotAnimate(savefile)
    # if args.animate3view:
    #     print('Plotting animation (3 View)')
    #     pw.plotAnimate3View(savefile)
    # if args.quiver>=0:
    #     print('Plotting static figure at time step ' + str(args.quiver))
    #     pw.plot3DQuiver(args.quiver)

    filename = '../data/temp.wake'
    pw = plotWake(filename)
    savefile = filename.replace(".wake",".mp4")
    pw.plotAnimate(savefile,quiver=False)
    
    