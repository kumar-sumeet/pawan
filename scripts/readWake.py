"""@readWake docstring
Documentation for readWakeFile package
@author Puneet Singh
@date 04/14/2021
"""
import struct
import numpy as np
import argparse as ap


class readWake:
    """ Documentation for readWake
    Wake file includes:
        double  time
        size_t  _numParticles
        matrix  _position 
        matrix  _vorticity
        vector  _radius
        The following have been removed:
        vector  _volume
        vector  _birthstrength
    """
    def __init__(self,wakefilepath):
        ''" Constructor """
        with open(wakefilepath, mode='rb') as file:
            fileContent = file.read()
        self.fileSize = len(fileContent)
        self.time = []
        self.nParticlesMax = []
        self.nParticles = []
        self.position = []
        self.vorticity = []
        self.radius = []
        self.active = []
        self.volume = []
        self.birthstrength = []
        stepnum=0
        datasize_onetimestep = 1
        while len(fileContent)>=datasize_onetimestep: #assumes that atleast one time step worth of data is in the wake binary file
            [t,nmax,n,p,q,r,a,v,b,fileContent] = self.getAllWakeState(fileContent)
            if stepnum ==0: 
                datasize_onetimestep = self.fileSize - len(fileContent)
            self.time.append(t)
            self.nParticlesMax.append(nmax)
            self.nParticles.append(n)
            self.position.append(p)
            self.vorticity.append(q)
            self.radius.append(r)
            self.active.append(a)
            self.volume.append(v)
            self.birthstrength.append(b)
            stepnum=stepnum+1
            print('\r Reading ', wakefilepath.split('/')[-1],': ',int(100*(self.fileSize - len(fileContent))/self.fileSize),'% COMPLETE', end='')
            # if stepnum==700:
            #     break
        if len(self.time):
            self.nTimesteps = len(self.time)
        else:
            print(f"File at {wakefilepath} is empty")
        
    def getInteger(self,fileContent):
        """ getInteger returns an integer 
        """
        ans = struct.unpack("i", fileContent[:4])[0]
        fileContent = fileContent[4:]
        return [ans, fileContent]
    
    def getDouble(self,fileContent):
        """ getDouble returns a double 
        """
        ans = struct.unpack("d", fileContent[:8])[0]
        fileContent = fileContent[8:]
        return [ans, fileContent]
    
    def getVector(self,fileContent,n):
        """ getVector returns a double vector of length n
        """
        ans = np.array(struct.unpack("d"*n,fileContent[:8*n]));
        fileContent = fileContent[8*n:]
        return [ans, fileContent]

    def getMatrix(self,fileContent,m,n=3):
        """ getMatrix returns a double matrix with m rows and n columns
        """
        ans = np.array(struct.unpack("d"*m*n,fileContent[:8*m*n])).reshape((m,n))
        fileContent = fileContent[8*m*n:]
        return [ans, fileContent]
    
    def getAllWakeState(self, fileContent):
        """ getAllWakeState returns wake data from file
        """
        [time, fileContent] = self.getDouble(fileContent)
        [nWake, fileContent] = self.getInteger(fileContent)
        fileContent = fileContent[4:]   # Skipping 4 bit blank space
        [nParticlesMax, fileContent] = self.getInteger(fileContent)
        fileContent = fileContent[4:]   # Skipping 4 bit blank space
        nParticles = [] 
        position = []
        vorticity = []
        radius = []
        active = []
        volume = []
        birthstrength = []
        for n in range(nWake):
            [npar,pos,vort,rad,act,vol,bs,fileContent] = self.getWakeState(fileContent, nParticlesMax)
            # nParticles = nParticles + npar
            nParticles.append(npar)
            position.append(pos)
            vorticity.append(vort)
            radius.append(rad)
            active.append(act)
            volume.append(vol)
            birthstrength.append(bs)
        position = np.concatenate(position)
        vorticity = np.concatenate(vorticity)
        radius = np.concatenate(radius)
        active = np.concatenate(active)
        volume = np.concatenate(volume)
        birthstrength = np.concatenate(birthstrength)
        return [time, nParticlesMax, nParticles, position, vorticity, radius, active, volume, birthstrength, fileContent]

    def getWakeState(self, fileContent, nParticlesMax):
        """ getWakeState returns wake data from file
        """
        [nParticles, fileContent] = self.getInteger(fileContent)
        fileContent = fileContent[4:]   # Skipping 4 bit blank space
        [position, fileContent] = self.getMatrix(fileContent,nParticlesMax)
        [vorticity, fileContent] = self.getMatrix(fileContent,nParticlesMax)
        [radius, fileContent] = self.getVector(fileContent,nParticlesMax)
        [active, fileContent] = self.getVector(fileContent,nParticlesMax)
        [volume, fileContent] = self.getVector(fileContent,nParticlesMax)
        [birthstrength, fileContent] = self.getVector(fileContent,nParticlesMax)
        return [nParticles, position, vorticity, radius, active, volume, birthstrength, fileContent]

    def printData(self):
        """ printData prints wake properties
        """
        print(self.nTimesteps)
        print(self.time)
        print(self.nParticles)
        print(self.position)
        print(self.vorticity)
        print(self.radius)
        # print(self.volume)
        # print(self.birthstrength)

    def writeTextFile(self,fname='temp.dat'):
        """ writeTextFile Writes wake data into a text file
        """
        with open(fname,"w") as f:
            f.write(r"Total number of timesteps = " + str(self.nTimesteps) + "\n")
            for n in range(self.nTimesteps):
                f.write(r"Timestep #" + str(n) + "\n")
                f.write(r"Time = " + str(self.time[n]) + " seconds\n" )
                f.write("x\ty\tz\tax\tay\taz\n")
                np.savetxt(f,np.hstack((self.position[n],self.vorticity[n])), delimiter='\t',newline='\n',header='', footer='', fmt="%0.6e")

if __name__ == "__main__":
    # parser = ap.ArgumentParser()
    # parser.add_argument("filename", nargs='?', default="../data/temp.wake", help=".wake file")
    # args = parser.parse_args()
    # rw = readWake(args.filename)
    # # rw.printData()
    # rw.writeTextFile("../data/temp.dat")
    directry = "../data"
    filename = "temp.wake"
    wake = readWake(f"{directry}/{filename}")


