import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict,deque
import os.path
import networkx as nx
import sys
import itertools
import time
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import freud
from scipy.spatial import distance
from textwrap import wrap
import string
from scipy.spatial.transform import Rotation as Rot

class Utils():
    """
    used in cluster deformation simulations
    """

    def __init__(self,input,frame):
        # print("Reading in : " + input )

        self.read_system(input,frame)

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    # print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    # print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.bodi = (frame.particles.body).copy()
                self.moment_inertia = (frame.particles.moment_inertia).copy()
                self.orientations = (frame.particles.orientation).copy()
                self.mass = (frame.particles.mass).copy()
                self.angmom = (frame.particles.angmom).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]
                self.box = frame.configuration.box

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.bodi = (input.particles.body).copy()
            self.moment_inertia = (input.particles.moment_inertia).copy()
            self.orientations = (input.particles.orientation).copy()
            self.mass = (input.particles.mass).copy()
            self.angmom = (input.particles.angmom).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx

        """
        snapshot.particles.body is broken, non body particles should have -1
        but the container I think only holds unsigned ints so -1 defaults to
        4294967295 which is very inconvenient for the rest oof the class
        so here I swithc it back to -1
        """
        self.body = np.array(self.bodi,dtype=int)
        self.body[self.body>9999999]= -1.0

    def get_N_shape(self):
        unique_bodies,count = np.unique(self.body,return_counts=True)
        unique_bodies = unique_bodies[unique_bodies>-0.5]
        N_shape = len(unique_bodies)
        return N_shape

    def get_Nbead(self):
        beads = self.typeid[self.typeid==1]
        N = len(beads)
        return N

    def calculate_Rg(self):
        """
        Dont consider the last one

        There are 2 definitions for Rg (over all pairs or using the c of mass)
        I will check both give the smae results (YES)
        """
        pos_center = self.positions[self.typeid==0]
        pos_center = pos_center[:-1]
        com = np.average(pos_center,axis=0)
        pos_center = pos_center - com
        d2 = np.linalg.norm(pos_center,axis=1)
        d2 = d2*d2
        d2 = np.sum(d2)
        Rg1 = np.sqrt(d2/len(pos_center))

        N = len(pos_center)
        dd2 = 0.0
        for i in range(N):
            for j in range(N):
                dd2 += np.linalg.norm(pos_center[i]-pos_center[j])*np.linalg.norm(pos_center[i]-pos_center[j])
        dd2 = dd2/(2.0*N*N)
        Rg2 = np.sqrt(dd2)

        if(np.abs(Rg2-Rg1)>0.1):
            print("Error lf09")
            exit()

        return Rg2



    def dump_snap(self,name):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.velocity  = self.velocities[:]
        snap.particles.body = self.body[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid  = self.typeid[:]
        snap.particles.moment_inertia = self.moment_inertia[:]
        snap.particles.orientation = self.orientations[:]
        snap.particles.mass = self.mass[:]
        snap.particles.angmom = self.angmom[:]

        with gsd.hoomd.open(name=name, mode='wb') as f:
            f.append(snap)
