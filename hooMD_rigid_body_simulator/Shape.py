import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as Rot
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
import string
from HelperFunctions import *


class Shape():
    """
    Handles the rigid-body shape so that in can be properly placed into the initial
    configuration of the simulation. Reads the shape gsd and diagonalizes the moment
    of inertia tensor if it isn't since hoomd assumes the unit orientation to be
    when moi is diagonalized

    Similar to MoleculeTemplate class in mcsim code.
    Any necessary info about the shape that is independent of the actual simulation box

    positions include the central particle at first index
    typeids are always 0 for central 1 for the constituents
    """
    def __init__(self,filename):
        self.positions = []
        self.typeids = []
        self.filename = filename
        self.read_shape(filename,0)


    def set_positions(self):
        """
        centers the shape to origin
        Adds the central particle
        """
        n_type = len(self.types) - 1
        alphabet_string = string.ascii_uppercase
        alphabet_list = list(alphabet_string)
        self.types = alphabet_list[:n_type]

        self.positions = self.positions - np.average(self.positions,axis=0)
        new_pos = np.zeros((len(self.positions)+1,3))
        new_pos[1:,:] = self.positions
        self.positions = new_pos
        self.typeid = np.ones(len(self.positions),dtype=int)
        self.typeid[0] = 0

        self.confirmMOI()

    def confirmMOI(self):
        """
        Not only confirms moi but sets it and sets masses etc.
        I don't rotate my shapes to diagonalize my MOI anymore. That should
        be done when the shape is made. Here I just make sure that it is diagonal.
        """
        matrix_moi = getMOI(self.positions[1:])

        ### Check ###
        sum_all = np.sum(np.abs(matrix_moi))
        sum_diag = np.abs(matrix_moi[0,0]) + np.abs(matrix_moi[1,1]) + np.abs(matrix_moi[2,2])
        if((sum_all - sum_diag) > 0.001 ):
            print("Error 223-OP")
            exit()

        moi = np.diag(matrix_moi)
        N_beads = len(self.positions)
        self.mass = np.ones(len(self.positions),dtype=float)/N_beads
        self.mass[0] = 1.0
        self.MOI = np.zeros_like(self.positions)
        self.MOI[0,:] = np.copy(moi)/N_beads
        print("moi, ",moi/N_beads)
        self.body = np.zeros_like(self.typeid,dtype=int)
        a = np.array([1.0,0.0,0.0,0.0])
        a = np.tile(a,(len(self.positions),1))
        self.orientation = a


    def getMOI2(self):
        return np.copy(self.diagonalMOI)

    def getPositions(self):
        return np.copy(self.positions)
    def getTypeids(self):
        return np.copy(self.typeid)
    def getmass(self):
        return np.copy(self.mass)
    def getbody(self):
        return np.copy(self.body)
    def getMOI(self):
        return np.copy(self.MOI)
    def getorientation(self):
        return np.copy(self.orientation)

    def read_shape(self,input,target_frame):
        print(input)
        # exit()
        try:
            with gsd.hoomd.open(name=input, mode='r') as f:
                if (target_frame==-1):
                    #frame = f.read_frame(len(f)-1)
                    frame = f[len(f)-1]
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    #frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                    frame=f[int(target_frame)]
                self.positions = (frame.particles.position).copy()
                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

        except:
            self.positions = (input.particles.position).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
