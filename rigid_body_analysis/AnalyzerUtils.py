import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from scipy.spatial import distance
np.set_printoptions(suppress=True,precision=5,linewidth=150)

def quaternion_multiplication(a,b):
    """
    Is not commutative
    (a.s*b.s - dot(a.v, b.v),  a.s*b.v + b.s * a.v + cross(a.v,b.v));
    Scalar part : a.s*b.s - dot(a.v, b.v)
    Vector part : a.s*b.v + b.s * a.v + cross(a.v,b.v)
    """
    res = np.zeros_like(a)

    s1 = a[:,0]*b[:,0]
    s2 = -np.sum(a[:,1:]*b[:,1:],axis=1)
    scalar = s1+s2

    v1 = a[:,0].reshape(-1,1)*b[:,1:]
    v2 = b[:,0].reshape(-1,1)*a[:,1:]
    v3 = np.cross(a[:,1:],b[:,1:])
    v = v1 + v2 + v3

    res[:,0] = scalar
    res[:,1:] = v

    return res

def ForceMag_HPHC(r,pwr,A,b,c):
    cutoff_right = (-np.pi+b*c)/c + np.pi*2/c
    force = -c*A*np.sin(c*(r-b))
    force[r<b] = 0.0
    force2 = pwr*np.power(b-r,pwr-1.0)
    force2[r>b] = 0.0
    force = force + force2
    force[r>cutoff_right] = 0.0
    return force

def get_theta_phi(v):
    """
    basically converts cartesian to spherical and returns theta and phi
    """
    r = np.linalg.norm(v)
    teta = np.arccos(v[2]/r)
    # phi = np.sign(v[1])*np.arccos(v[0]/(v[0]**2+v[1]**2))
    phi = np.arctan2(v[1],v[0])
    return teta,phi

def in_deg(x):
    return (180.0*x)/np.pi

def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta


def com(a,Box):
    theta = np.divide(a + 0.5 * Box, Box)*np.multiply(2,np.pi)
    xi_average = np.average(np.cos(theta), axis = 0)
    zeta_average = np.average(np.sin(theta), axis = 0)
    theta_average = np.arctan2(-zeta_average,-xi_average) + np.pi
    com = np.multiply(Box,theta_average)/np.multiply(2,np.pi)-0.5 * Box
    return com

def rotZ(teta):
    """returns a rotation matrix that rotates around z axis"""
    c = np.cos(teta)
    s = np.sin(teta)
    mat = np.array([
    [c,-s,0.0],
    [s,c,0.0],
    [0.0,0.0,1.0]
    ])
    return mat


class Analyzer():

    """
    April 17 23
    """

    def __init__(self,input,frame):
        self.read_system(input,frame)
        self.mass = 1.0
        self.noise_id = 0

    def getNshapes(self):
        return len(self.positions[self.typeid==0])

    def getCentralPos(self):
        return self.positions[self.typeid==0]
    def getCentralImages(self):
        return self.images[self.typeid==0]

    def getOrientations(self):
        return self.orientations[self.typeid==0]

    def getBoxSize(self):
        return self.Lx

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        self.target_frame = target_frame
        self.frame = 0
        try:
            with gsd.hoomd.open(name=input, mode='r') as f:
                if (target_frame==-1):
                    # frame = f.read_frame(len(f)-1)
                    frame = f[len(f)-1]
                    self.frame = len(f)-1
                else:
                    self.frame = target_frame
                    frame=f[int(target_frame)]
                    # frame = f.read_frame(target_frame)
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
        self.N_dof = (len(self.positions[self.typeid==0]) - 1)*3


    def fee(self):
        pass
