import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict
import os.path
import networkx as nx
import itertools
import time
from sklearn.cluster import DBSCAN, OPTICS, MeanShift, estimate_bandwidth
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import freud
from scipy.spatial import distance
from textwrap import wrap
import cupy as cp
np.set_printoptions(suppress=True,precision=5,linewidth=150)

"""
VII : Very Important Info
"""
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

def WCA(r,eps,sigma):
    term = np.power(sigma/r,6)
    x = 4.0*eps*(np.power(term,2)-term + 0.25)
    cutoff = sigma * np.power(2.0,1.0/6.0)
    x[r>cutoff] = 0.0
    x[x>1000] = 1000.0
    x = np.nan_to_num(x, copy=True, nan=1000.0, posinf=1000.0)
    # if(len(r)==0):
    #     x = 0.0
    return x

def HalfPowerHalfCosine(r,pwr,A,b,c):
    cutoff_right = (-np.pi+b*c)/c + np.pi*2/c
    pot = -A*np.cos(c*r-b*c)-A
    pot[r<b] = 0.0
    pot2 = np.power(b-r,pwr) - A*2.0
    pot2[r>b] = 0.0
    pot = pot + pot2
    pot[r>cutoff_right] = 0.0
    return pot

def sample_configuration_silindir01(p1,p2,l,f):
    """
    extract a1,b1,a2,b2,l,frame (all without pbc)
    assume that the p[0] is origin and p[1] is top
    """
    a1 = p1[0]
    a2 = p2[0]
    b1 = p1[1]
    b2 = p2[1]
    config = np.zeros(14)
    config[:3] = a1
    config[3:6] = b1
    config[6:9] = a2
    config[9:12] = b2
    config[12] = l
    config[13] = f
    return config

def sample_configuration_cylinder02(p1,p2,l,f):
    """
    extract a1,b1,a2,b2,l,frame (all without pbc)
    assume that the p[0] is origin and p[15] is top
    """
    a1 = p1[0]
    a2 = p2[0]
    b1 = p1[15]
    b2 = p2[15]
    config = np.zeros(14)
    config[:3] = a1
    config[3:6] = b1
    config[6:9] = a2
    config[9:12] = b2
    config[12] = l
    config[13] = f
    return config

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

class ExtractDataUtils():

    """
    Jan 3 23

    Gets a frame from the simulation with rigid body cubes
    Detects pair of cubes that potentially interact
    Records the quaternions and positions of  pairs of cubes as raw input data
    Calculates forces and torques of pairs of cubes as raw output data
    Force-Torque calculation is costly so it is done on gpu using cupy
    by simply replacing np. functions with cp. 

    """

    def __init__(self,input,frame):
        self.read_system(input,frame)
        self.calculate_beads_per_shape()
        self.noise_id = 0


    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        self.target_frame = target_frame
        self.frame = 0
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
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


    def calculate_beads_per_shape(self):
        """
        Assumes every shape is the same
        There are shape beads only in the shape
        The constituent bead typeid is 1, central particle typeid is 0
        """
        self.Nbead_per_shape = len(self.positions[self.typeid==1])//len(self.positions[self.typeid==0]) + 1
        self.N_shape = len(self.positions[self.typeid==0])

    def detect_pairs(self,cutoff):
        """
        For pairs use rigid body centers, if they're closer than a given distance
        than we have a pair
        For each shape :
        Index 0 is the rigid center
        Index 1 is the top plate center
        so the first two points are enough to define all of them

        """
        self.center_distance = cutoff ### depends on the shape should be an input
        ## 5.6 for the silindir_01.gsd

        pos_tree = self.positions[self.typeid==0] + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        pairs = tree.query_pairs(r=self.center_distance)
        self.pairs_body = np.zeros((len(pairs),2),dtype=np.int32)
        self.Npairs = len(self.pairs_body)
        pair1_body = self.pairs_body[:,0]
        self.pair1_pos = np.zeros((self.Npairs,self.Nbead_per_shape,3),dtype=np.float32)
        self.pair2_pos = np.zeros((self.Npairs,self.Nbead_per_shape,3),dtype=np.float32)

        self.pair1_ori = np.zeros((self.Npairs,self.Nbead_per_shape,4),dtype=np.float32)
        self.pair2_ori = np.zeros((self.Npairs,self.Nbead_per_shape,4),dtype=np.float32)

        self.pairs_body = np.zeros((len(pairs),2),dtype=np.int32)
        for i,pair in enumerate(pairs):
            # self.pairs_body[i,0] = pair[0]*self.Nbead_per_shape
            # self.pairs_body[i,1] = pair[1]*self.Nbead_per_shape
            self.pair1_pos[i] = self.positions[self.body==pair[0]*self.Nbead_per_shape]
            self.pair2_pos[i] = self.positions[self.body==pair[1]*self.Nbead_per_shape]

            self.pair1_ori[i] = self.orientations[self.body==pair[0]*self.Nbead_per_shape]
            self.pair2_ori[i] = self.orientations[self.body==pair[1]*self.Nbead_per_shape]



    def calculate_pairwise_torques(self,indexing):
        """
        See vdw - 31

        can we optimize pbc calculation?
        https://stackoverflow.com/questions/11108869/optimizing-python-distance-calculation-while-accounting-for-periodic-boundary-co
        COM1 (3), COM2 (3), QUAT1 (4), QUAT2 (4), Lx(1), Frame(1) -> 16 columns in total
        """

        """pair configurations (no need for cupy)"""
        self.pair_configurations = np.zeros((len(self.pairs_body),16))
        self.pair_configurations[:,15] = self.target_frame
        self.pair_configurations[:,14] = self.Lx
        self.pair_configurations[:,:3] = self.pair1_pos[:,0,:]
        self.pair_configurations[:,3:6] = self.pair2_pos[:,0,:]
        self.pair_configurations[:,6:10] = self.pair1_ori[:,0,:]
        self.pair_configurations[:,10:14] = self.pair2_ori[:,0,:]

        """
        cupy implementation

        pair1_pos :
        p00 p01 p02
        p00 p01 p02
        ...

        pair2_pos :
        p01 p02 p03
        p11 p12 p13
        ...

        """
        dev1 = cp.cuda.Device(1)
        dev1.use()
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        self.pair1_pos = cp.asarray(self.pair1_pos)
        self.pair2_pos = cp.asarray(self.pair2_pos)
        self.pair2_pos_ = cp.tile(self.pair2_pos,(1,self.Nbead_per_shape,1))
        self.pair1_pos_ = cp.repeat(self.pair1_pos,self.Nbead_per_shape,axis=1)
        # print(self.pair1_pos_)
        # print(self.pair2_pos_)
        # exit()

        delta = self.pair1_pos_ - self.pair2_pos_
        self.pair1_pos_ = 0.0
        self.pair2_pos_ = 0.0
        Box = self.Lx
        mempool.free_all_blocks()
        delta = cp.where(delta > 0.5 * Box, delta- Box, delta)
        delta = cp.where(delta <- 0.5 * Box, Box + delta, delta)
        dists = cp.linalg.norm(delta,axis=2)

        """
        Energy Calculation
        """
        pw = 2.5
        A = 0.00035
        b = 1.46375
        c = 3.3833
        bc = b*c
        depth = 2.0*A
        cutoff_right = (-np.pi+bc)/c + np.pi*2/c
        cutoff_left = b
        pot = -A*cp.cos(c*dists-bc)-A
        pot[dists<cutoff_left] = 0.0
        pot2 = cp.power(-dists+cutoff_left,pw) - depth
        pot2[dists>cutoff_left] = 0.0
        pot = pot + pot2
        pot[dists>cutoff_right] = 0.0
        pair_energies = cp.sum(pot,axis=1).get()
        pair_energies = np.expand_dims(pair_energies,axis=1)
        mempool.free_all_blocks()
        del(pot,pot2)
        mempool.free_all_blocks()

        """
        Force Calculation

        cutoff_right = (-np.pi+b*c)/c + np.pi*2/c
        force = -c*A*np.sin(c*(r-b))
        force[r<b] = 0.0
        force2 = pwr*np.power(b-r,pwr-1.0)
        force2[r>b] = 0.0
        force = force + force2
        force[r>cutoff_right] = 0.0
        """
        cutoff_right = (-np.pi+bc)/c + np.pi*2/c
        cutoff_left = b
        force = -c*A*cp.sin(c*dists-bc)
        force[dists<cutoff_left] = 0.0
        force2 = pw*cp.power(-dists+cutoff_left,pw-1.0)
        force2[dists>cutoff_left] = 0.0
        force = force + force2
        force[dists>cutoff_right] = 0.0
        Np,Ncomb = force.shape
        force = force.reshape(Np,Ncomb,1)
        dists = dists.reshape(Np,Ncomb,1)
        delta = cp.multiply(delta,force)
        delta = cp.divide(delta,dists)
        del(force,dists)
        mempool.free_all_blocks()
        """ on 1 """
        delta_on_1 = delta.reshape(Np,self.Nbead_per_shape,self.Nbead_per_shape,3)
        net_force_vectors_on_1 = cp.sum(delta_on_1,axis=2)

        """
        on 2
        indexing is done in the main script
        """
        # k = self.Nbead_per_shape
        # indexing = np.zeros(k*k,dtype=np.int32)
        # for jj in range(k):
        #     indexing[jj*k:(jj+1)*k] = np.arange(0,k*k,k,dtype=np.int32)+jj
        delta = -delta[:,indexing,:]
        delta_on_2 = delta.reshape(Np,self.Nbead_per_shape,self.Nbead_per_shape,3)
        net_force_vectors_on_2 = cp.sum(delta_on_2,axis=2)

        """
        Translation force is simply the sum of all forces on every bead
        Torque is the sum of the cross products (rxf) on all beads
        """

        total_force_vectors_on_1 = cp.sum(net_force_vectors_on_1,axis=1).get()
        total_force_vectors_on_2 = cp.sum(net_force_vectors_on_2,axis=1).get()

        self.pair1_pos_COM = cp.asarray(self.pair1_pos)
        self.pos_COMs = cp.copy(self.pair1_pos_COM[:,0,:])
        self.pos_COMs = cp.expand_dims(self.pos_COMs,axis=1)
        self.pos_COMs = cp.repeat(self.pos_COMs,self.Nbead_per_shape,axis=1)
        self.pair1_pos_COM =  self.pair1_pos_COM - self.pos_COMs
        self.pair1_pos_COM = cp.where(self.pair1_pos_COM > 0.5 * Box, self.pair1_pos_COM- Box, self.pair1_pos_COM)
        self.pair1_pos_COM = cp.where(self.pair1_pos_COM <- 0.5 * Box, Box + self.pair1_pos_COM, self.pair1_pos_COM)

        self.pair2_pos_COM = cp.asarray(self.pair2_pos)
        self.pos2_COMs = cp.copy(self.pair2_pos_COM[:,0,:])
        self.pos2_COMs = cp.expand_dims(self.pos2_COMs,axis=1)
        self.pos2_COMs = cp.repeat(self.pos2_COMs,self.Nbead_per_shape,axis=1)
        self.pair2_pos_COM =  self.pair2_pos_COM - self.pos2_COMs
        self.pair2_pos_COM = cp.where(self.pair2_pos_COM > 0.5 * Box, self.pair2_pos_COM- Box, self.pair2_pos_COM)
        self.pair2_pos_COM = cp.where(self.pair2_pos_COM <- 0.5 * Box, Box + self.pair2_pos_COM, self.pair2_pos_COM)

        ### take the cross product to find the torque vectors
        net_force_vectors_on_1 = net_force_vectors_on_1.reshape(-1,3)
        self.pair1_pos_COM = self.pair1_pos_COM.reshape(-1,3)
        torques_on_1 = cp.cross(self.pair1_pos_COM,net_force_vectors_on_1)
        torques_on_1 = torques_on_1.reshape(self.Npairs,-1,3)
        total_torque_on_1 = cp.sum(torques_on_1,axis=1).get()

        net_force_vectors_on_2 = net_force_vectors_on_2.reshape(-1,3)
        self.pair2_pos_COM = self.pair2_pos_COM.reshape(-1,3)
        torques_on_2 = cp.cross(self.pair2_pos_COM,net_force_vectors_on_2)
        torques_on_2 = torques_on_2.reshape(self.Npairs,-1,3)
        total_torque_on_2 = cp.sum(torques_on_2,axis=1).get()

        force_torque_info = np.hstack((pair_energies,total_torque_on_1,total_torque_on_2,total_force_vectors_on_1))

        return force_torque_info,self.pair_configurations





    def calculate_pairwise_interactions_HPHC(self):
        """
        The other version of the same function (calculate_pairwise_interactions)
        is very slow with the HPHC function and many beads so we need a new method
        other than trees (method2 may be the better choice here since there are many
        beads interacting with each other)

        Method 1 :
        Use two trees for the neighbors and use query_ball_tree (faster x4)

        Method 2 :
        Repeat pos1, tile pos 2 and calculate all the distances

        Method 3 :
        cdist (bunu niye en bastan denemedik ki?)
        We should also sample the pair configuration at this point

        self.pair_configurations = [a1,b1,a2,b2,box_length,frame] (14 floats for each)
        all without pbc, just absolute coordinates

        """
        pos_tree = self.positions + self.Lx*0.5
        self.pair_interactions = np.zeros(len(self.pairs_body))
        self.pair_configurations = np.zeros((len(self.pairs_body),14))
        for i,pair in enumerate(self.pairs_body):
            """method 2 - slower"""
            # p1 = self.positions[self.body==pair[0]]
            # p2 = self.positions[self.body==pair[1]]
            # # pair_config = sample_configuration_silindir01(p1,p2,self.Lx,self.target_frame)
            # pair_config = sample_configuration_cylinder02(p1,p2,self.Lx,self.target_frame)
            # self.pair_configurations[i,:] = pair_config
            # p1 = np.tile(p1,(self.Nbead_per_shape,1))
            # p2 = np.repeat(p2,self.Nbead_per_shape,axis=0)
            # dists = wrap_pbc(p1-p2,self.Lx)
            # dists = np.linalg.norm(dists,axis=1)
            # en = HalfPowerHalfCosine(dists[dists<2.30],2.5,0.00044,1.40,3.5)
            """method 1"""
            p1 = self.positions[self.body==pair[0]]
            p2 = self.positions[self.body==pair[1]]
            pair_config = sample_configuration_cylinder02(p1,p2,self.Lx,self.target_frame)
            self.pair_configurations[i,:] = pair_config
            tree1 = KDTree(data=pos_tree[self.body==pair[0]], leafsize=12, boxsize=self.Lx+0.0000001)
            tree2 = KDTree(data=pos_tree[self.body==pair[1]], leafsize=12, boxsize=self.Lx+0.0000001)
            indexes = tree1.query_ball_tree(tree2, r=2.35)
            dists = []
            for i1,i2s in enumerate(indexes):
                for i2 in i2s:
                    dists.append(wrap_pbc(p1[i1]-p2[i2],self.Lx))
            if(len(dists)==0):
                en = 0.0
            else:
                dists = np.linalg.norm(np.array(dists),axis=1)
                # en = WCA(dists,1.0,1.0)
                en = HalfPowerHalfCosine(dists,2.3,0.00035,1.46375,3.3833)
            self.pair_interactions[i] = np.sum(en)

        return self.pair_interactions, self.pair_configurations


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

class ConvertDataUtils():
    def __init__(self,data):
        self.data = data
        self.Ndata = len(data)
        self.frames = data[:,13]
        self.Lx = np.copy(self.data[0,-2])

    def convert(self):
        """
        8 farkli combination var. Her configuration icin bunlardan birini veya
        birkacini secebiliriz. Parametre range'ini en cok daraltabilen hangisiyse
        onu secmemiz lazim

        2 (unit vector direction for 1) * 2 (unit vector direction for 2) * 2 (1 - 2 can switch) = 8
        """
        self.all_configs = np.zeros((8,self.Ndata,4))
        combis = np.array([
        [1.0,1.0],
        [1.0,-1.0],
        [-1.0,1.0],
        [-1.0,-1.0],
        ])
        for i,combi in enumerate(combis):
            self.convert_with_combi(combi,i)

        self.revert_data()
        for i,combi in enumerate(combis):
            self.convert_with_combi(combi,i+4)


    def revert_data(self):
        """
        After reducing all configurations with 4 combinations once call this function
        To swap cylinders (1) and (2) then reduce with 4 configurations again
        """
        swapped_data = np.zeros_like(self.data)
        swapped_data[:,6:9] = np.copy(self.data[:,:3])
        swapped_data[:,9:12] = np.copy(self.data[:,3:6])
        swapped_data[:,:3] = np.copy(self.data[:,6:9])
        swapped_data[:,3:6] = np.copy(self.data[:,9:12])
        self.data = swapped_data


    def convert_with_combi(self,combi,combination_index):
        disp = wrap_pbc(self.data[:,6:9] - self.data[:,:3],self.Lx)
        axis1 = wrap_pbc(self.data[:,3:6] - self.data[:,:3],self.Lx)
        b2 = wrap_pbc(self.data[:,9:12] - self.data[:,:3],self.Lx)
        axis1 = (axis1/np.linalg.norm(axis1,axis=1).reshape(-1,1))*combi[0]
        for i in range(self.Ndata):
            # print("%d/%d"%(i,self.Ndata))
            """ set the frame such that axis 1 is now z direction"""
            k_dir = np.array([0.0,0.0,1.0])
            v = np.cross(axis1[i],k_dir)
            s = np.linalg.norm(v)
            c = np.dot(axis1[i],k_dir.T)
            skew_sym_v = np.array([
            [0.0,-v[2],v[1]],
            [v[2],0.0,-v[0]],
            [-v[1],v[0],0.0]
            ])
            rotmax1 = np.identity(3) + skew_sym_v + (skew_sym_v@skew_sym_v)*((1-c)/s**2)
            disp_rel = rotmax1@disp[i].T
            b2_rel = rotmax1@b2[i].T
            """ set the frame such that disp is on x direction on xy plane """
            teta = np.arctan2(disp_rel[1],disp_rel[0])
            rotmax2 = rotZ(-teta)
            disp_rel = rotmax2@disp_rel.T
            b2_rel = rotmax2@b2_rel.T

            r = disp_rel[0]
            z = disp_rel[2]
            axis2 = b2_rel-disp_rel
            axis2 = (axis2/np.linalg.norm(axis2))*(combi[1])

            teta,phi = get_theta_phi(axis2)
            rr = np.array([[r,z,teta,phi]])
            self.all_configs[combination_index,i,:] = rr

    def get_reduced_configs(self):
        print("Is NaN? : ",np.isnan(np.sum(self.all_configs)))
        return self.all_configs




    def yarrak(self):
        pass
