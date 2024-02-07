import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
import os.path
import sys
import time
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as Rot
from HelperFunctions import *
from Shape import Shape
np.set_printoptions(suppress=True,threshold=sys.maxsize)




class SimulatorHandler():
    """
    used in oneboxer.hoomd
    """
    def __init__(self,input):
        self.shape_filename = input
        self.setShape()

    def setShape(self):
        """
        Create the shape class
        """
        self.shape = Shape(self.shape_filename)
        self.shape.set_positions()

    def setSeed(self,seed):
        np.random.seed(seed)

    def setBox(self,Box_size,N_mp):
        """
        Set Stuff
        """
        self.Lx = Box_size
        self.Ly = Box_size
        self.Lz = Box_size
        self.N_mp = N_mp
        self.placeMps()


    def placeMps(self):
        """
        First MP goes to origin without rotation,
        Than set the rigid body stuff since this is how it was done in rras_v6
        the rest is added using AddShape
        function from p_shear
        """
        self.positions = self.shape.getPositions()
        self.typeid = self.shape.getTypeids()
        self.types = ['Re','Os']
        self.velocities = np.zeros_like(self.positions)
        self.angmom = np.zeros((len(self.positions),4))
        self.setRigidBody()
        print("Placing Shapes ...")
        for i in range(self.N_mp-1):
            if(i%100==0):
	    	          print("%d/%d" %(i,self.N_mp-1))
            self.addShape()

    def setRigidBody(self):
        """
        Sets rigid body related things like mass, body, inertia etc.
        """
        mass_shape = self.shape.getmass()
        body_shape = self.shape.getbody()
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        # print(orientation_shape)
        # exit()
        # mass_noms = np.ones(self.NomPerShape)
        # body_noms = np.ones(self.NomPerShape)*-1.0

        # inertia_noms = np.zeros((self.NomPerShape,3))
        # orientation_noms = np.tile([1.0,0.0,0.0,0.0],(self.NomPerShape,1))

        self.mass = mass_shape
        self.body = body_shape
        self.moment_inertia = inertia_shape
        self.orientation = orientation_shape


    def getMOIShape(self):
        return self.shape.getMOI2()

    def addShape(self):
        """
        add the new shape away from the previous shapes
        Central particles must have a body tag identical to their contiguous tag.
        """

        t = 0
        current_body = len(self.positions)
        pos_new_shape = self.shape.getPositions()
        typeid_new_shape = self.shape.getTypeids()
        velocity_new_shape = np.zeros_like(pos_new_shape)
        angmom_new_shape = np.zeros((len(pos_new_shape),4))
        pos_old = self.positions
        good = False
        margin = 1.0
        while(good is False):
            pos_new_shape = self.shape.getPositions()
            t += 1
            p = (np.random.rand(3) - 0.5)*self.Lx ### 0.70 to make encounter earlier see 28.1
            # p = np.array([3.0,3.0,3.0]) ## when debugging with 2 shapes
            ### turn these on for old style - this is for approach2
            # quat = random_quat()
            # rot_max = quat_to_matrix(quat)
            # quat_hoomd = quat_converter(quat)
            pos_new_shape = translate(pos_new_shape,p,self.Lx)
            # pos_new_shape = rotatePbc(pos_new_shape,rot_max,self.Lx)
            good = check_overlap2(pos_new_shape,pos_old,margin,self.Lx)
            if(t>100000):
                print("AddShape tried too much!!!")
                exit()

        self.positions = np.vstack((self.positions,pos_new_shape))
        self.typeid = np.hstack((self.typeid,typeid_new_shape))
        self.velocities = np.vstack((self.velocities,velocity_new_shape))
        self.angmom = np.vstack((self.angmom,angmom_new_shape))


        mass_shape = self.shape.getmass()
        body_shape = np.ones_like(mass_shape)*current_body
        inertia_shape = self.shape.getMOI()
        orientation_shape = self.shape.getorientation()
        # orientation_shape[0,:] = quat_hoomd
        # print(quat_hoomd)
        self.mass = np.hstack((self.mass,mass_shape))
        self.body = np.hstack((self.body,body_shape))
        self.moment_inertia = np.vstack((self.moment_inertia,inertia_shape))
        self.orientation = np.vstack((self.orientation,orientation_shape))



    def getRigidConstituents(self):
        """
        Return the shape template positions except the central
        """
        pos = self.shape.getPositions()
        pos = pos[1:]
        return pos

    def get_snap(self,context):
        with context:


            snap = make_snapshot(N=len(self.positions),
                                particle_types=self.types,
                                # bond_types=self.bond_types,
                                box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))


            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
                snap.particles.body[k] = self.body[k]
                snap.particles.moment_inertia[k] = self.moment_inertia[k]
                snap.particles.orientation[k] = self.orientation[k]
                snap.particles.mass[k] = self.mass[k]
                snap.particles.velocity[k] = self.velocities[k]
                snap.particles.angmom[k] = self.angmom[k]

                # snap.particles.charge[k] = self.charges[k]
            # set bond typeids and groups
            # snap.bonds.resize(len(self.bond_group))
            # for k in range(len(self.bond_group)):
            #     snap.bonds.typeid[k] = self.bond_typeid[k]
            #     snap.bonds.group[k] = self.bond_group[k]
            # if(len(snap.particles.body)>4):
            #     snap.particles.body[snap.particles.body[:] > 5000]=-1

            # print(self.body)
            # print(type(snap.particles.body))
            # exit()

        return snap
