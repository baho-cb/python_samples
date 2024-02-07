import numpy as np
import torch
from Force import Force
import gsd.hoomd
from sklearn import tree
from scipy.spatial import cKDTree as KDTree
from HelperFunctions import *
from GsdHandler import GsdHandler
import time
import matplotlib.pyplot as plt
import sys

np.set_printoptions(suppress=True,precision=5,linewidth=150,threshold=sys.maxsize)
torch.set_printoptions(precision=3,threshold=sys.maxsize)
def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta


class Sim():

    def __init__(self):
        print("A new simulation instance created.")

    def setBox(self,L):
        self.Lx = L

    def setGradientMethod(self,method_str):
        self.gradient_method = method_str


    def placeParticlesFromGsd(self,gsd_file):
        gsd_handler = GsdHandler(gsd_file)
        self.central_pos = gsd_handler.get_central_pos()
        self.Nparticles = len(self.central_pos)
        self.N_dof = (self.Nparticles - 1)*3
        self.RN_dof = (self.Nparticles)*3
        self.velocities = gsd_handler.get_COM_velocities()
        self.accel = np.zeros_like(self.velocities)
        self.mass = gsd_handler.getMass()
        ### Angular Data
        self.orientations = gsd_handler.getOrientations()
        self.angmom = gsd_handler.getAngmom()
        self.moi = gsd_handler.getMoi()
        self.moi = self.moi[0]
        self.Lx = gsd_handler.getLx()

        self.moi_to_dump = gsd_handler.getMoi()
        self.charges = gsd_handler.getCharges()
        print("Simulation will be initialized from %s"%(gsd_file))

    def setkT(self,kT):
        self.kT = kT

    def setTau(self,tau):
        self.tau = tau

    def setdt(self,dt):
        self.dt = dt

    def setForces(self,force_path):
        self.force = Force(force_path)

    def setDecisionTree(self,tree_path):
        tree_file = open(tree_path, 'rb')
        self.decision_tree = pickle.load(tree_file)

    def setCutoff(self,cutoff):
        self.cutoff = cutoff

    def setNeighborList(self,cutoff,freq):
        self.Nlist_cutoff = cutoff
        self.Nlist_freq = freq
        pos_tree = self.central_pos + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        dd,ii = tree.query(pos_tree,k=45, distance_upper_bound=self.Nlist_cutoff)
        self.Nlist = ii

    def refreshNList(self):
        pos_tree = self.central_pos + self.Lx*0.5
        tree = KDTree(data=pos_tree, leafsize=12, boxsize=self.Lx+0.0001)
        dd,ii = tree.query(pos_tree,k=45, distance_upper_bound=self.Nlist_cutoff)
        self.Nlist = ii


    def set_integrator(self,ints):
        self.integrator_vars = np.copy(ints)

    def set_m_exp_factor(self,val):
        self.m_exp_thermo_fac = val

    def set_accelerations(self,accel):

        if(self.is_cont==0):
            self.accel = np.zeros_like(self.velocities)
        elif(self.is_cont==1):
            if(accel=='no_accel'):
                print('you must supply an acceleration file if continuing')
                exit()
            accel = np.load(accel)
            self.accel = accel


    def setDumpFreq(self,df):
        self.dumpFreq = df

    def is_continue(self,is_cont):
        """
        if is cont = 1, this means we continue a simulation (probably for debugging)
        so it will replace the usual integrator variables with the ones that come in
        the initiator gsd file.
        """
        self.is_cont = is_cont
        if(is_cont==1):
            print("Integrator variables are set from gsd file - Continuing ")
            self.integrator_vars = np.copy(self.charges[:4])
            self.m_exp_thermo_fac = np.copy(self.charges[4])
            print(self.charges[:5])
        elif(is_cont==0):
            print("Integrator variables are set as default - Not continuing")
        else:
            print("Invalid --is_cont argument should be 0 or 1.")

    def setDump(self,filename):
        """
        Each dump file has 14 columns
        central_pos(3) - orientation(4) - velocities(3) - angular_moms(4)
        All of the above can be dumped as a gsd file
        """
        self.dumpFilename = filename

    def dumpAcceleration(self):
        """
        numpy save replaces/overwrites the existing data which is what we want
        """
        data = self.accel
        names = self.dumpFilename[:-4] + '_accel.npy'
        np.save(names,data)


    def dumpConfig(self,ts):
        pos_all = self.central_pos
        typeid_all = np.zeros(len(self.central_pos),dtype=int)
        velocities_all = self.velocities
        orientations_all = self.orientations
        angmoms_all = self.angmom

        charges = np.zeros_like(self.mass)
        charges[:4] = self.integrator_vars
        charges[4] = self.m_exp_thermo_fac
        snap = gsd.hoomd.Frame()
        # snap = gsd.hoomd.Snapshot()
        snap.configuration.step = ts
        snap.configuration.box = [self.Lx, self.Lx, self.Lx, 0, 0, 0]
        snap.particles.N = len(pos_all)
        snap.particles.position = pos_all
        snap.particles.types  = ['A','B']
        snap.particles.typeid  = typeid_all
        snap.particles.moment_inertia = self.moi_to_dump
        snap.particles.mass = self.mass
        snap.particles.charge = charges

        snap.particles.orientation  = orientations_all
        snap.particles.angmom  = angmoms_all
        snap.particles.velocity  = velocities_all

        if(ts==0):
            with gsd.hoomd.open(name=self.dumpFilename, mode='w') as f:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='wb') as f:
                f.append(snap)
        else:
            # with gsd.hoomd.open(name=self.dumpFilename, mode='rb+') as f:
            with gsd.hoomd.open(name=self.dumpFilename, mode='r+') as f:
                f.append(snap)



    def run(self,N_steps):
        c_dump = 0
        N_dump = N_steps//self.dumpFreq
        self.timestep = 0
        for i in range(N_steps+1):
            if(i%self.Nlist_freq==0 and i>0):
                self.refreshNList()
            if(i%self.dumpFreq==0 and i>0):
                print("%d/%d"%(c_dump,N_dump))
                self.dumpConfig(c_dump)
                c_dump += 1
            self.integrate()


    def integrate(self):
        """
        Nose-Hoover NVT, Two-step Integration
        Implemented from HOOMD Source Code
        """
        if(self.timestep==0):
            self.integrate_mid_step()
            self.integrate_step_two()
        else:
            self.integrate_step_one()
            self.integrate_mid_step()
            self.integrate_step_two()



    def getRotKin(self):
        """
        From ComputeThermo.cc - ke_rot_total
        Scalar3 I = h_inertia.data[j];
        quat<Scalar> q(h_orientation.data[j]);
        quat<Scalar> p(h_angmom.data[j]);
        quat<Scalar> s(Scalar(0.5)*conj(q)*p);
        ke_rot_total /= Scalar(2.0);
        """

        conj_q = np.copy(self.orientations)
        conj_q = -conj_q
        conj_q[:,0] = -conj_q[:,0]
        s = quaternion_multiplication(conj_q,np.copy(self.angmom))
        s = s*0.5
        rot_en = s[:,1]*s[:,1]/self.moi[0] + s[:,2]*s[:,2]/self.moi[1] + s[:,3]*s[:,3]/self.moi[2]
        rot_en = rot_en*0.5
        return np.sum(rot_en)


    def initialize_integrator(self):
        """ see void TwoStepNVTMTK::randomizeVelocities """
        self.integrator_vars = np.zeros(4) # xi, eta, xi_rot, eta_rot
        sigmasq_t = 1.0/(self.N_dof*self.tau**2)
        s = np.random.normal(0.0, np.sqrt(sigmasq_t)) # TODO : I'm not sure about this
        self.integrator_vars[0] = s
        self.m_exp_thermo_fac = 1.0

    def integrate_step_one(self):

        """
        Integrate Step 1 - Trans
        Nothing about the thermostat regular update - No need for for loop
        """

        self.velocities = self.velocities+ (0.5)*self.accel*self.dt
        self.velocities = self.m_exp_thermo_fac*self.velocities
        self.central_pos = self.central_pos +  self.velocities*self.dt
        self.central_pos = np.where(self.central_pos > 0.5 * self.Lx, self.central_pos- self.Lx, self.central_pos)
        self.central_pos = np.where(self.central_pos <- 0.5 * self.Lx, self.Lx + self.central_pos, self.central_pos)

        """
        Integrate Step 1 - Rotation
        """
        tt = rotate_torks_to_body_frame(self.orientations,self.torks)
        xi_rot = np.copy(self.integrator_vars[2])
        exp_fac = np.exp((-self.dt/2.0)*xi_rot)
        dp = get_dp(self.orientations,tt,self.dt)
        self.angmom += dp
        self.angmom *= exp_fac
        self.angmom,self.orientations = permutation1(self.angmom,self.orientations,self.moi[2],self.dt)
        self.angmom,self.orientations = permutation2(self.angmom,self.orientations,self.moi[1],self.dt)
        self.angmom,self.orientations = permutation3(self.angmom,self.orientations,self.moi[0],self.dt)
        self.angmom,self.orientations = permutation2(self.angmom,self.orientations,self.moi[1],self.dt)
        self.angmom,self.orientations = permutation1(self.angmom,self.orientations,self.moi[2],self.dt)
        self.orientations = renormalize_quat(self.orientations)

        """
        Advance Thermostat - Trans
        """
        trans_kin_en = (0.5)*(self.mass)*(np.sum(self.velocities*self.velocities,axis=1))
        trans_kin_en = np.sum(trans_kin_en)
        trans_temp = (2.0/self.N_dof)*trans_kin_en
        xi_prime = self.integrator_vars[0] + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)
        self.integrator_vars[0] = xi_prime + (0.5)*((self.dt/self.tau)/self.tau)*(trans_temp/self.kT - 1.0)
        self.integrator_vars[1] += xi_prime*self.dt
        self.m_exp_thermo_fac = np.exp(-0.5*self.integrator_vars[0]*self.dt);


        """
        Advance Thermostat - Rot
        Scalar xi_prime_rot = xi_rot + Scalar(1.0/2.0)*m_deltaT/m_tau/m_tau*
            (Scalar(2.0)*curr_ke_rot/ndof_rot/m_T->getValue(timestep) - Scalar(1.0));
        """
        xi_rot = np.copy(self.integrator_vars[2])
        eta_rot = np.copy(self.integrator_vars[3])
        rot_kin_en = self.getRotKin()
        xi_prime_rot = xi_rot + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*rot_kin_en)/self.RN_dof)/self.kT - 1.0)
        xi_rot =  xi_prime_rot + (0.5)*((self.dt/self.tau)/self.tau)*( ((2.0*rot_kin_en)/self.RN_dof)/self.kT - 1.0)

        eta_rot = eta_rot + xi_prime_rot*self.dt
        self.integrator_vars[2] = xi_rot
        self.integrator_vars[3] = eta_rot

    def integrate_step_two(self):
        """
        Integrate Step 2 - Trans
        4. velocity component is mass in HOOMD code

        """
        self.accel = self.forces/self.mass.reshape(-1,1)
        self.velocities = self.velocities*self.m_exp_thermo_fac
        self.velocities = self.velocities + (0.5)*(self.dt)*(self.accel) ### sikintili step bu
        """
        Integrate Step 2 - Rot

        Only advanced angular momentum, don't touch orientations

        Forces are fine at the box frame but the torks must be in the hoomd - shape
        frame which is defined by the orientation quaternion. To convert the box frame
        torks to just repeat the hoomd code rotate(conj(quat),tork)
        """

        tt = rotate_torks_to_body_frame(self.orientations,self.torks)
        xi_rot = np.copy(self.integrator_vars[2])
        exp_fac = np.exp((-self.dt/2.0)*xi_rot)
        self.angmom *= exp_fac
        dp = get_dp(self.orientations,tt,self.dt)
        self.angmom += dp

        self.timestep += 1


    def integrate_mid_step(self):
        self.obtain_reduced_pair_configs()
        self.infer_selector_net()
        if(self.gradient_method=="forward"):
            self.infer_energy_net_forward()
        elif(self.gradient_method=="midpoint"):
            self.infer_energy_net_midpoint()
        else:
            print("Gradient method can't be %s."%self.gradient_method)
        self.calculate_force_and_torques()


    def obtain_reduced_pair_configs(self):
        """
        From raw representation (2 quaternions and 2 positions) of the pairs of
        cubes in the simulation obtain the 6 descriptors as the input parameters
        of the neural net for each pair.
        Details of geometric operations are explained in the SI of the paper. 
        """
        """ extract pairs """
        NlistRHS = np.copy(self.Nlist[:,1:])
        pair1 = NlistRHS.flatten()
        pair0 = np.repeat(np.arange(self.Nparticles),44)
        pair0 = pair0[pair1<self.Nparticles-0.5]
        pair1 = pair1[pair1<self.Nparticles-0.5]

        delta = self.central_pos[pair0]-self.central_pos[pair1]
        delta = np.where(delta > 0.5 * self.Lx, delta- self.Lx, delta)
        delta = np.where(delta <- 0.5 * self.Lx, self.Lx + delta, delta)
        delta = np.linalg.norm(delta,axis=1)

        min_d = np.min(delta)
        inocu = np.argmin(delta)

        if(min_d<3.4):
            print(min_d)
            print(pair0[inocu],pair1[inocu])

        pair0 = pair0[delta<self.cutoff]
        pair1 = pair1[delta<self.cutoff]
        delta = delta[delta<self.cutoff]

        N_pair = len(pair0)

        self.data = np.zeros((N_pair,12))
        self.N_pair = N_pair

        COM1 = self.central_pos[pair0]
        COM2 = self.central_pos[pair1]
        QUAT1 = self.orientations[pair0]
        QUAT2 = self.orientations[pair1]

        """ FIND INTERACTING FACE 1 """

        true_faces1 = np.array([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [-1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,-1.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,-1.0]
        ])

        true_faces1 = np.tile(true_faces1,(self.N_pair,1))
        q1_faces = np.repeat(QUAT1,7,axis=0)
        com1_faces = np.repeat(COM1,7,axis=0)
        com2_faces = np.repeat(COM2,7,axis=0)
        faces1 = rotate(q1_faces,true_faces1)
        faces1_abs = faces1 + com1_faces
        com2_faces1_rel = wrap_pbc(com2_faces-faces1_abs,self.Lx)
        dist2faces1 = np.linalg.norm(com2_faces1_rel,axis=1)
        dist2faces1 = dist2faces1.reshape(-1,7)
        face1_index = np.argmin(dist2faces1,axis=1)

        """ FIND INTERACTING FACE 2 """

        true_faces2 = np.array([
        [0.0,0.0,0.0],
        [1.0,0.0,0.0],
        [-1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,-1.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,-1.0]
        ])

        true_faces2 = np.tile(true_faces2,(self.N_pair,1))
        q2_faces = np.repeat(QUAT2,7,axis=0)
        com1_faces = np.repeat(COM1,7,axis=0)
        com2_faces = np.repeat(COM2,7,axis=0)
        faces2 = rotate(q2_faces,true_faces2)
        faces2_abs = faces2 + com2_faces

        com1_faces2_rel = wrap_pbc(faces2_abs - com1_faces,self.Lx)
        dist2faces2 = np.linalg.norm(com1_faces2_rel,axis=1)

        dist2faces2 = dist2faces2.reshape(-1,7)
        face2_index = np.argmin(dist2faces2,axis=1)

        """ Rotate everything such that interacting face1 is [1.0,0.0,0.0] """
        right_true = np.array([1.0,0.0,0.0])
        right_true = np.tile(right_true,(self.N_pair,1))

        faces1i = faces1.reshape(-1,7,3)
        faces1_inter = np.copy(faces1i[np.arange(self.N_pair), face1_index, :])

        q_rot1u = quat_from_two_vectors(faces1_inter,right_true)
        q_rot1 = np.repeat(q_rot1u,7,axis=0)
        faces1_r1 = rotate(q_rot1,faces1)
        faces2_r1 = rotate(q_rot1,com1_faces2_rel)

        face1p_index = (face1_index + 2)%7
        face1p_index[face1p_index==0] = 1

        forward_true = np.array([0.0,1.0,0.0])
        forward_true = np.tile(forward_true,(self.N_pair,1))
        faces1ri = faces1_r1.reshape(-1,7,3)


        faces1p_inter = np.copy(faces1ri[np.arange(self.N_pair), face1p_index, :])

        q_rot2u = quat_from_two_vectors(faces1p_inter,forward_true)
        q_rot2 = np.repeat(q_rot2u,7,axis=0)
        faces1_r2 = rotate(q_rot2,faces1_r1)
        faces2_r2 = rotate(q_rot2,faces2_r1)

        faces2ri = faces2_r2.reshape(-1,7,3)
        faces2_r2_com = faces2ri[:,0,:]

        multiplier = np.ones_like(faces2_r2)
        y_signs = np.sign(faces2_r2_com[:,1])
        z_signs = np.sign(faces2_r2_com[:,2])

        multiplier_force = np.ones((self.N_pair,3))
        multiplier_force[:,1] = y_signs
        multiplier_force[:,2] = z_signs

        multiplier_tork = np.ones((self.N_pair,3))
        multiplier_tork[:,0] = y_signs*z_signs
        multiplier_tork[:,1] = z_signs
        multiplier_tork[:,2] = y_signs

        y_signs = np.repeat(y_signs,7)
        z_signs = np.repeat(z_signs,7)

        multiplier[:,1] = y_signs
        multiplier[:,2] = z_signs

        faces2_r2 = faces2_r2*multiplier
        faces2ri = faces2_r2.reshape(-1,7,3)
        faces2_r2_com = faces2ri[:,0,:]

        will_switch = np.where(faces2_r2_com[:,2]>faces2_r2_com[:,1],np.ones(self.N_pair,dtype=int),np.zeros(self.N_pair,dtype=int))
        will_switch = np.ma.make_mask(will_switch, shrink=False)

        faces2ri_c23 = np.copy(faces2ri[:,:,[1,2]])
        faces2ri_c23[will_switch] = faces2ri_c23[will_switch,:, ::-1]
        faces2ri[:,:,[1,2]] = faces2ri_c23

        faces2_r2 = faces2ri.reshape(-1,3)
        faces2_inter = np.copy(faces2ri[np.arange(self.N_pair), face2_index, :])


        faces2_r2_com = faces2ri[:,0,:]
        faces2_r2_com7 = np.repeat(faces2_r2_com,7,axis=0)

        faces2_r2_relcom2 = faces2_r2 - faces2_r2_com7


        faces2_inter_relcom2 = np.zeros((self.N_pair,3))
        faces2_r2_relcom2_3d = faces2_r2_relcom2.reshape(-1,7,3)
        faces2_inter_relcom2 = np.copy(faces2_r2_relcom2_3d[np.arange(self.N_pair), face2_index, :])


        """
        Be careful we take the cosine from the opposite direction
        Very rarely arccos won't work because the x dimension will be larger than 1 due to numerical errors
        """
        faces2_inter_relcom2[faces2_inter_relcom2[:,0]<-1.0,0] = -1.0
        faces2_inter_relcom2[faces2_inter_relcom2[:,0]>1.0,0] = 1.0

        xcos_angle2 = np.arccos(-faces2_inter_relcom2[:,0])
        yztan_angle2 = np.arctan2(-faces2_inter_relcom2[:,2],-faces2_inter_relcom2[:,1])


        """
        - Only the last angle of 2 is left
        (1) Find the quat that will rotate face2_inter_relcom2 towards [-1,0,0]
        (2) Apply the quat to the faces2_r2_relcom2
        (3) See that opposite of the face2_inter_relcom2 is towards [1,0,0]
        (4) Find the angle to rotate other faces towards z and y axis
        """

        true_left = np.array([-1.0,0.0,0.0])
        true_left = np.tile(true_left,(len(faces2_inter_relcom2),1))
        q21 = quat_from_two_vectors(faces2_inter_relcom2,true_left)
        q21 = np.repeat(q21,7,axis=0)
        faces2_r2_r21_relcom2 = rotate(q21,faces2_r2_relcom2)

        faces2p_inter = np.zeros((self.N_pair,3))
        faces2r21_3d = faces2_r2_r21_relcom2.reshape(-1,7,3)

        ff2 = faces2r21_3d[:, 1:, :]

        # Filter elements based on conditions using boolean indexing
        condition1 = np.abs(ff2[:,:,0]) < 0.001
        condition2 = ff2[:,:,2] >= 0.0
        condition3 = ff2[:,:,1] > 0.0
        # Apply conditions using logical AND
        condition_all = condition1 & condition2 & condition3
        # Apply the combined condition to filter ff
        faces2p_inter = ff2[condition_all]

        ## last angle is correct
        last_angle = np.arctan2(faces2p_inter[:,2],faces2p_inter[:,1])
        reduced_configs = np.zeros((len(last_angle),6))

        reduced_configs[:,:3] = faces2_r2_com
        reduced_configs[:,3] = xcos_angle2
        reduced_configs[:,4] = yztan_angle2
        reduced_configs[:,5] = last_angle

        """
        Store the reverse rotation quaternions
        Reverse rotation for quaternions are obtained by conjugating them
        which is equivalent to negate the vector part
        """

        self.rotQ1 = -q_rot1u
        self.rotQ1[:,0] = -self.rotQ1[:,0]
        self.rotQ2 = -q_rot2u
        self.rotQ2[:,0] = -self.rotQ2[:,0]

        self.reduced_configs = reduced_configs
        self.pair0 = pair0
        self.pair1 = pair1
        self.multiplier_tork = multiplier_tork
        self.multiplier_force =  multiplier_force
        self.will_switch = will_switch

    def infer_selector_net(self):
        """
        First neural net is binary classifier that eliminates non-interacting
        pairs of cubes to reduce the computational cost of energy inference
        """
        self.mins_ = torch.tensor([0.98,0.0,0.0,0.0,-3.142,0.0],dtype=torch.float32)
        self.maxs_ = torch.tensor([6.31,5.43,3.65,2.04, 3.142,1.572],dtype=torch.float32)

        reduced_configs_torch = torch.from_numpy(self.reduced_configs)
        x_data = torch.zeros_like(reduced_configs_torch)
        x_data = x_data.to(torch.float32)
        reduced_configs_torch = reduced_configs_torch.to(torch.float32)
        index6 = np.array([0,1,2,3,4,5])
        x_data[:,index6] = (reduced_configs_torch[:,index6] - self.mins_[index6]) / (self.maxs_[index6] - self.mins_[index6])

        selector_output = self.force.selector(x_data)

        selector_output = selector_output.detach().numpy()
        selector_output = np.argmax(selector_output,axis=1)
        interacting_index_nn = np.where(selector_output==1)[0]
        self.interacting_reduced_configs = self.reduced_configs[interacting_index_nn]
        self.pair0 = self.pair0[interacting_index_nn]
        self.pair1 = self.pair1[interacting_index_nn]
        self.rotQ1 = self.rotQ1[interacting_index_nn]
        self.rotQ2 = self.rotQ2[interacting_index_nn]
        self.multiplier_tork = self.multiplier_tork[interacting_index_nn]
        self.multiplier_force = self.multiplier_force[interacting_index_nn]
        self.will_switch = self.will_switch[interacting_index_nn]
        self.NpairInter = len(self.pair0)
        self.x_true = self.interacting_reduced_configs
        self.x_data = x_data[interacting_index_nn]

    def infer_energy_net_forward(self):
        """
        To take gradient wrt 6 inputs you need to do 7 passes through the neural net
        E(x), E(x+dx1), E(x+dx2), ...
        Here I stack the inputs on top of each other and pass them together to save time

        """
        dxs = np.array([0.006,0.01,0.03,0.03,0.01,0.03]) # 0.
        xx_ender = torch.clone(self.x_data)
        mm = xx_ender.size()[0]
        xx_ender = torch.tile(xx_ender,(7,1))
        dxx = torch.zeros_like(xx_ender)
        for i in range(6):
            dxx[(i+1)*mm:(i+2)*mm,i] = dxs[i]
        xx_ender = xx_ender + dxx
        e_xx = self.force.energy_net(xx_ender)
        e_x = e_xx[:mm]
        e_xx = torch.flatten(e_xx)
        e_xdx = torch.zeros((mm,6))
        for i in range(1,7):
            e_xdx[:,i-1] = e_xx[i*mm:(i+1)*mm]

        gradient = (e_xdx - e_x)
        gradient = gradient.detach().numpy()
        self.gradient = gradient/dxs

    def infer_energy_net_midpoint(self):
        """
        Gradients are more accurate than forward but pass through the neural-net
        is twice as expensive
        """

        dxs = np.array([0.006,0.01,0.03,0.03,0.01,0.03]) ## these dx are shown to be most accurate
        xx_ender = torch.clone(x_data)

        mm = xx_ender.size()[0]
        xx_ender = torch.tile(xx_ender,(12,1))
        dxx = torch.zeros_like(xx_ender)

        for i in range(6):
            dxx[i*mm:(i+1)*mm,i] = -dxs[i]
        for i in range(6,12):
            dxx[i*mm:(i+1)*mm,i-6] = dxs[i-6]

        xx_ender = xx_ender + dxx
        e_xx = self.force.energy_net(xx_ender)

        e_x_m_dx = e_xx[:mm*6]
        e_x_p_dx = e_xx[mm*6:]
        dE = e_x_p_dx - e_x_m_dx

        dxs = np.repeat(dxs,mm)
        dE = dE.detach().numpy().flatten()
        gradient = dE/(dxs*2.0)
        self.gradient = gradient.reshape(-1,6,order='F')

    def calculate_force_and_torques(self):

        """
        FORCES AND TORQUES AS DERIVATIVES OF ENERGY
        WITHOUT TORCH AUTOGRAD
        """
        x_true = self.x_true
        gradient = self.gradient
        forces_inter = np.zeros((self.NpairInter,3))
        torks_inter = np.zeros((self.NpairInter,3))

        en_min = -5.1
        en_max = 17.0

        gradient[:,0] = gradient[:,0]*(en_max-en_min)/(self.maxs_[0]-self.mins_[0])
        gradient[:,1] = gradient[:,1]*(en_max-en_min)/(self.maxs_[1]-self.mins_[1])
        gradient[:,2] = gradient[:,2]*(en_max-en_min)/(self.maxs_[2]-self.mins_[2])
        gradient[:,3] = gradient[:,3]*(en_max-en_min)/(self.maxs_[3]-self.mins_[3])
        gradient[:,4] = gradient[:,4]*(en_max-en_min)/(self.maxs_[4]-self.mins_[4])
        gradient[:,5] = gradient[:,5]*(en_max-en_min)/(self.maxs_[5]-self.mins_[5])


        a1 = x_true[:,3]
        a2 = x_true[:,4]

        tz = gradient[:,0]*(-x_true[:,1]) + gradient[:,1]*(x_true[:,0]) + gradient[:,3]*np.cos(x_true[:,4])
        tz = tz + gradient[:,4]*(-np.cos(a1)*np.sin(a2)/np.sin(a1)) + gradient[:,5]*np.sin(a2)*(1-np.cos(a1))/np.sin(a1)

        ty = gradient[:,0]*(x_true[:,2]) + gradient[:,1]*0 + gradient[:,2]*(-x_true[:,0])
        ty = ty + gradient[:,3]*-np.sin(x_true[:,4]) + gradient[:,4]*(-np.cos(a2)*np.cos(a1)/np.sin(a1)) +  gradient[:,5]*(np.cos(a2)*(1-np.cos(a1))/np.sin(a1))

        tx = gradient[:,0]*0 + gradient[:,1]*(-x_true[:,2]) + gradient[:,2]*(x_true[:,1])
        tx = tx + gradient[:,3]*0.0 + gradient[:,4]*1.0 +  gradient[:,5]*1.0


        forces_inter[:,0] = gradient[:,0]
        forces_inter[:,1] = gradient[:,1]
        forces_inter[:,2] = gradient[:,2]

        torks_inter[:,0] = tx
        torks_inter[:,1] = ty
        torks_inter[:,2] = tz

        forces_inter[forces_inter>100.0] = 100.0
        torks_inter[torks_inter>100.0] = 100.0

        forces_inter[forces_inter<-100.0] = -100.0
        torks_inter[torks_inter<-100.0] = -100.0

        """
        Calculate Net Force-Tork On each Shape
        Forces are at the interaction frame, reorient them to be in the box frame
        """

        for i in range(self.NpairInter):
            if(self.will_switch[i]==1):
                forces_inter[i,[1,2]] = forces_inter[i,[2,1]]
                torks_inter[i,[1,2]] = -torks_inter[i,[2,1]]
                torks_inter[i,0] = -torks_inter[i,0]


        forces_inter = forces_inter*self.multiplier_force
        torks_inter = torks_inter*self.multiplier_tork



        forces_inter_box = rotate(self.rotQ2,forces_inter)
        forces_inter_box = rotate(self.rotQ1,forces_inter_box)

        torks_inter_box = rotate(self.rotQ2,torks_inter)
        torks_inter_box = rotate(self.rotQ1,torks_inter_box)


        forces_net = np.zeros((self.Nparticles,3))
        torks_net = np.zeros((self.Nparticles,3))

        forces_net = torch.zeros((self.Nparticles,3),dtype=torch.float64)
        torks_net = torch.zeros((self.Nparticles,3),dtype=torch.float64)

        forces_net.index_add_(0, torch.from_numpy(self.pair0), torch.from_numpy(forces_inter_box))
        torks_net.index_add_(0, torch.from_numpy(self.pair0), torch.from_numpy(torks_inter_box))

        self.torks = torks_net.detach().numpy()
        self.forces = forces_net.detach().numpy()
