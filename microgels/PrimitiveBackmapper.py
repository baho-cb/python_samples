import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict
import os.path
import networkx as nx
import itertools
import time
# ------------------------------------------------------------------------------
#                              Helper functions
#-------------------------------------------------------------------------------
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def get_all_bonds(id,bond_group):
    old_bonds_1 = bond_group[bond_group[:,0]==id]
    old_bonds_2 = bond_group[bond_group[:,1]==id]
    old_bonds = np.vstack((old_bonds_1,old_bonds_2))
    return old_bonds


class backbone:

    ### 3 carbons and one oxygen - first carbon at origin others have relative position scaled
    def __init__(self,i_bead):
        self.i_bead = i_bead
        self.ids = np.array([0,1,2,3])
        self.positions = np.array([
        [0.18917,2.26431,0.15113],
        [1.25304,3.34238,-0.05715],
        [2.65389,2.86961,0.25625],
        [2.84529,1.72449,0.64341],
        ])
        self.positions[1,:] = self.positions[1,:] - self.positions[0,:]
        self.positions[2,:] = self.positions[2,:] - self.positions[0,:]
        self.positions[3,:] = self.positions[3,:] - self.positions[0,:]
        self.positions[0,:] = np.array([0.0,0.0,0.0])
        # self.velx = np.ones(4)
        self.types = np.array([0,0,3,2])

    def place(self,id_offset,c_of_m):
        self.positions = np.add(self.positions,c_of_m) # not sure about the axis
        self.ids = self.ids + id_offset
        return(id_offset + 4)

class t_junction:
    # 6 carbon 2 azot
    # N -C -C -C -C -C -C N
    # First azot two ways chain, last azot one way chain
    # 7 only new bond btw the middle carbons
    # No angles, too much work
    def __init__(self,i_bead):
        self.i_bead = i_bead
        self.ids = np.array([0,1,2,3,4,5,6,7])
        self.positions = np.array([
        [-15.03978    ,    3.99001    ,   -0.01578],
        [-13.50190   ,     3.00670   ,     0.32864],
        [-14.52597    ,    1.78065     ,   0.03969],
        [-12.69822      ,  1.69573    ,    0.82281],
        [-11.82610   ,     3.78152 ,      -0.12051],
        [-12.02356    ,    4.32563   ,    -1.52178],
        [-11.58592   ,     4.94801   ,     0.86812],
        [-10.16999    ,    3.04633  ,     -0.02516]
        ])

        for i in range(1,8):
            self.positions[i,:] = self.positions[i,:] - self.positions[0,:]
        self.positions[0,:] = np.array([0.0,0.0,0.0])
        self.types = np.array([1,0,0,0,0,0,0,1])
    def place(self,id_offset,c_of_m):
        self.positions = np.add(self.positions,c_of_m) # not sure about the axis
        self.ids = self.ids + id_offset
        return(id_offset + 8)

class crosslinker:
    # N - C - N
    def __init__(self,i_bead):
        self.i_bead = i_bead
        self.ids = np.array([0,1,2])
        self.positions = np.array([
        [0.54402    ,    2.67061    ,    0.05782],
        [1.92254     ,   2.05704        ,0.02995],
        [3.17180     ,   2.90341 ,       0.00711]
        ])

        self.positions[1,:] = self.positions[1,:] - self.positions[0,:]
        self.positions[2,:] = self.positions[2,:] - self.positions[0,:]
        self.positions[0,:] = np.array([0.0,0.0,0.0])
        self.types = np.array([1,3,1])

    def place(self,id_offset,c_of_m):
        self.positions = np.add(self.positions,c_of_m) # not sure about the axis
        self.ids = self.ids + id_offset
        return(id_offset + 3)


class sidechain:
        ### 1 azot and 3 carbons
    def __init__(self,i_bead):
        self.i_bead = i_bead
        self.ids = np.array([0,1,2,3])
        self.positions = np.array([
        [ 3.69612     ,   3.73635  ,      0.11733],
        [5.08177   ,     3.37636    ,    0.39879],
        [5.97803   ,     4.61820    ,    0.30430],
        [5.58596    ,    2.30824     ,  -0.58488],
        ])
        self.positions[1,:] = self.positions[1,:] - self.positions[0,:]
        self.positions[2,:] = self.positions[2,:] - self.positions[0,:]
        self.positions[3,:] = self.positions[3,:] - self.positions[0,:]
        self.positions[0,:] = np.array([0.0,0.0,0.0])
        self.types = np.array([1,0,0,0])

    def place(self,id_offset,c_of_m):
        self.positions = np.add(self.positions,c_of_m) # not sure about the axis
        self.ids = self.ids + id_offset
        return(id_offset + 4)

class PrimitiveBackmapper():
    def __init__(self,input,frame):
        print("Reading in : " + input)
        self.positions = []
        self.types = []
        self.bond_types = []
        self.typeid = []
        self.bond_typeid = []
        self.bond_group = []

        self.angle_types = []
        self.angle_typeid = []
        self.angle_group = []

        self.dihedral_types = []
        self.dihedral_typeid = []
        self.dihedral_group = []

        self.pair_types = []
        self.pair_typeid = []
        self.pair_group = []

        self.Lx = 0
        self.Ly = 0
        self.Lz = 0
        self.msd = []

        self.coarse_grain_level = -1
        self.read_system(input,frame)
        self.detect_coarse_grain_level()

    def return_level(self):
        return(self.coarse_grain_level)

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    print("Reading last frame ")
                else:
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()
                self.charges = (frame.particles.charge).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.bond_types = (frame.bonds.types).copy()
                self.bond_group = (frame.bonds.group).copy()
                self.bond_typeid = (frame.bonds.typeid).copy()

                self.angle_types = (frame.angles.types).copy()
                self.angle_group = (frame.angles.group).copy()
                self.angle_typeid = (frame.angles.typeid).copy()

                self.dihedral_types = (frame.dihedrals.types).copy()
                self.dihedral_group = (frame.dihedrals.group).copy()
                self.dihedral_typeid = (frame.dihedrals.typeid).copy()

                self.pair_types = (frame.pairs.types).copy()
                self.pair_group = (frame.pairs.group).copy()
                self.pair_typeid = (frame.pairs.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()
            self.charges = (input.particles.charge).copy()

            self.bond_types = (input.bonds.types).copy()
            self.bond_group = (input.bonds.group).copy()
            self.bond_typeid = (input.bonds.typeid).copy()

            self.angle_types = (input.angles.types).copy()
            self.angle_group = (input.angles.group).copy()
            self.angle_typeid = (input.angles.typeid).copy()

            self.dihedral_types = (input.dihedrals.types).copy()
            self.dihedral_group = (input.dihedrals.group).copy()
            self.dihedral_typeid = (input.dihedrals.typeid).copy()

            self.pair_types = (input.pairs.types).copy()
            self.pair_group = (input.pairs.group).copy()
            self.pair_typeid = (input.pairs.typeid).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx


    def detect_coarse_grain_level(self):
        if 'A' and 'B' in self.types:
            self.coarse_grain_level = 0
            print("Input level 0")
        elif 'sidechain' and 'backbone' in self.types:
            self.coarse_grain_level = 1
            print("Input level 1")
        elif ('CT' in self.types and 'HC' not in self.types):
            self.coarse_grain_level = 2
            print("Input level 2")
        elif ('CT' and 'HC' in self.types):
            self.coarse_grain_level = 3
            print("Input level 3")
        else:
            self.coarse_grain_level = -1
            print('?')


    def exclusion_check(self,new_bonds,old_types):
            bonds = new_bonds
            all_ids = new_bonds
            ids = list(np.unique((np.array(new_bonds)).flatten()))
            G = nx.Graph()
            G.add_nodes_from(ids)
            G.add_edges_from(bonds)
            N_tj = np.count_nonzero(old_types==3)
            count_4_exclusion = 0
            for id in ids:
                neighbors = [n for n in G.neighbors(id)]
                if(len(neighbors)>3.5):
                    count_4_exclusion = count_4_exclusion + 1

            if(N_tj*2!=count_4_exclusion):
                print("Expected 4 exlusions : " + str(N_tj*2) + " // Current 4 exclusions : " + str(count_4_exclusion))
                exit()

    def check_consistency_lvl2(self,old_types):
        bonds = self.bond_group
        all_ids = self.bond_group

        ids = list(np.unique((np.array(self.bond_group)).flatten()))
        particle_types = np.array(self.typeid)

        N_carbons = np.count_nonzero(particle_types == 0)
        N_nitrogens = np.count_nonzero(particle_types == 1)
        N_oxygens = np.count_nonzero(particle_types == 2)
        N_carbonamide = np.count_nonzero(particle_types == 3)


        N_tj = np.count_nonzero(old_types == 3)
        N_cr = np.count_nonzero(old_types == 2)
        N_bb = np.count_nonzero(old_types == 1)
        N_sc = np.count_nonzero(old_types == 0)

        print("N bb : " + str(N_bb))
        print("N_sc : " + str(N_sc))
        print("N_tj : " + str(N_tj))
        print("N_cr : " + str(N_cr))

        if(N_nitrogens!=N_oxygens):
            print("N nitrogen : " + str(N_nitrogens))
            print("N_oxygens : " + str(N_oxygens))
            print("N_carbons : " + str(N_carbons))
            print("N_carbonamide : " +  str(N_carbonamide))
            print(len(particle_types))
            print(len(ids))
            print(len(self.positions))

        if(len(particle_types)!=len(ids) or len(ids)!=len(self.positions)):
            print("Error 29 -- Lvl 2 Consistency Check")

        carbons = []
        nitrogens = []
        oxygens = []
        carbonamides = []

        for i in ids:
            if(particle_types[i]==0):
                carbons.append(i)
            if(particle_types[i]==1):
                nitrogens.append(i)
            if(particle_types[i]==2):
                oxygens.append(i)
            if(particle_types[i]==3):
                carbonamides.append(i)

        N_total = len(oxygens) + len(carbons) + len(nitrogens) + len(carbonamides)





    def reduce_coarse_grain_level(self,context):
        with context:
            if self.coarse_grain_level==0:
                print("Nope")
                exit()
            elif self.coarse_grain_level==1:
                self.from_1_to_2()
                output_level = 2
            elif self.coarse_grain_level==2:
                print("Nope")
                exit()

            if(output_level==2):
                dist = np.linalg.norm(self.positions,axis=1)
                ind_p = np.unravel_index(np.argsort(dist,axis=None),dist.shape)
                ind_p = ind_p[0][::-1]
                maxdist = dist[ind_p[0]]
                self.Lx = maxdist*2.0 + 60.0
                self.Ly = maxdist*2.0 + 60.0
                self.Lz = maxdist*2.0 + 60.0
            snap = make_snapshot(N=len(self.positions),
                                 particle_types=self.types,
                                 bond_types=self.bond_types,
                                 angle_types=self.angle_types,
                                 dihedral_types=self.dihedral_types,
                                 pair_types=self.pair_types,
                                 box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz)
                                 )

            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]

            return snap,output_level

    def from_1_to_2(self):
        ### doesnt matter if msd is in charge ot velx both are
        ### carried from lvl1 to lvl2
        old_positions = self.positions
        old_types = self.typeid
        old_positions = np.multiply(old_positions,3.0)
        old_velx = self.velocities[:,0]
        old_charges = self.charges[:]
        self.types = ['CT','N','O','C']
        new_positions = []
        new_typeid = []
        last_id = 0
        beads = []
        new_velx = []
        new_charges = []

        print("Placing atoms ...")
        for i_bead ,position in enumerate(old_positions):
            if(old_types[i_bead]==0):
                sc = sidechain(i_bead)
                last_id = sc.place(last_id,position)
                new_positions.extend(sc.positions)
                new_typeid.extend(sc.types)
                new_velx.extend(np.multiply(np.ones(4),old_velx[i_bead]))
                new_charges.extend(np.multiply(np.ones(4),old_charges[i_bead]))
                beads.append(sc)
            if(old_types[i_bead]==1):
                bb = backbone(i_bead)
                last_id = bb.place(last_id,position)
                new_positions.extend(bb.positions)
                new_typeid.extend(bb.types)
                new_velx.extend(np.multiply(np.ones(4),old_velx[i_bead]))
                new_charges.extend(np.multiply(np.ones(4),old_charges[i_bead]))
                beads.append(bb)
            if(old_types[i_bead]==2):
                cr = crosslinker(i_bead)
                last_id = cr.place(last_id,position)
                new_positions.extend(cr.positions)
                new_typeid.extend(cr.types)
                new_velx.extend(np.multiply(np.ones(3),old_velx[i_bead]))
                new_charges.extend(np.multiply(np.ones(3),old_charges[i_bead]))
                beads.append(cr)
            if(old_types[i_bead]==3):
                tj = t_junction(i_bead)
                last_id = tj.place(last_id,position)
                new_positions.extend(tj.positions)
                new_typeid.extend(tj.types)
                new_velx.extend(np.multiply(np.ones(8),old_velx[i_bead]))
                new_charges.extend(np.multiply(np.ones(8),old_charges[i_bead]))
                beads.append(tj)
        self.positions = new_positions
        self.typeid = new_typeid
        new_velocities = np.zeros((len(new_positions),3))
        # new_velocities[:,0] = new_velx
        new_velocities[:,0] = new_charges
        self.velocities = new_velocities
        self.charges = new_charges
        print("######    1->2 (Primitive) : Finished    ######")
        print("Number of atoms : " + str(len(self.positions)))
        print("MSD is taken from charges at lvl1 and carried to both at velx and charges at lvl2")

    def get_snap(self,context):
        with context:
            #print(self.coarse_grain_level)
            #probably don't need the if level == 3 here because it is never and input (ie no lvl4)
            if self.coarse_grain_level==0:
                snap = make_snapshot(N=len(self.positions),
                                     particle_types=self.types,
                                     bond_types=self.bond_types,
                                     box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))
            if self.coarse_grain_level==1:
                snap = make_snapshot(N=len(self.positions),
                                     particle_types=self.types,
                                     bond_types=self.bond_types,
                                     angle_types=self.angle_types,
                                     box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))

                # set angle typeids and groups
                snap.angles.resize(len(self.angle_group))
                for k in range(len(self.angle_group)):
                    snap.angles.typeid[k] = self.angle_typeid[k]
                    snap.angles.group[k] = self.angle_group[k]

            if self.coarse_grain_level==2:
                snap = make_snapshot(N=len(self.positions),
                                     particle_types=self.types,
                                     bond_types=self.bond_types,
                                     angle_types=self.angle_types,
                                     dihedral_types=self.dihedral_types,
                                     pair_types=self.pair_types,
                                     box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))

                # set angle typeids and groups
                snap.angles.resize(len(self.angle_group))
                for k in range(len(self.angle_group)):
                    snap.angles.typeid[k] = self.angle_typeid[k]
                    snap.angles.group[k] = self.angle_group[k]

                # set angle typeids and groups
                snap.dihedrals.resize(len(self.dihedral_group))
                for k in range(len(self.dihedral_group)):
                    snap.dihedrals.typeid[k] = self.dihedral_typeid[k]
                    snap.dihedrals.group[k] = self.dihedral_group[k]

                # set specialpairs(4th neighbors)
                snap.pairs.resize(len(self.pair_group))
                for k in range(len(self.pair_group)):
                    snap.pairs.typeid[k] = self.pair_typeid[k]
                    snap.pairs.group[k] = self.pair_group[k]



            # set typeids and positions
            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
            # set bond typeids and groups
            snap.bonds.resize(len(self.bond_group))
            for k in range(len(self.bond_group)):
                snap.bonds.typeid[k] = self.bond_typeid[k]
                snap.bonds.group[k] = self.bond_group[k]
        return snap

    def dump_snap(self, outfile):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]
        snap.particles.charge = self.charges[:]

        # bonds
        snap.bonds.N = len(self.bond_group)
        snap.bonds.types  = self.bond_types[:]
        snap.bonds.typeid = self.bond_typeid[:]
        snap.bonds.group  = self.bond_group[:]
        # angles
        snap.angles.N = len(self.angle_group)
        snap.angles.types  = self.angle_types[:]
        snap.angles.typeid = self.angle_typeid[:]
        snap.angles.group  = self.angle_group[:]
        # dihedrals
        snap.dihedrals.N = len(self.dihedral_group)
        snap.dihedrals.types  = self.dihedral_types[:]
        snap.dihedrals.typeid = self.dihedral_typeid[:]
        snap.dihedrals.group  = self.dihedral_group[:]
        # pairs
        snap.pairs.N = len(self.pair_group)
        snap.pairs.types  = self.pair_types[:]
        snap.pairs.typeid = self.pair_typeid[:]
        snap.pairs.group  = self.pair_group[:]
        snap.particles.velocity = self.velocities[:]
        # save the configuration
        with gsd.hoomd.open(name=outfile, mode='wb') as f:
            f.append(snap)

    def level2_as_CG(self,context):
        with context:
            if self.coarse_grain_level==0:
                print("Nope")
                exit()
            elif self.coarse_grain_level==1:
                self.level2_as_coarse_grained()
                output_level = 2
            elif self.coarse_grain_level==2:
                print("Nope")
                exit()

            if(output_level==2):
                dist = np.linalg.norm(self.positions,axis=1)
                ind_p = np.unravel_index(np.argsort(dist,axis=None),dist.shape)
                ind_p = ind_p[0][::-1]
                maxdist = dist[ind_p[0]]
                self.Lx = maxdist*2.0 + 60.0
                self.Ly = maxdist*2.0 + 60.0
                self.Lz = maxdist*2.0 + 60.0
            snap = make_snapshot(N=len(self.positions),
                                 particle_types=self.types,
                                 bond_types=self.bond_types,
                                 angle_types=self.angle_types,
                                 box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz)
                                 )

            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]

            return snap,output_level



    def level2_as_coarse_grained(self):
        scale = 3.0
        old_positions = self.positions
        old_positions = np.multiply(old_positions,scale)
        self.positions = old_positions

    def dump_txt(self, outfile):
        positions = np.array(self.positions)
        particle_types = np.array(self.typeid)
        particle_types[particle_types==3]=0
        print(particle_types)
        print(positions)

        data = np.zeros((len(positions),5))
        data[:,0] = np.arange(len(positions))
        data[:,1] = particle_types
        data[:,2:] = positions[:,:]
        np.savetxt(outfile, data, fmt='%i %i %1.4f %1.4f %1.4f', header="Line Atom_type x y z -- 0-carbon 1-nitrogen 2-oxygen")

    def dump_txt_with_msd_at_velx(self, outfile):
        positions = np.array(self.positions)
        particle_types = np.array(self.typeid)
        particle_types[particle_types==3]=0
        msd = self.velocities[:,0]
        data = np.zeros((len(positions),6))
        data[:,0] = np.arange(len(positions))
        data[:,1] = particle_types
        data[:,2:5] = positions[:,:]
        data[:,5] = msd
        np.savetxt(outfile, data, fmt='%i %i %1.4f %1.4f %1.4f %1.4f', header="Line Atom_type x y z msd(length^2) -- 0-carbon 1-nitrogen 2-oxygen")

    def dump_txt_with_msd_at_charge(self, outfile):
        positions = np.array(self.positions)
        particle_types = np.array(self.typeid)
        particle_types[particle_types==3]=0
        msd = self.charges
        data = np.zeros((len(positions),6))
        data[:,0] = np.arange(len(positions))
        data[:,1] = particle_types
        data[:,2:5] = positions[:,:]
        data[:,5] = msd
        np.savetxt(outfile, data, fmt='%i %i %1.4f %1.4f %1.4f %1.4f', header="Line Atom_type x y z msd(length^2) -- 0-carbon 1-nitrogen 2-oxygen")
