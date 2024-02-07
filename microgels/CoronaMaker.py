import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict
import os.path
import networkx as nx
import itertools
import random
# ------------------------------------------------------------------------------
#                              Helper functions
#-------------------------------------------------------------------------------
def random_point_on_cone(R,theta,prev):
    R"""
    Returns random vector with length R and angle
    theta between previos one, uniformly distributed.

    """
    #theta *=np.pi/180.
    v = prev/np.linalg.norm(prev)
    # find "mostly orthogonal" vector to prev
    a = np.zeros((3,))
    a[np.argmin(np.abs(prev))]=1
    # find orthonormal coordinate system {x_hat, y_hat, v}
    x_hat = np.cross(a,v)/np.linalg.norm(np.cross(a,v))
    y_hat = np.cross(v,x_hat)
    # draw random rotation
    phi = np.random.uniform(0.,2.*np.pi)
    # determine vector (random rotation + rotation with theta to guarantee the right angle between v,w)
    w = np.sin(theta)*np.cos(phi)*x_hat + np.sin(theta)*np.sin(phi)*y_hat + np.cos(theta)*v
    w *=R
    return w

def getPoint_in_shpere(R):
    """
    Returns a random point inside a sphere with radius R, uniformly distributed.
    """
    u = np.random.uniform(0,1)
    v = np.random.uniform(0,1)
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    r = (np.random.uniform(0,1))**(1./3.)*R
    sinTheta = np.sin(theta)
    cosTheta = np.cos(theta)
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    x = r * sinPhi * cosTheta
    y = r * sinPhi * sinTheta
    z = r * cosPhi
    return np.array([x,y,z])

def random_FRC(m,characteristic_ratio,bond_length,zz):
    """
    Returns a freely jointed chain with defined characteristic ratio and
    bond lenths. Particles can overlap.

    TODO: is characteristic_ratio correctly defined or is it 1/c ?

    Used to build random walk in 3D with correct characteristic ratio
    Freely rotating chain model - LJ characteristic ratio
    is 1.88 (http://dx.doi.org/10.1016/j.cplett.2011.12.040)
     c = 1+<cos theta>/(1-<cos theta>)
     <cos theta> = (c-1)/(c+1)

    """

    theta = np.arccos((characteristic_ratio-1)/(characteristic_ratio+1))
    coords = np.zeros((m,3))
    if(zz==0):
        coords[1]=[1,0,0]
    else:
        coords[1]=[-1,0,0]
        
    for i in range(2,m):
        prev = coords[i-2]-coords[i-1]
        n = random_point_on_cone(bond_length,theta,prev)
        new = coords[i-1]+n
        coords[i]=new

    return coords[1:]

class CoronaMaker():
    """

    """
    def __init__(self,input,target_frame):
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
        self.input_file = input ## can be snapshot too
        self.target_frame = target_frame


    def add_corona(self,distribution_parameters,corona_ratio,exclusion):
        self.check_consistency()
        ### exclusion is how far away a monomer converted should be from crosslinkers ends etc 
        ### add parameters for number and chain length distribution
        ### can end the added chains with normal monomer particle type (we know it is not causing any problems from forced free ends)
        ### distribution parameters : mean and variance (can convert to mu and sigma here)
        ### Need a parameter to set number of chains to add or the total bead number to add as new chains
        ### corona_ratio is btw 0-1 beads to add / beads before
        ### OR may just use N as the nuber of chains to add but the ratio seems more ideal
        N_core = len(self.positions) - np.count_nonzero(self.typeid==3)
        N_corona = int(np.round(N_core * corona_ratio))
        mu = distribution_parameters[0]
        sigma = distribution_parameters[1]
        chain_lengths = np.round(np.random.lognormal(mu, sigma, 100000))
        chain_lengths = chain_lengths[chain_lengths>4]
        chain_lengths = chain_lengths.astype(np.int)
        chain_lengths = chain_lengths[np.cumsum(chain_lengths)<N_corona]
        print("The core has " + str(N_core) + " particles. With this corona ratio("+str(corona_ratio)+"), " + str(N_corona) + "(approx) particles will be added as corona." )
        print("Average chain length : " + str(np.average(chain_lengths)) )
        print("Total number of chains to add : " + str(len(chain_lengths)))
        print("Exact number of particles to add : " +str(np.sum(chain_lengths)))
        #### pick the potential candidates for corona roots
        dist = np.linalg.norm(self.positions,axis=1)
        ind_p = np.unravel_index(np.argsort(dist,axis=None),dist.shape)
        ind_p = ind_p[0][::-1]


        bonds1 = self.bond_group
        all_ids1 = self.bond_group

        ids1 = list(np.unique((np.array(self.bond_group)).flatten()))
        particle_types1 = np.array(self.typeid)

        G1 = nx.Graph()
        G1.add_nodes_from(ids1)
        G1.add_edges_from(bonds1)

        bad_candidates = []
        ind_p = list(ind_p)
        for id in ind_p:
            is_bad = False
            if(particle_types1[id]!=1):
                is_bad = True
            if(particle_types1[id]!=3):    
                neighbors = list(nx.dfs_preorder_nodes(G1, source=id, depth_limit=exclusion))
            else:
                neighbors=[]
            #### The for loop below is just incase you want to build corona on a microgel that you've
            #### already forced free ends on(in this case ends might be type 1 and we need number of neighbors to tell them apart)
            for ne in neighbors:
                ns = [n for n in G1.neighbors(ne)]
                if(ns==1):
                    is_bad = True
            if(0 in particle_types1[neighbors] or 2 in particle_types1[neighbors]):
                is_bad = True
            if(is_bad==True):
                bad_candidates.append(id)


        for bad_candidate in bad_candidates:
            ind_p.remove(bad_candidate)

        ### Each root can be a tjunction or a crosslinker
        ### Here I assume 0.1 tj and 0.9 cr but this can be a parameter
        ### Or It can replicate the ratio in the core

        N_chain = len(chain_lengths)
        tj_ratio = 0.1
        N_root_cr = N_chain/(2.0+tj_ratio)
        N_root_cr = int(int(np.floor(N_root_cr*0.5))*2)
        N_root_tj = N_chain - N_root_cr*2
        print("TJ roots : " + str(N_root_tj) + " - CR roots : " + str(N_root_cr))
        N_root = N_root_tj + N_root_cr
        print("Exclusion is set to ", exclusion)
        ### Need to pick N_root points out from ind_p
        ### divide root candidates into 10 sub lists pick from first(if it is empty second)
        #### THe problem is we don't want to pick two roots that are closer than 4 points
        #### One can add dangerous sites and check if the ranomly pick index is in them or not
        #### Or one can remove them from ind_p so we won't pick them at all (but this can be costly)
        #### so instead we make the two lists so removing won't take that long
        outer1 = ind_p[0:int(np.floor(len(ind_p)*0.1))]
        outer2 = ind_p[int(np.floor(len(ind_p)*0.1)):int(np.floor(len(ind_p)*0.15))]
        new_bonds = []
        new_typeids = []
        new_positions = []
        new_bonds_typeid = []
        current_chain = 0
        index = len(self.positions)
        for cr in range(0,N_root_cr):
            if(len(outer1)>0.5):
                cr_ind = random.choice(outer1)
            elif(len(outer2)>0.5):
                cr_ind = random.choice(outer2)
            else:
                print("error 65")
                exit()
            self.typeid[cr_ind] = 2
            first_neighbors = [n for n in G1.neighbors(cr_ind)]
            self.typeid[first_neighbors] = 0
            neighborhood = list(nx.dfs_preorder_nodes(G1, source=cr_ind, depth_limit=exclusion))
            outer1 = [ele for ele in outer1 if ele not in neighborhood]
            outer2 = [ele for ele in outer2 if ele not in neighborhood]
            ### Build the chains
            for zz in range(0,2):
                new_bonds.append([cr_ind,index])
                new_bonds_typeid.append(0)
                new_positions.extend(random_FRC(chain_lengths[current_chain]+1,1.88,0.95,zz) + self.positions[cr_ind])
                new_typeids.append(0)
                for bead in range(0,chain_lengths[current_chain]-1):
                    new_bonds.append([index,index+1])
                    new_bonds_typeid.append(0)
                    index = index + 1
                    new_typeids.append(1)
                new_typeids[-1]=0
                index = index + 1
                current_chain = current_chain + 1
        for tj in range(0,N_root_tj):
            if(len(outer1)>0.5):
                tj_ind = random.choice(outer1)
            elif(len(outer2)>0.5):
                tj_ind = random.choice(outer2)
            else:
                print("error65")
                exit()
            self.typeid[tj_ind] = 2
            first_neighbors = [n for n in G1.neighbors(tj_ind)]
            self.typeid[first_neighbors] = 0
            neighborhood = list(nx.dfs_preorder_nodes(G1, source=tj_ind, depth_limit=exclusion))
            outer1 = [ele for ele in outer1 if ele not in neighborhood]
            outer2 = [ele for ele in outer2 if ele not in neighborhood]


            new_bonds.append([tj_ind,index])
            new_bonds_typeid.append(0)
            new_positions.extend(random_FRC(chain_lengths[current_chain]+1,1.88,0.95,0) + self.positions[tj_ind])
            new_typeids.append(0)
            for bead in range(0,chain_lengths[current_chain]-1):
                new_bonds.append([index,index+1])
                new_bonds_typeid.append(0)
                index = index + 1
                new_typeids.append(1)
            new_typeids[-1]=0
            index = index + 1
            current_chain = current_chain + 1


        print(str(len(new_positions))+ " new particles added.")
        print(str(current_chain) + " new chains added.")
        print(str(len(new_bonds))+ " new bonds added.")
        if(len(new_bonds)!=len(new_bonds_typeid)):
            print("error45")
        if(len(new_bonds)!=len(new_positions) or len(new_bonds)!=len(new_typeids) ):
            print("error55")
        if(len(new_bonds)!=index-len(self.positions)):
            print("error75")

        self.positions = np.vstack((self.positions,new_positions))
        self.typeid = np.hstack((self.typeid,new_typeids))
        self.bond_group =np.vstack((self.bond_group,new_bonds))
        self.bond_typeid = np.hstack((self.bond_typeid,new_bonds_typeid))

        self.bond_group,index = np.unique(self.bond_group, return_index=True,axis=0)
        self.bond_typeid = self.bond_typeid[index]

        
        print("Checking consistency ...")
        self.check_consistency()


    def return_snapshot(self):
        dist = np.linalg.norm(self.positions,axis=1)
        ind_p = np.unravel_index(np.argsort(dist,axis=None),dist.shape)
        ind_p = ind_p[0][::-1]
        maxdist = dist[ind_p[0]]
        self.Lx = maxdist*2.0 + 80.0
        self.Ly = maxdist*2.0 + 80.0
        self.Lz = maxdist*2.0 + 80.0

        snap = make_snapshot(N=len(self.positions),
                             particle_types=self.types,
                             bond_types=self.bond_types,
                             # angle_types=self.angle_types,
                             # dihedral_types=self.dihedral_types,
                             # pair_types=self.pair_types,
                             box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz)
                             )


        # set typeids and positions
        for k in range(len(self.positions)):
            snap.particles.position[k] = self.positions[k]
            snap.particles.typeid[k] = self.typeid[k]
        # set bond typeids and groups
        snap.bonds.resize(len(self.bond_group))
        for k in range(len(self.bond_group)):
            snap.bonds.typeid[k] = self.bond_typeid[k]
            snap.bonds.group[k] = self.bond_group[k]
        # set angle typeids and groups
        # snap.angles.resize(len(self.angle_group))
        # for k in range(len(self.angle_group)):
        #     snap.angles.typeid[k] = self.angle_typeid[k]
        #     snap.angles.group[k] = self.angle_group[k]
        # # set dihedral typeids and groups
        # snap.dihedrals.resize(len(self.dihedral_group))
        # for k in range(len(self.dihedral_group)):
        #     snap.dihedrals.typeid[k] = self.dihedral_typeid[k]
        #     snap.dihedrals.group[k] = self.dihedral_group[k]
        # # set pairs typeids and groups
        # snap.pairs.resize(len(self.pair_group))
        # for k in range(len(self.pair_group)):
        #     snap.pairs.typeid[k] = self.pair_typeid[k]
        #     snap.pairs.group[k] = self.pair_group[k]

        return snap


    def read_system(self):
        try:
            print("Reading in : " + self.input_file)
        except:
            print("Reading a snapshot not gsd file")
        print("Reading the last frame ")
        input=self.input_file

        """
        Read in a snapshot from a gsd file or snapshot.
        """
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                ### t_frame is irrelevent for this one I only want the last frame
                frame = f.read_frame(len(f)-1)
                self.positions = (frame.particles.position).copy()
                self.velocities = (frame.particles.velocity).copy()

                self.types = (frame.particles.types).copy()
                self.typeid = (frame.particles.typeid).copy()

                self.bond_types = (frame.bonds.types).copy()
                self.bond_group = (frame.bonds.group).copy()
                self.bond_typeid = (frame.bonds.typeid).copy()

                # self.angle_types = (frame.angles.types).copy()
                # self.angle_group = (frame.angles.group).copy()
                # self.angle_typeid = (frame.angles.typeid).copy()
                #
                # self.dihedral_types = (frame.dihedrals.types).copy()
                # self.dihedral_group = (frame.dihedrals.group).copy()
                # self.dihedral_typeid = (frame.dihedrals.typeid).copy()
                #
                # self.pair_types = (frame.pairs.types).copy()
                # self.pair_group = (frame.pairs.group).copy()
                # self.pair_typeid = (frame.pairs.typeid).copy()

                self.Lx,self.Ly,self.Lz = frame.configuration.box[0:3]

        except:
            self.positions = (input.particles.position).copy()
            self.velocities = (input.particles.velocity).copy()
            self.types = (input.particles.types).copy()
            self.typeid = (input.particles.typeid).copy()

            self.bond_types = (input.bonds.types).copy()
            self.bond_group = (input.bonds.group).copy()
            self.bond_typeid = (input.bonds.typeid).copy()

            # self.angle_types = (input.angles.types).copy()
            # self.angle_group = (input.angles.group).copy()
            # self.angle_typeid = (input.angles.typeid).copy()
            #
            # self.dihedral_types = (input.dihedrals.types).copy()
            # self.dihedral_group = (input.dihedrals.group).copy()
            # self.dihedral_typeid = (input.dihedrals.typeid).copy()
            #
            # self.pair_types = (input.pairs.types).copy()
            # self.pair_group = (input.pairs.group).copy()
            # self.pair_typeid = (input.pairs.typeid).copy()

            self.Lx = input.box.Lx
            self.Ly = input.box.Lx
            self.Lz = input.box.Lx

        N_D = np.count_nonzero(np.array(self.typeid) == 3)
        if(N_D==0):
            print("Unconnected parts are not yet neutralized(ie microgel unswollen), choose a later frame")
            exit()



    # def remove_dummy_particles(self):
    #     # remove dummy bonds first
    #     self.bond_types = [self.bond_types[0]]
    #     self.bond_group = self.bond_group[self.bond_typeid!=1]
    #     self.bond_typeid = self.bond_typeid[self.bond_typeid!=1]
    #     # shift all indices in bond groups down
    #     if np.all((self.positions[self.typeid==3] == 0)) and np.all((self.velocities[self.typeid==3]== 0)):
    #         deleted = 0
    #         for i,t in enumerate(self.typeid):
    #             if t==3:
    #                 self.bond_group[self.bond_group>i-deleted] -=1
    #                 deleted += 1
    #     # remove dummy particles
    #     self.types = ['A','B','C']
    #     self.positions=self.positions[self.typeid!=3]
    #     self.velocities=self.velocities[self.typeid!=3]
    #     self.typeid=self.typeid[self.typeid!=3]


    def check_consistency(self):
        bonds = self.bond_group
        all_ids = self.bond_group
        ids = list(np.unique((all_ids).flatten()))
        particle_types = self.typeid


        G = nx.Graph()
        G.add_nodes_from(ids)
        G.add_edges_from(bonds)

        N_monomer = np.count_nonzero(particle_types == 1)
        N_crosslinker = np.count_nonzero(particle_types == 2)
        N_end = np.count_nonzero(particle_types == 0)

        crosslinkers = []
        ends = []
        monomers = []

        for i in ids:
            if(particle_types[i]==2):
                crosslinkers.append(i)
            if(particle_types[i]==0):
                ends.append(i)
            if(particle_types[i]==1):
                monomers.append(i)

        for monomer in monomers:
            neighbors = [n for n in G.neighbors(monomer)]
            # if(len(neighbors)!=2):
            #     print("A monomer doesn't have 2 neighbors")
            #     print(particle_types[neighbors])
            #     print(neighbors)
            #     exit()
            if(2 in particle_types[neighbors]):
                print("A monomer is bonded to crosslinker")
                exit()

        for crosslinker in crosslinkers:
            neighbors = [n for n in G.neighbors(crosslinker)]
            if(len(neighbors)>4.5):
                print("A crosslinker has more than 4 neighbors")
                exit()
            if(2 in particle_types[neighbors] or 1 in particle_types[neighbors]):
                print("Wrong bonding for a crosslinker")
                exit()

        for end in ends:
            neighbors = [n for n in G.neighbors(end)]
            if(len(neighbors)>2.5):
                print("An end has more than 2 neighbors")
                exit()
            # if(0 in particle_types[neighbors]):
            #     print("An end bonded to another end")
            #     print(particle_types[neighbors])
            #     print(neighbors)
            #     print(end)
            #     exit()


    def get_snap(self):
        #with context:
            #print(self.coarse_grain_level)
            #probably don't need the if level == 3 here because it is never and input (ie no lvl4)
        snap = make_snapshot(N=len(self.positions),
                             particle_types=self.types,
                             bond_types=self.bond_types,
                             box=boxdim(Lx=self.Lx,Ly=self.Ly,Lz=self.Lz))


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
        # bonds
        snap.bonds.N = len(self.bond_group)
        snap.bonds.types  = self.bond_types[:]
        snap.bonds.typeid = self.bond_typeid[:]
        snap.bonds.group  = self.bond_group[:]

        # save the configuration
        with gsd.hoomd.open(name=outfile, mode='wb') as f:
            f.append(snap)
