import numpy as np
import gsd.hoomd
from hoomd.data import make_snapshot, boxdim
from collections import defaultdict
import random
import networkx as nx

def connected_components(lists):
    """
    merges lists with common elements

    args: bonds from configuration (frame.bonds.group)

    returns: list of connected particles by id

    Useful for finding bonded particles in a configuration, tested for
    linear polymers with consecutive bonds (0-1-2-3-4-5, 6-7-8-9-10,..)
    and non consecutive ids ( 0-5-6-8-10, 1-4-3-2-9,...).
    Works with ints as well as str. Is rather slow for big connected networks,
    networkx is faster.

    """
    neighbors = defaultdict(set)
    seen = set()
    for each in lists:
        for item in each:
            neighbors[item].update(each)
    def component(node, neighbors=neighbors, seen=seen, see=seen.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            see(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield sorted(component(node))

def random_point_on_cone(R,theta,prev):
    """
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


def random_FRC(m,characteristic_ratio,bond_length):
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
    coords[1]=[1,0,0]
    for i in range(2,m):
        prev = coords[i-2]-coords[i-1]
        n = random_point_on_cone(bond_length,theta,prev)
        new = coords[i-1]+n
        coords[i]=new

    return coords


class SphereMicrogelMixtureMaker():
    """

    Initializes a mixture of chains with different lengths and crosslinkers in
    a spherical cavity.

    Particles/chains overlap.

    """
    def __init__(self, R,rho,f,distribution_parameters,free_end_params,is_angle):
        self.R = R
        self.rho = rho
        self.f = f
        self.Npol = np.int(np.round(rho*4*np.pi/3*R**3))
        self.mu = distribution_parameters[0]
        self.sigma = distribution_parameters[1]
        self.types = ['A','B','C','D']
        self.bond_types = ['bond','dummybond']
        self.free_end_length = free_end_params[0]
        self.free_end_rate = free_end_params[1]
        self.is_angle = is_angle
        self.angle_types = ['angle', 'dummyangle']

        self.set_system()

    def set_system(self):
        """
        Setup the system.

        TODO: simplify/devide into set_bonds,set_types,set_positions (maybe)

        """
        # draw chain lengts from a gamma distribution
        mu, sigma = self.mu, self.sigma
        s = np.round(np.random.lognormal(mu, sigma, 1000000))
        s = s[s>1]
        s = s.astype(np.int)
        s = s[np.cumsum(s)<self.Npol]

        bonds = []
        pts = []
        B = 0
        i = 0
        while B < self.Npol:
            c = s[i]
            for a in range(c):
                bonds.append([B+a,B+1+a])
            i += 1
            B += c+1
            Q = random_FRC(c+1,1.88,0.95)
            A = getPoint_in_shpere(self.R-2.0)
            pts.append(Q+A)

        self.N_crosslink = np.int(np.round(2*i*self.f))
        self.Ntot =  self.Npol + self.N_crosslink

        bonds = np.asarray(bonds)
        bonds = bonds[bonds[:,0]<self.Npol]
        bonds = bonds[bonds[:,1]<self.Npol]
        self.bond_group = np.asarray(bonds, dtype=np.int32)
        self.bond_typeid = np.zeros(len(bonds), dtype=np.int32)


        pts = np.vstack(pts)
        pts = pts - np.average(pts,axis=0)

        crosslinker_pts = np.random.uniform(-self.R,self.R,size=((self.N_crosslink)*5,3))
        crosslinker_pts = crosslinker_pts[np.linalg.norm(crosslinker_pts,axis=1)<self.R]

        self.positions = np.vstack((pts[:self.Npol],crosslinker_pts[:self.N_crosslink]))

        self.type_list = ['A']*self.Ntot
        map_types = {t:i for i, t in enumerate(self.types)}
        self.typeid = np.array([map_types[t] for t in self.type_list], dtype=np.int32)


        begin_crosslinkers=bonds[-1,1]+1
        self.typeid[begin_crosslinkers:-1]=2

        all_bonds,counts = np.unique(bonds.flatten(),return_counts=True)
        middle_chains = all_bonds[counts==2]
        self.typeid[middle_chains]=1


        if(self.is_angle==1):
            ## To add angles : Find the beads with to neighbors and put the angle
            print('----------------------- ADDING ANGLES -------------------------')

            G1 = nx.Graph()
            G1.add_nodes_from(list(np.unique((self.bond_group).flatten())))
            G1.add_edges_from(self.bond_group)

            angles = []
            for ii, bead in enumerate(middle_chains):
                neighbors = [n for n in G1.neighbors(bead)]
                if (len(neighbors)!=2):
                    print('Error - middle bead with not 2 neighbors') ## just a security check
                    exit()
                angles.append([neighbors[0],bead,neighbors[1]])
            self.angle_group = np.asarray(angles, dtype=np.int32)
            self.angle_typeid = np.zeros(len(angles), dtype=np.int32)

        ## below moves chains that are far(R+2.5) outside of the confinement inside
        ## since it takes time for them to move in and they fly out of the box
        ## this is especially problematic with angles

        beads_out = list()

        for i_bead,bead in enumerate(self.positions):
            dist = bead[0]**2 + bead[1]**2 + bead[2]**2
            if(dist>(self.R+2.5)**2 ):
                beads_out.append(i_bead)
        print("Beads outside the confinement : ", len(beads_out))

        ## detect beads of chains that has beads outside
        chain_beads_out = list()
        for bead in beads_out:
            to_add = list(nx.dfs_preorder_nodes(G1, source=bead))
            chain_beads_out.extend(to_add)

        chain_beads_out = np.array(chain_beads_out)
        chain_beads_out = np.unique(chain_beads_out)

        ## move the beads
        for bead in chain_beads_out:
            x = np.sign(self.positions[bead,0])*(-15.0) + self.positions[bead,0]
            y = np.sign(self.positions[bead,1])*(-15.0) + self.positions[bead,1]
            z = np.sign(self.positions[bead,2])*(-15.0) + self.positions[bead,2]
            self.positions[bead,:] = [x,y,z]

        print("Total beads moved inside : ", len(chain_beads_out))

        if(self.free_end_length < 99.0):
            ### Forced free ends :
            print("-------------------- Forcing free ends with Length : " + str(self.free_end_length) + " Rate : " + str(self.free_end_rate) + " ---------------")
            G = nx.Graph()
            G.add_nodes_from(list(np.unique((self.bond_group).flatten())))
            G.add_edges_from(self.bond_group)

            A = sorted(nx.connected_components(G), key = len, reverse=True)

            for chain in A:
                if (len(chain)>self.free_end_length):
                    l_chain = list(chain)
                    if (self.free_end_rate > np.random.rand()):
                        ## find the ends of the chain
                        ## l_chain[0] and l_chain[-1] won't work
                        chain_ends = []
                        for bead in l_chain:
                            neighbors = [n for n in G.neighbors(bead)]
                            if(len(neighbors)==1):
                                chain_ends.append(bead)
                        if(len(chain_ends)!=2):
                            print("Error in SphereMicrogelMixtureMaker - 22")
                        if(random.choice([True, False])):
                            self.typeid[chain_ends[0]] = 1
                        else:
                            self.typeid[chain_ends[1]] = 1
        else:
            print("----------------- No Forced Free Ends --------------")



    def get_snap(self, context):
        with context:
            snap = make_snapshot(N=self.Ntot,
                                 particle_types=self.types,
                                 bond_types=self.bond_types,
                                 angle_types=self.angle_types,
                                 box=boxdim(Lx=2*self.R+100,Ly=2*self.R+100,Lz=2*self.R+100))

    	    # set typeids and positions
            for k in range(self.Ntot):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
            # set bond typeids and groups
            snap.bonds.resize(len(self.bond_group))
            for k in range(len(self.bond_group)):
                snap.bonds.typeid[k] = self.bond_typeid[k]
                snap.bonds.group[k] = self.bond_group[k]
            snap.angles.resize(len(self.angle_group))
            for k in range(len(self.angle_group)):
                snap.angles.typeid[k] = self.angle_typeid[k]
                snap.angles.group[k] = self.angle_group[k]

        return snap

    def dump_snap(self, outfile):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.L, self.L, self.L, 0, 0, 0]
        snap.particles.position = self.positions[:]
        # types
        snap.particles.N = self.Ntot
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
