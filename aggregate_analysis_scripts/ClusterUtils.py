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
from ClusterNetwork import Cluster
np.set_printoptions(suppress=True)

"""
VII : Very Important Info
"""


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

class ClusterUtils():
    """
    Mat 30 2022
    Originally was for mc_sim clusters (p_hetero/cluster)
    This one is the modified version for hoomd clusters so it is a little different
    (p_shear/cluster)

    """
    def __init__(self,input,frame):
        print("Reading in : " + input)
        self.read_system(input,frame)
        self.cluster_ids = [] ### unique cluster ids
        self.noise_id = 0
        self.cluster_N = []
        self.filename = input[:-4]
        self.full_filename = input
        self.frame = 0
        self.clusters = []

    def read_system(self,input,target_frame):
        """
        Read in a snapshot from a gsd file or snapshot.
        """
        self.target_frame = target_frame
        try:
            with gsd.hoomd.open(name=input, mode='rb') as f:
                if (target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    self.frame = target_frame
                    frame = f.read_frame(target_frame)
                    print("Reading frame ", target_frame)
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


    def remove_unattached(self,cutoff,min_samples):
        """
        Due to the way rras_v?.hoomd works if you want to make a cluster with
        70 shapes the code adds 71th before ending so here I remove that extra
        which is probably unattached, not sure what to do if it managed to be attached
        """
        pos_dbscan = self.positions + self.box[0]*0.5
        tree = KDTree(data=pos_dbscan, leafsize=12, boxsize=self.Lx+0.0001)
        pairs = tree.sparse_distance_matrix(tree,cutoff+1.0)
        cutoff_eff = cutoff
        if(cutoff_eff<1.0):
            print("Small NOM -- so cutoff = 1.0 is used for DBSCAN")
            cutoff_eff = 1.0
        dbscan = DBSCAN(eps=cutoff_eff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        labels_shape = labels0[self.typeid==1]
        unq_lbl_shape,counts = np.unique(labels_shape,return_counts=True)
        if(len(unq_lbl_shape)==1):
            print("There is only one cluster so not deleting anything")
        elif(len(unq_lbl_shape)>2.5):
            print("There can't be more than 2 clusters of shapes !!!!")
            exit()
        else:
            if(counts[1]>counts[0]):
                print("Why first cluster is not the largest???!!!")
                exit()
            print("There are 2 clusters of shapes : ")
            print("1st with %d beads"%counts[0])
            print("2nd with %d beads"%counts[1])
            print("Deleting the 2nd one")
            cluster_id_to_remove = unq_lbl_shape[1]
            ### this is necessary since central particle is probably not in the cluster
            ### and needs to be removed as well
            body_to_remove = self.body[labels0==cluster_id_to_remove]
            body_to_remove = np.max(body_to_remove)
            index_to_remove = np.where(self.body==body_to_remove)
            index_to_remove = index_to_remove[0]
            self.delete_indexes(index_to_remove)
            if(len(index_to_remove)>250):
                print("Error 500T")

    def read_bonds(self):
        try:
            with gsd.hoomd.open(name=self.full_filename, mode='rb') as f:
                if (self.target_frame==-1):
                    frame = f.read_frame(len(f)-1)
                    self.frame = len(f)-1
                    print("Reading last frame ")
                else:
                    frame = f.read_frame(self.target_frame)
                    print("Reading frame ", self.target_frame)

                self.bond_group = (frame.bonds.group).copy()

        except:
            self.bond_group = (input.bonds.group).copy()


    def add_bodies(self):
        """
        Sep 26 2022
        Will use this class for bond lifetime analysis on deformation but this
        class was made for rigid bodies aggregation. For it to work I will
        assign bodies to particles offff
        """
        self.read_bonds()
        G1 = nx.Graph()
        ids = np.arange(len(self.typeid))
        bonds = self.bond_group
        G1.add_nodes_from(ids)
        G1.add_edges_from(bonds)
        con = nx.connected_components(G1)
        body_index = 0
        for m in con:
            if(len(m)>5):
                m = list(m)
                self.body[m] = body_index
                body_index += 1
        return self.body

    def assign_bodies(self,bodies):
        """
        I want to use smae bodies since I want to track the lifetime of bonds
        """
        self.body = bodies

    def delete_indexes(self,indexes):
        """
        There are lots of properties, attributes for particles so removing some
        particles takes a lot of time. Not to repeat I wrote a function.
        """
        print("Deleting %d particles " %len(indexes))
        self.positions = np.delete(self.positions,indexes,axis=0)
        self.velocities = np.delete(self.velocities,indexes,axis=0)
        self.typeid = np.delete(self.typeid,indexes)
        self.body = np.delete(self.body,indexes)
        self.moment_inertia = np.delete(self.moment_inertia,indexes,axis=0)
        self.orientations = np.delete(self.orientations,indexes,axis=0)
        self.mass = np.delete(self.mass,indexes)
        self.angmom = np.delete(self.angmom,indexes,axis=0)

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

    def check_overlap(self):
        pos_dbscan = self.positions + self.Lx*0.5
        tree = KDTree(pos_dbscan,boxsize=self.Lx+0.0001)
        pairs = tree.query_pairs(r=0.6)
        if(len(pairs)>0):
            print("There is overlap")
            exit()

    def remove_dummy(self,dummy_type):
        if(dummy_type == -1):
            print("No dummy particles specified")
        else:
            print("removing particles of type %d" %dummy_type)
            self.positions = self.positions[self.typeid!=dummy_type]
            self.velocities = self.velocities[self.typeid!=dummy_type]
            self.moleculeid = self.moleculeid[self.typeid!=dummy_type]
            self.typeid = self.typeid[self.typeid!=dummy_type]

    def dbscan(self,cutoff,min_samples):
        """
        Must be careful with noms making clusters so I only consider clusters
        with at least 10 beads

        """
        self.cutoff = cutoff
        isNoNoise = False
        pos_dbscan = self.positions + self.Lx*0.5
        tree = KDTree(data=pos_dbscan, leafsize=12, boxsize=self.Lx+0.0001)
        pairs = tree.sparse_distance_matrix(tree,cutoff+1.0)
        cutoff_eff = cutoff
        if(cutoff_eff<1.0):
            print("Small NOM -- so cutoff = 1.0 is used for DBSCAN")
            cutoff_eff = 1.0
        dbscan = DBSCAN(eps=cutoff_eff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        self.cluster_ids = labels0
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        if -1 not in labels0 :
            isNoNoise=True
            print("There is no noise !!!")
            exit()
        cluster_ids,cluster_count = np.unique(labels0,return_counts=True)
        cluster_ids = cluster_ids[cluster_count>30]
        cluster_count = cluster_count[cluster_count>30]
        if(len(cluster_ids)>2):
            print("There are more than 2 clusters !!!")
            print(cluster_count)
            print(cluster_ids)
            ### VII This exit was commented out on Sep 27 2022
            ### to run bond_lifetime_v1.py. For any other scripts it should be
            ### turned on
            # exit()

        target_cluster_id = cluster_ids[cluster_ids>-0.5]
        target_cluster_id = target_cluster_id[0]
        self.construct_cluster(target_cluster_id)


    def construct_cluster(self,cl_id):
        """
        Make the cluster, there is only one
        """
        pos = self.positions[self.cluster_ids==cl_id]
        type = self.typeid[self.cluster_ids==cl_id]
        body = self.body[self.cluster_ids==cl_id]
        c = Cluster(pos,type,body,self.Lx,self.cutoff)
        c.construct_network()
        c.calculate_neighbors()
        c.calculate_edge_frequency()
        c.calculate_edge_strength()
        c.calculate_size() #VII : uncomment this, this was commented to run
        # bond_lifetime_v1.py to calculate bond number evolution during deformation
        # and we don't need the cluster size during this. Calculating size may cause
        # cluster can't fit to box error

        ## The maesure_gyration_tensor.py also does dbscan on every single frame of
        # deformation gsds but it is using the ClusterUtils class from Breakup
        # that is why it never errors out. But for bonds we need the more detailed
        # ClusterUtils which is this script.
        self.cluster = c

    def setShapeId(self,shape_id):
        self.cluster.setShapeId(shape_id)

    def getCluster(self):
        return self.cluster

    def calculate_pair_distance(self,rp):
        """
        For fractal dimension calculation see script fractal_dimension.py

        Rp decribed in paper A simple model for the structure of fractal aggregates
        Since my shapes arenot spherical, Rp is somewhat arbitrary. Here I go to the
        largest aggregate and calculate all the distances btw neighboring shapes
        by distance I mean the distance between the central particles of those shapes.

        """
        dist = self.cluster.calculate_pair_distance(rp)
        return dist

    def get_pair_distance_distribution(self):
        """
        No average, all connected distances
        """
        dists = self.cluster.get_pair_distance_distribution()
        return dists

    def get_fake_df(self):
        df = self.cluster.get_fake_df()
        return df

    def box_counting(self):
        self.cluster.box_counting()

    def getAngles(self):
        """
        used by angles_all.py for final aggregates made with rigid
        """
        angles = self.cluster.getAngles()
        return angles

    def getAngles2(self):
        """
        used by angle_acf_v1.py to record angles during mpcd deformation
        """
        triplets,angles = self.cluster.getAngles2()
        return triplets,angles

    def getBonds(self):
        """
        called from cluster/bond_lifetime_v1.py, see that script
        """
        bonds = self.cluster.getBonds()
        return bonds

    def dump_mp_aggregate(self,outname):
        """
        July 11, used in dump_mp_aggregate.py
        """
        dist = self.cluster.calculate_pair_distance(-1)
        mp_pos = self.cluster.get_mp_aggregate()
        com = np.average(mp_pos,axis=0)
        mp_pos = mp_pos - com
        mp_typeids = np.zeros(len(mp_pos))
        mp_types = ['ko']
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Lx, self.Lx, 0, 0, 0]

        # particles
        snap.particles.N = len(mp_pos)
        snap.particles.position = mp_pos[:]
        snap.particles.types  = mp_types[:]
        snap.particles.typeid = mp_typeids[:]


        # save the configuration
        with gsd.hoomd.open(name=outname, mode='wb') as f:
            f.append(snap)

    def dump_all_aggregate(self,outname):
        """
        July 19, used in dump_all_aggregate.py
        """
        pos = self.cluster.get_pos()
        com = np.average(pos,axis=0)
        pos = pos - com
        typeids = np.zeros(len(pos))
        types = ['so']
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Lx, self.Lx, 0, 0, 0]

        # particles
        snap.particles.N = len(pos)
        snap.particles.position = pos[:]
        snap.particles.types  = types[:]
        snap.particles.typeid = typeids[:]


        # save the configuration
        with gsd.hoomd.open(name=outname, mode='wb') as f:
            f.append(snap)

    def dump_debug1(self,outname,amk):
        outname = outname[:-4] + "_body.gsd"
        snap = gsd.hoomd.Snapshot()
        snap.configuration.box = [self.Lx, self.Lx, self.Lx, 0, 0, 0]
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]
        snap.particles.charge = self.body[:]

        if(amk==0):
            with gsd.hoomd.open(name=outname, mode='xb') as f:
                f.append(snap)
        else:
            with gsd.hoomd.open(name=outname, mode='rb+') as f:
                f.append(snap)

############################################# REST NOT USED ###############################################





    def remove_overlap(self):
        """
        for the sphere that had overlaps not needed normally
        """
        tree = KDTree(self.positions)
        pairs = tree.query_pairs(r=0.03)
        delete_rows = []

        for pair in pairs:
            delete_rows.append(pair[1])

        print("%d overlaps found." %len(delete_rows))
        N_old = len(self.positions)
        self.positions = np.delete(self.positions,delete_rows,0)
        self.velocities = np.delete(self.velocities,delete_rows,0)
        self.moleculeid = np.delete(self.moleculeid,delete_rows)
        self.typeid = np.delete(self.typeid,delete_rows)
        print("%d overlaps removed "%(N_old - len(self.positions)))


    def aps(self):
        sizes = []
        for c_id in self.cluster_ids[1:]:
            pos = self.positions[self.velocities[:,0]==c_id]
            typeid = self.typeid[self.velocities[:,0]==c_id]
            moleculeid = self.moleculeid[self.velocities[:,0]==c_id]
            ids = np.where(self.velocities[:,0]==c_id)

            unique_mol_ids, counts = np.unique(moleculeid,return_counts=True)
            shape_mol_ids = unique_mol_ids[counts>5]
            sizes.append(len(shape_mol_ids))
        sizes = np.array(sizes,dtype=int)

        return([len(sizes),np.max(sizes)])


    def construct_clusters(self):
        """
        recreate individual clusters as instances of some other cluster class
        assumes there is noise for now
        """

        for c_id in self.cluster_ids[1:]:
            pos = self.positions[self.velocities[:,0]==c_id]
            if(len(pos)<1600): ## when min samples is set to 3 random 3 nom close
                continue     ## enough can make a cluster

            typeid = self.typeid[self.velocities[:,0]==c_id]
            moleculeid = self.moleculeid[self.velocities[:,0]==c_id]
            ids = np.where(self.velocities[:,0]==c_id)
            c = Cluster(pos,typeid,moleculeid,ids,self.box[0],self.cutoff)
            c.construct_network()
            c.calculate_neighbors()
            c.calculate_edge_frequency()
            c.calculate_edge_strength()
            # c.plot_cluster_network()
            # c.plot_multigraph()
            # exit()
            # c.calculate_edge_distance()
            self.clusters.append(c)
        # exit()

    def nematic(self):
        for i_c,c in enumerate(self.clusters):
            c.nematic()

    def plot_average_degree(self,save,show):
        avg_degrees = []
        sizes = []
        for i_c,c in enumerate(self.clusters):
            avg_degrees.append(c.average_degree)
            sizes.append(c.N_shape)
        plt.figure(1)
        plt.title("Average degree of each cluster of %s at frame %d" %(self.filename,self.frame))
        plt.scatter(sizes,avg_degrees)
        plt.xlabel("cluster size (N)")
        plt.ylabel("Average degree")
        if(save==1):
            plt.savefig("./cluster_connectivity/average_degree_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def get_avg_degrees(self):
        avg_degrees = []
        for i_c,c in enumerate(self.clusters):
            avg_degrees.append(c.average_degree)
        return avg_degrees

    def get_edge_freqs(self):
        edge_freqs = []
        for i_c,c in enumerate(self.clusters):
            edge_freqs.append(c.edge_frequency)
        return edge_freqs

    def get_edge_str_max_avg(self):
        edge_str_max_avg = []
        for i_c,c in enumerate(self.clusters):
            edge_str_max_avg.append(c.avg_edg_str_max)
        return edge_str_max_avg

    def get_edge_str_min_avg(self):
        edge_str_min_avg = []
        for i_c,c in enumerate(self.clusters):
            edge_str_min_avg.append(c.avg_edg_str_min)
        return edge_str_min_avg

    def get_edge_str_mid_avg(self):
        edge_str_mid_avg = []
        for i_c,c in enumerate(self.clusters):
            edge_str_mid_avg.append(c.avg_edg_str_mid)
        return edge_str_mid_avg

    def get_cluster_sizes_N(self):
        sizes = []
        for i_c,c in enumerate(self.clusters):
            sizes.append(c.N_shape)
        return sizes

    def get_N_clusters(self):
        return(len(self.clusters))

    def get_frame_index(self):
        return self.frame

    def get_a_mol_id(self):
        """
        so that I can identify it in the snapshot
        """
        a_mol_id = []
        for i_c,c in enumerate(self.clusters):
            a_mol_id.append(c.moleculeids[0])
        return(a_mol_id)

    def find_cluster(self,seed_id):
        seed_mol_id = seed_id
        self.moleculeid = np.array(self.moleculeid,dtype=np.int)
        mss = np.where(self.moleculeid==seed_id)
        mss = mss[0]
        seed_id = mss[0]
        box = np.array([self.Lx,self.Lx,self.Lx])
        target_cluster = self.velocities[seed_id,0]
        N_target_cluster = self.velocities[seed_id,1]
        print("%d particles are in the target cluster" %N_target_cluster)
        target_ids = np.where(self.velocities[:,0]==target_cluster)
        target_types = self.typeid[target_ids]
        types,type_counts = np.unique(target_types,return_counts=True)
        print("Count for types (last is NOM) : ",type_counts )
        target_pos = self.positions[target_ids] - self.box[0]*0.5
        ## Move molecule to the origin, I don't care about orientation
        ## since I will only do pmf vs distance
        relevant_ids = np.where(target_types==0)
        relevant_ids = relevant_ids[0]
        relevant_pos = target_pos[relevant_ids]
        cm = com(relevant_pos,box)
        target_pos = np.subtract(target_pos,cm)
        target_pos = wrap_pbc(target_pos,box)
        return target_pos,target_types

    def find_cluster_sphere(self,seed_id):
        """
        input seed id is intended to be mol id not particle id
        positions are all + at the moment
        """
        seed_mol_id = seed_id
        self.moleculeid = np.array(self.moleculeid,dtype=np.int)
        mss = np.where(self.moleculeid==seed_id)
        mss = mss[0]
        seed_id = mss[0]

        box = np.array([self.Lx,self.Lx,self.Lx])
        target_cluster = self.velocities[seed_id,0]
        N_target_cluster = self.velocities[seed_id,1]
        print("%d particles are in the target cluster" %N_target_cluster)
        target_ids = np.where(self.velocities[:,0]==target_cluster)
        target_types = self.typeid[target_ids]
        target_molids = self.moleculeid[target_ids]

        types,type_counts = np.unique(target_types,return_counts=True)
        print("Count for types (last is NOM) : ",type_counts )
        target_pos = self.positions[target_ids] - self.box[0]*0.5
        relevant_ids = np.where(target_types==0)
        relevant_ids = relevant_ids[0]
        relevant_pos = target_pos[relevant_ids]
        sphere_pos = self.positions[self.moleculeid==seed_mol_id]
        sphere_types = self.typeid[self.moleculeid==seed_mol_id]
        center_pos = sphere_pos[sphere_types==1]
        center_pos = center_pos - self.box[0]*0.5
        target_pos = np.vstack((target_pos,center_pos))
        cm = com(relevant_pos,box)
        target_pos = np.subtract(target_pos,cm)
        target_pos = wrap_pbc(target_pos,box)
        target_types = np.hstack((target_types,np.ones(len(center_pos))))
        return target_pos,target_types


    def cluster_molecules(self,N_mol):
        """
        used in gsd_cluster_frames.py - Detect and
        assign a cluster ID for each molecule-shape
        Be careful single particles are also molecules I only want the big guys
        Molecule ids are in charge, particle cluster ids are in self.velocities[:,0]
        """
        unique_molecule_ids, molecule_counts = np.unique(self.moleculeid,return_counts=True)
        cluster_ids = self.velocities[:,0]
        unique_molecule_ids = unique_molecule_ids[molecule_counts>2]
        if(len(unique_molecule_ids)!=N_mol):
            print("Number of molecules can't change in a simulation")
            exit()
        unique_molecule_ids = np.sort(unique_molecule_ids)
        result = []
        for i_m,m_id in enumerate(unique_molecule_ids):
            p_cluster_ids = cluster_ids[self.moleculeid==m_id]
            unique_p_cluster_ids = np.unique(p_cluster_ids)
            if(len(unique_p_cluster_ids)>1.5):
                print("A molecule cannot be in two different clusters!!")
                exit()
            else:
                current_id = int(unique_p_cluster_ids[0])
            result.append(current_id)

        if(N_mol!=len(result)):
            print("error 23")
            exit()

        result = np.array(result)
        return result


    def getNmol(self):
        """
        Set the # of large molecules to make sure it is the same at all frames
        """
        unique_molecule_ids, molecule_counts = np.unique(self.moleculeid,return_counts=True)
        unique_molecule_ids = unique_molecule_ids[molecule_counts>2]
        return len(unique_molecule_ids)


    def size_analysis(self):
        ### calculating end-to-end size (ie longest distance) of a cluster
        ### if cluster is very larger than half the box size the size measurement
        ### will be wrong

        ### when calculating fractal dimensions the void inside the shapes should
        ### be accounted for but how ?

        ### Assumes sigma = 1 for every bead not - always true
        ### Assumes the largest cluster is smaller than half the box size

        ### Not clear how to calculate fractal dimension
        self.cluster_size = np.zeros_like(self.cluster_ids,dtype=float)
        self.cluster_f_dim = np.zeros_like(self.cluster_ids,dtype=float)
        for i,c_id in enumerate(self.cluster_ids):
            if (c_id!=self.noise_id):
                pos = self.positions[self.velocities[:,0]==c_id]
                tree = KDTree(data=pos, leafsize=12, boxsize=self.box[0])
                pairs = tree.sparse_distance_matrix(tree,np.sqrt(3)*self.box[0]*0.5)
                pairs = pairs.toarray()
                pairs = pairs.flatten()
                max_r = np.amax(pairs)*0.5
                N = self.cluster_N[i]
                d_fractal = np.log(N*((0.5)**3))/np.log(max_r)
                self.cluster_size[i] = max_r
                self.cluster_f_dim[i] = d_fractal

    def size_analysis_multibox(self):
        ### explicitly put image boxes for the cluster
        ### do a dbscan for that configuration
        ### the largest cluster should be the original one but without
        ### crossing any boundaries so you can calculate the distance directly
        ### if the largest cluster is bigger than the original one than there is
        ### percolation which we shouldn't have in any scenario anyway
        self.cluster_size = np.zeros_like(self.cluster_ids, dtype=float)
        self.cluster_f_dim = np.zeros_like(self.cluster_ids, dtype=float)
        for i,c_id in enumerate(self.cluster_ids):
            if (c_id!=self.noise_id):
                pos_0 = self.positions[self.velocities[:,0]==c_id]
                pos = pos_0
                for ii in range(-1,2):
                    for jj in range(-1,2):
                        for kk in range (-1,2):
                            if(ii==0 and jj==0 and kk==0):
                                pass
                            else:
                                image = np.array([ii,jj,kk])
                                pos = np.vstack((pos,np.add(pos_0,np.multiply((np.ones_like(pos_0)*self.box[0]),image[np.newaxis,:]))))
                tree = KDTree(data=pos, leafsize=12)
                pairs = tree.sparse_distance_matrix(tree,self.cutoff+1.0)
                dbscan = DBSCAN(eps=self.cutoff, min_samples=self.min_samples, metric="precomputed", n_jobs=-1)
                labels0 = dbscan.fit_predict(pairs)
                n0,cluster_count = np.unique(labels0,return_counts=True)
                max_count = np.amax(cluster_count)
                if(max_count>self.cluster_N[i]):
                    print("Check snapshot for percolation!!!!")
                    print("Multibox cluster_N : %d " %max_count)
                    print("Original cluster_N : %d " %self.cluster_N[i])
                    exit()
                elif(max_count==self.cluster_N[i]):
                    safe_pos = pos[labels0==n0[np.argmax(cluster_count)]]
                    distances = distance.pdist(safe_pos)
                    max_r = np.amax(distances)*0.5
                    N = self.cluster_N[i]
                    d_fractal = np.log(N*((0.5)**3))/np.log(max_r)
                    self.cluster_size[i] = max_r
                    self.cluster_f_dim[i] = d_fractal
                else:
                    print("error 163")
                    exit()
        self.cluster_size = self.cluster_size[self.cluster_size!=0]
        self.cluster_f_dim = self.cluster_f_dim[self.cluster_f_dim!=0]

    def plot_N_histogram(self,save,show):
        cluster_N_wo_noise = self.cluster_N[1:]
        hist, edges = np.histogram(cluster_N_wo_noise) #range=(0,220))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(1)
        plt.title("Cluster N histogram of %s at frame %d - total # %d" %(self.filename,self.frame,len(self.typeid)))
        bar_width = (np.amax(self.cluster_N)/len(hist))*0.2
        plt.bar(center,hist, width=(edges[1] - edges[0])*0.8)
        plt.xlabel("N_paricle")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/cluster_N_hist_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_fraction_histogram(self,save,show):
        cluster_N_sorted = np.flip(np.sort(self.cluster_N[1:]))
        hist, edges = np.histogram(cluster_N_sorted) #range=(0,220))
        bin_sums = np.zeros(len(hist))
        for i in range(0,len(hist)):
            current_clusters = cluster_N_sorted[cluster_N_sorted>edges[i]]
            current_clusters = current_clusters[current_clusters<=edges[i+1]]
            bin_sums[i] = np.sum(current_clusters)
        bin_fractions = np.divide(bin_sums,len(self.positions))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(2)
        plt.title("\n".join(wrap("fraction of the particles in the given cluster size range %s at frame %d - noise %.3f " %(self.filename,self.frame,1.0-np.sum(bin_fractions)), 60)))
        bar_width = (np.amax(self.cluster_N)/len(hist))*0.2
        plt.bar(center,bin_fractions, width=(edges[1] - edges[0])*0.7)
        # plt.plot(np.linspace(0,len(hist)+1,num=10000),np.zeros(10000) )#,marker='o',c='r',s=20.0)
        plt.xlabel("N_particle")
        plt.ylabel("fractions")
        if(save==1):
            plt.savefig("./cluster_gsd/fractions_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_size_histogram(self,save,show):
        hist, edges = np.histogram(self.cluster_size) #range=(0,220))
        center = (edges[1:] + edges[:-1])/2
        plt.figure(3)
        plt.title("Cluster size histogram of %s at frame %d" %(self.filename,self.frame))
        plt.bar(center,hist,width=(edges[1] - edges[0])*0.7)#,marker='o',c='r',s=20.0)
        plt.xlabel("max_r/\u03C3")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/cluster_size_hist_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    # title = ax.set_title("\n".join(wrap("fraction of the particles in the given cluster size range %s at frame %d - noise %.3f " %(self.filename,self.frame,1.0-np.sum(bin_fractions)), 60)))
    def plot_fractal_dim_histogram(self,save,show):
        hist, edges = np.histogram(self.cluster_f_dim,bins=10)
        center = (edges[1:] + edges[:-1])/2
        plt.figure(4)
        plt.title("Fractal dimension histogram")
        bar_width = ((np.amax(self.cluster_f_dim) - np.amin(self.cluster_f_dim)) / len(hist))*0.2
        plt.bar(center,hist,width=(edges[1] - edges[0])*0.7)#,marker='o',c='r',s=20.0)
        plt.xlabel("df")
        plt.ylabel("cluster count")
        if(save==1):
            plt.savefig("./cluster_gsd/fractal_dim_histogram_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()

    def plot_2d_scatter_size_vs_fractaldim(self,save,show):
        plt.figure(5)
        plt.title("fractal dim vs cluster size scatter of %s at frame %d" %(self.filename,self.frame))
        plt.scatter(self.cluster_size,self.cluster_f_dim)#,marker='o',c='r',s=20.0)
        plt.xlabel("size")
        plt.ylabel("fractal dim")
        plt.ylim(0.0,3.0)
        if(save==1):
            plt.savefig("./cluster_gsd/fractal_dim_vs_cluster_size_%s_frame_%d_.pdf" %(self.filename,self.frame))
        if(show==1):
            plt.show()


    def dump_snap_w_cluster_info_at_vel(self, outfile):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]

        snap.particles.velocity = self.velocities[:]

        # save the configuration
        with gsd.hoomd.open(name=outfile, mode='wb') as f:
            f.append(snap)

### ---------------- NO NEED BELOW ----------------------------- ####

    def color_by_position(self): ### exampe function

        velocities = np.zeros_like(self.positions)
        velocities[:,:] = np.sign(self.positions)
        self.velocities = velocities

    def cr_pair_correlation(self,bins=50,r_max=10.0):
        rdf = freud.density.RDF(bins=bins, r_max=r_max)
        rdf.compute(system=(self.box, self.positions[self.typeid==2]), reset=False)
        plt.figure(1)
        plt.plot(rdf.bin_centers,rdf.rdf)
        plt.show()

    def cr_avg_dist_density(self):
        positions = self.positions[self.typeid==2]
        origin = np.array([[0.0,0.0,0.0]])
        dists_to_origin = distance.cdist(origin,positions)
        max_dist = np.max(dists_to_origin[0])
        radius = 0.75 * max_dist ### better measure
        positions = positions[dists_to_origin[0]<radius]
        avg_dist = np.power(len(positions)/((4.0/3.0)*(3.14)*(radius**3)),-1.0/3.0  )
        return(avg_dist)


    def histogram_cubic(self,bin_size):
        positions = self.positions[self.typeid!=3]
        bin_edges = edges_for_cubic_histogram(positions,bin_size)
        H, edges = np.histogramdd(positions, bins = bin_edges)
        edges=np.array(edges)
        bin_volume = (edges[0][1]-edges[0][0])*(edges[1][1]-edges[1][0])*(edges[2][1]-edges[2][0])
        print("Bin edges : %.2f %.2f %.2f" %((edges[0][1]-edges[0][0]),(edges[1][1]-edges[1][0]),(edges[2][1]-edges[2][0])))
        print("Bin volume : ",  bin_volume)
        return H,edges

    def color_by_crosslinker_cluster(self,cutoff,min_samples):
        ## two points that are closer thn cutoff distance are considered to be
        ## in the same cluster
        ### if I understand correctly : to say that a group of points is a cluster
        ### there neds to be at least min_samples points (if min sample is 1, two
        ### points very close to each other but far from everything else is a cluster
        ### but if min_samp. is 3 they are not cluster but interpreted as noise(-1
        ### for the cluster label)    )

        ### get the crosslinker positions
        tree = KDTree(data=self.positions[self.typeid==2], leafsize=12)
        pairs = tree.sparse_distance_matrix(tree,cutoff)
        # dbscan = DBSCAN(eps=cutoff, min_samples=1, metric="precomputed", n_jobs=-1)
        dbscan = DBSCAN(eps=cutoff, min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = dbscan.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters : ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        # print(cluster_count)
        cluster_info = np.ones_like(self.positions[self.typeid==2])
        cluster_info = np.multiply(cluster_info,999)
        #labels0[labels0==0]=7

        cluster_info[:,0] = labels0

        cluster_info[:,1] = cluster_count[labels0+1]
        ### +15 below is to create a color contrast btw nonclusters(when they are -1)
        ### some clusters cnt be separated visually with ease



        cluster_info[cluster_info==-1]=len(cluster_count + 5)
        self.velocities[self.typeid==2] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))

    def color_by_monomer_cluster_MEANSHIFT(self,cutoff):
        print("Calculating tree ...")
        #tree = KDTree(data=self.positions[self.typeid!=3], leafsize=12)
        #pairs = tree.sparse_distance_matrix(tree,cutoff)
        data=self.positions[self.typeid!=3]
        # data=data[data[:,0]>0]
        # data=data[data[:,1]>0]
        # data=data[data[:,2]>0]
        meanshift = MeanShift(bandwidth=4.0, cluster_all=False, bin_seeding=True, n_jobs=-1)
        print("Calculating clusters ...")
        labels0 = meanshift.fit_predict(data)
        print(labels0)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters : ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        cluster_info = np.ones_like(data)
        cluster_info = np.multiply(cluster_info,999)
        cluster_info[:,0] = labels0
        cluster_info[:,1] = cluster_count[labels0]
        self.velocities[self.typeid!=3] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))

    def color_by_crosslinker_cluster_OPTICS(self,min_samples):
        ### OPTICS is similar to DBSCAN I just try it to s if it is better or not

        tree = KDTree(data=self.positions[self.typeid==2], leafsize=12)
        pairs = tree.sparse_distance_matrix(tree,100)
        pairs = pairs.toarray()
        optics = OPTICS(min_samples=min_samples, metric="precomputed", n_jobs=-1)
        labels0 = optics.fit_predict(pairs)
        n_clusters0 = len(set(labels0)) - (1 if -1 in labels0 else 0)
        n0,cluster_count = np.unique(labels0,return_counts=True)
        print("Detected clusters (OPTICS): ", len(cluster_count)-1)
        print(np.flip(np.sort(cluster_count[1:])))
        cluster_info = np.ones_like(self.positions[self.typeid==2])
        cluster_info = np.multiply(cluster_info,999)
        cluster_info[:,0] = labels0
        cluster_info[:,1] = cluster_count[labels0]
        self.velocities[self.typeid==2] = cluster_info
        return (np.flip(np.sort(cluster_count[1:])))



    def get_snap(self,context):
        with context:


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

            for k in range(len(self.positions)):
                snap.particles.position[k] = self.positions[k]
                snap.particles.typeid[k] = self.typeid[k]
            # set bond typeids and groups
            snap.bonds.resize(len(self.bond_group))
            for k in range(len(self.bond_group)):
                snap.bonds.typeid[k] = self.bond_typeid[k]
                snap.bonds.group[k] = self.bond_group[k]

        return snap



    def append_snap(self, outfile):
        ### if the gsd file exist append as the next frame, useful for example
        ### if you want to cluster with different cutoff values and pick best later
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]\

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
        with gsd.hoomd.open(name=outfile, mode='rb+') as f:
            f.append(snap)


    # def __del__(self):
    #     print ("deleted")

def edges_for_cubic_histogram(positions,bin_size):
    ### not specifying the bin size (using N_bins instead) might result in slightly different
    ### bin volumes and end result is very sensitive to it. So one should spec. it
    ### clearly

    x_up,x_down = np.max(positions[:,0]),np.min(positions[:,0])
    x_down = x_down - 3.0*bin_size -0.1
    kk = x_down
    while (kk<x_up + 3.0*bin_size + 0.1):
        kk = kk + bin_size
    x_up = kk
    n_x = int(( x_up - x_down  ) // bin_size)
    x_edges = np.linspace(x_down,x_up,num=n_x+1)

    y_up,y_down = np.max(positions[:,1]),np.min(positions[:,1])
    y_down = y_down - 3.0*bin_size -0.1
    kk = y_down
    while (kk<y_up + 3.0*bin_size + 0.1):
        kk = kk + bin_size
    y_up = kk
    n_y = int(( y_up - y_down  ) // bin_size)
    y_edges = np.linspace(y_down,y_up,num=n_y+1)

    z_up,z_down = np.max(positions[:,2]),np.min(positions[:,2])
    z_down = z_down - 3.0*bin_size -0.1
    kk = z_down
    while (kk<z_up + 3.0*bin_size + 0.1):
        kk = kk + bin_size
    z_up = kk
    n_z = int(( z_up - z_down  ) // bin_size)
    z_edges = np.linspace(z_down,z_up,num=n_z+1)

    edges = [x_edges,y_edges,z_edges]
    return edges

def histogram_cleaner(H):
    ### this function is used at cubic bin histogrammers
    ### this histogram has a lot of zeros and I dont really want to take periphery into account
    ### so I will in the end reduce it to a 1d array with all cubes having only nonzero
    ### neighbors.
    ### Deleting the neighborhood of a zero is not the best approach because there are
    ### zeros in the core part and they should b taken into account. If I delete their
    ### neighborhood I lose valuable info so i will first scan for 0s that are embedded
    ### in microgel (ie all neighbrs are non zero) and mark them as -1 so that I dont
    ### delete their neighborhood
    # H.setflags(write=1)
    internal_zeros = list()
    with np.nditer(H[1:-1,1:-1,1:-1],flags=['multi_index'],op_flags=['readwrite']) as it:
        for x in it:
            if(x==0):
                index = np.array(it.multi_index)
                index = np.add(index,1)
                neighborhood = H[index[0]-1:index[0]+2,index[1]-1:index[1]+2,index[2]-1:index[2]+2]
                neighborhood = neighborhood.flatten()
                n_zeros = (27 - np.count_nonzero(neighborhood)) - 1
                if (n_zeros < 3.5): ### if there is one or more zero in theneigh. it is internal
                    internal_zeros.append(index)

    for index in internal_zeros:
        H[index[0],index[1],index[2]] = -1

    neighbor_to_zero = list()
    with np.nditer(H[1:-1,1:-1,1:-1],flags=['multi_index'],op_flags=['readwrite']) as it:
        for x in it:
            index = np.array(it.multi_index)
            index = np.add(index,1)
            neighborhood = np.array([H[index[0]+1,index[1],index[2]], H[index[0]-1,index[1],index[2]]])
            neighborhood = np.append(neighborhood,[H[index[0],index[1]+1,index[2]], H[index[0],index[1]-1,index[2]]])
            neighborhood = np.append(neighborhood,[H[index[0],index[1],index[2]+1], H[index[0],index[1],index[2]-1]])
            neighborhood = neighborhood.flatten()
            if 0 in neighborhood:
                neighbor_to_zero.append(index)

    for index in neighbor_to_zero:
        H[index[0],index[1],index[2]] = 0

    H = H.flatten()
    H = H[H!=0]
    H = np.where(H==-1, 0,H)
    return(H)

def histogram_cleaner_gaus(H):
    ### same as above but for Gaussian density bins
    ### so the rules for deleting periphery is more strict
    ### internal zeros is still here but it is unlikely to have any for Gaussian
    internal_zeros = list()
    with np.nditer(H[1:-1,1:-1,1:-1],flags=['multi_index'],op_flags=['readwrite']) as it:
        for x in it:
            if(x==0):
                index = np.array(it.multi_index)
                index = np.add(index,1)
                neighborhood = H[index[0]-1:index[0]+2,index[1]-1:index[1]+2,index[2]-1:index[2]+2]
                neighborhood = neighborhood.flatten()
                n_zeros = (27 - np.count_nonzero(neighborhood)) - 1
                if (n_zeros < 3.5): ### if there is one or more zero in theneigh. it is internal
                    internal_zeros.append(index)

    if(len(internal_zeros)>0):
        print("Internal zeros for Gaus ??  ")

    for index in internal_zeros:
        H[index[0],index[1],index[2]] = -1

    neighbor_to_zero = list()
    with np.nditer(H[1:-1,1:-1,1:-1],flags=['multi_index'],op_flags=['readwrite']) as it:
        for x in it:
            index = np.array(it.multi_index)
            index = np.add(index,1)
            neighborhood = np.array([H[index[0]+1,index[1],index[2]], H[index[0]-1,index[1],index[2]]])
            neighborhood = np.append(neighborhood,[H[index[0],index[1]+1,index[2]], H[index[0],index[1]-1,index[2]]])
            neighborhood = np.append(neighborhood,[H[index[0],index[1],index[2]+1], H[index[0],index[1],index[2]-1]])
            neighborhood = neighborhood.flatten()
            if 0 in neighborhood:
                neighbor_to_zero.append(index)

    for index in neighbor_to_zero:
        H[index[0],index[1],index[2]] = 0

    return(H)

class ExtractedMolecule():
    """
    Molecule you extract from the snapshot
    Bonds currently dont matter so I just put 2 bonds for each molecule here
    So that the simulation doesn't give an error (it shouldn't anyway)
    """
    def __init__(self,pos,types):
        self.positions = pos
        self.types = ['A','B','C']
        self.typeid = types
        self.Lx = 20.0
        self.Ly = 20.0
        self.Lz = 20.0
        self.Nbonds = 2
        self.bond_group = [[0,1],[1,2]]
        self.bond_typeid = [0,0]
        self.Nbondtypes = 1
        self.RemoveOverlaps()

    def RemoveOverlaps(self):
        """
        AQ'nun sphere'i
        """
        tree = KDTree(self.positions)
        pairs = tree.query_pairs(r=0.03)
        print("!!!!!!!!!!!!!!!!!!!!!!!!! OVERLAPS : @@@@@@@@@@@@@@@@@@@@@@@@")
        to_delete = []
        for pair in pairs:
            to_delete.append(pair[1])
        print("Deleting %d overlapped beads " %len(to_delete))
        self.positions = np.delete(self.positions,to_delete,0)
        self.typeid = np.delete(self.typeid,to_delete)



    def dump_gsd(self,name):
        snap = gsd.hoomd.Snapshot()
        snap.configuration.step = 0
        snap.configuration.box = [self.Lx, self.Ly, self.Lz, 0, 0, 0]

        # particles
        snap.particles.N = len(self.positions)
        snap.particles.position = self.positions[:]
        snap.particles.types  = self.types[:]
        snap.particles.typeid = self.typeid[:]

        with gsd.hoomd.open(name=name, mode='wb') as f:
            f.append(snap)

    def dump_txt(self,name):
        fout = open(name,'w')
        fout.write("# LAMMPS-inspired data file to be used for MC code\n")
        fout.write("\n")
        fout.write("%d sites\n" %len(self.positions))
        fout.write("%d bonds\n" %self.Nbonds)
        fout.write("\n")
        fout.write("%d site types\n" %len(self.types))
        fout.write("%d bond types\n" %self.Nbondtypes)
        fout.write("\n")
        fout.write("Site Properties\n")
        fout.write("\n")
        ## these don't matter
        fout.write("0 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("1 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("2 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("\n")
        fout.write("Bond Properties\n")
        fout.write("\n")
        fout.write("0 length 1.0 delta 100.0 \n")
        fout.write("\n")
        fout.write("Sites\n")
        fout.write("\n")
        for i,p in enumerate(self.positions):
            fout.write("%d %d %.3f %.3f %.3f\n" %(i,self.typeid[i],p[0],p[1],p[2]))
        fout.write("\n")
        fout.write("Bonds\n")
        fout.write("\n")
        for i,b in enumerate(self.bond_group):
            fout.write("%d %d %d %d\n" %(i,self.bond_typeid[i],b[0],b[1]))


    def dump_txt_increment(self,name):
        fout = open(name,'w')
        fout.write("# LAMMPS-inspired data file to be used for MC code\n")
        fout.write("\n")
        fout.write("%d sites\n" %len(self.positions))
        fout.write("%d bonds\n" %self.Nbonds)
        fout.write("\n")
        fout.write("%d site types\n" %len(self.types))
        fout.write("%d bond types\n" %self.Nbondtypes)
        fout.write("\n")
        fout.write("Site Properties\n")
        fout.write("\n")
        ## these don't matter
        fout.write("3 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("4 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("5 epsilon 1.0 sigma 1.0 cutoff 3.0\n" )
        fout.write("\n")
        fout.write("Bond Properties\n")
        fout.write("\n")
        fout.write("0 length 1.0 delta 100.0 \n")
        fout.write("\n")
        fout.write("Sites\n")
        fout.write("\n")
        for i,p in enumerate(self.positions):
            fout.write("%d %d %.3f %.3f %.3f\n" %(i,self.typeid[i]+3,p[0],p[1],p[2]))
        fout.write("\n")
        fout.write("Bonds\n")
        fout.write("\n")
        for i,b in enumerate(self.bond_group):
            fout.write("%d %d %d %d\n" %(i,self.bond_typeid[i],b[0],b[1]))
