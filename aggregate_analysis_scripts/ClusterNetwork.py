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
from scipy.spatial import distance
np.set_printoptions(suppress=True)

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

def calc_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def pbc(pos,L):
    """
    Don't use, i think it doesn't work.
    """
    pos = pos - ((pos - L*0.5)//L + 1)*L
    return pos

def calc_angle_pbc(a,b,c,lx):
    ba = a - b
    bc = c - b
    # ba = pbc(ba,lx)
    # bc = pbc(bc,lx)
    ba = wrap_pbc(ba,lx)
    bc = wrap_pbc(bc,lx)
    n_ba = np.linalg.norm(ba)
    n_bc = np.linalg.norm(bc)
    if(n_ba>12 or n_bc>12):
        print("Error 5N1K")
        print(n_ba)
        print(n_bc)
        exit()
    cosine_angle = np.dot(ba, bc) / (n_ba * n_bc)
    angle = np.arccos(cosine_angle)
    return angle

class Cluster():
    """
    Originally was for mc_sim clusters (p_hetero/cluster)
    This one is the modified version for hoomd clusters so it is a little different
    (p_shear/cluster)
    """
    def __init__(self,pos,type,body,Lx,cutoff):
        self.pos = pos
        self.type = type
        self.body = body
        self.Lx = Lx
        self.cutoff = cutoff
        self.N_shape = 0


    def construct_network(self):
        """
        Shapes are nodes, bridging NOMs are edges
        """
        unique_body, counts = np.unique(self.body[self.body>-0.5],return_counts=True)
        self.N_shape = len(unique_body)
        # print(unique_body)
        print(self.N_shape)
        nom_ids = np.where(self.body==-1)
        nom_ids = nom_ids[0]
        G_multi = nx.MultiGraph()
        G_regular = nx.Graph()
        self.nodes = unique_body
        G_multi.add_nodes_from(unique_body)
        G_regular.add_nodes_from(unique_body)
        self.pos_tree = self.pos + self.Lx*0.5
        tree = KDTree(data=self.pos_tree,boxsize=self.Lx+0.0001)
        edges = []
        edge_weights = []
        for nom in nom_ids:
            pos_nom = self.pos_tree[nom]
            nearby = tree.query_ball_point(pos_nom,r=self.cutoff)
            # nearby = nearby[0]
            # print(nearby)
            nearby = np.array(nearby)
            mask = np.isin(self.body[nearby],unique_body)
            nearby = nearby[mask]
            nearby_bodies = self.body[nearby]
            unique_nearby_bodies, nearby_counts = np.unique(nearby_bodies,return_counts=True)
            if(len(unique_nearby_bodies)>1.5):
                for i,id1 in enumerate(unique_nearby_bodies):
                    for j,id2 in enumerate(unique_nearby_bodies):
                        if(j>i):
                            edges.append([id1,id2])
                            edge_weights.append([nearby_counts[i],nearby_counts[j]])

        G_regular.add_edges_from(edges)
        G_multi.add_edges_from(edges)
        self.edge_weights = edge_weights
        self.unique_edges,self.edge_counts = np.unique(edges,return_counts=True,axis=0)
        self.G_regular = G_regular
        self.G_multi = G_multi

    def calculate_neighbors(self):
        """
        each shape is on average bridged to self.average_degree other shapes
        """
        node_degrees = self.G_regular.degree
        total_degree = 0
        for node in node_degrees:
            total_degree = node[1] + total_degree
        self.average_degree = total_degree/self.N_shape

    def calculate_edge_frequency(self):
        """
        A bridge from a shape to another on average has "self.edge_frequency" noms
        """
        self.edge_frequency = np.average(self.edge_counts)

    def calculate_edge_strength(self):
        """
        Each bridging NOM is an edge, each NOM bridges by attracting n1 beads from
        shape 1 and n2 beads from shape 2, more beads means stronger edge
        """
        self.edge_str_max = np.max(self.edge_weights,axis=1)
        self.edge_str_min = np.min(self.edge_weights,axis=1)
        self.edge_str_mid = (self.edge_str_max + self.edge_str_min)*0.5

        self.avg_edg_str_max = np.average(self.edge_str_max)
        self.avg_edg_str_min = np.average(self.edge_str_min)
        self.avg_edg_str_mid = np.average(self.edge_str_mid)

    def calculate_size(self):
        """
        Largest distance btw any two points in the cluster
        First check the boundary conditions, the way we run the simulations sort
        of guarantees this check but anyways

        1 - center cluster
        2 - check
        """

        self.centerCluster2()
        good,disp = self.checkClusterAfter()
        tt = 0
        while(good is False):
            tt += 1
            if(tt>1000):
                print("Cluster can't fit to the box !!!!")
                exit()
            self.shiftCluster(disp)
            good,disp = self.checkClusterAfter()

        ### now positions can be used independent of the periodic boundary conditions
        dists = distance.pdist(self.pos)
        self.max_dist = np.max(dists)

    def checkClusterAfter(self):
        """
        Checks if the extremes of the cluster are too close to boundaries or not
        Also check that distance from the cm of the cluster to the extremes of the
        cluster is less than half the box size This makes sure that cluster can be
        translated noicely
        """
        pos_cluster = self.pos
        x = np.array([np.min(pos_cluster[:,0]),np.max(pos_cluster[:,0])])
        y = np.array([np.min(pos_cluster[:,1]),np.max(pos_cluster[:,1])])
        z = np.array([np.min(pos_cluster[:,2]),np.max(pos_cluster[:,2])])
        walls = np.array([self.Lx*(-0.5), self.Lx*(0.5)])
        dists = []
        dists.append(x[0]-walls[0])
        dists.append(walls[1]-x[1])
        dists.append(y[0]-walls[0])
        dists.append(walls[1]-y[1])
        dists.append(z[0]-walls[0])
        dists.append(walls[1]-z[1])
        min_dist = np.min(dists)

        good = True
        disp = False
        disps = np.array([
        [1.0,0.0,0.0],
        [-1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,-1.0,0.0],
        [0.0,0.0,1.0],
        [0.0,0.0,-1.0],
        ])

        if(min_dist<0.7):
            direction = np.argmin(dists)
            disp = disps[direction]
            print("Cluster will be shifted by :")
            print(disp)
            good = False

        return good,disp

    def centerCluster2(self):
        """
        1 - Find the COM of the cluster
        2 - Instead of translating cluster and redistributing the noms just
        tranlate everything in the box
        """

        cluster_cm = com(self.pos,self.Lx)
        new_pos = self.pos - cluster_cm
        new_pos = wrap_pbc(new_pos,self.Lx)
        self.pos = new_pos

    def getBonds(self):
        return self.unique_edges

    def getAngles2(self):
        """
        October 11, called by ClusterUtils.py when called by angle_acf_v1.py
        I don't know why I'm writing the 2nd function to get the bonds.
        I will directly use unique edges to recreate a graph and get the
        angles from there. Not the best way but the other function was for aggregation
        simulations so I can't be sure that it will work well.
        """
        self.assign_central_pos()
        G = nx.Graph()
        # nodes = np.unique(self.unique_edges.flatten())
        G.add_nodes_from(self.nodes)
        """
        nodes vs self.nodes :
        October 13 bug
        When the shapes are bonded with 2 noms to the cluster that shape is actually
        counted in the cluster by dbscan so it is given to this class as part of
        the cluster but in the "construct_network" function i construct the graph
        with edges of single NOM as it should be so the shape that is only connected
        by a 2 noms is left out. To fix that difference here I take the nodes
        from dbscan bodies and edges from my graph network construction
        """

        G.add_edges_from(self.unique_edges)

        N_nodes = len(self.nodes)
        triplets = []
        for start in self.nodes:
            for end in self.nodes:
                paths = list(nx.all_simple_paths(G,start,end,cutoff=2))
                for path in paths:
                    if(len(path)==3):
                        triplets.append(path)

        triplets_np = np.array(triplets)

        angles = []
        for triplet in triplets:
            p1 = self.central_pos[self.nodes==triplet[0]][0]
            p2 = self.central_pos[self.nodes==triplet[1]][0]
            p3 = self.central_pos[self.nodes==triplet[2]][0]
            angle = calc_angle_pbc(p1,p2,p3,self.Lx)
            angles.append(angle)


        angles = np.array(angles)
        unique_angles,angle_index = np.unique(angles, return_index=True)
        unique_triplets = triplets_np[angle_index,:]
        return unique_triplets,unique_angles



    def shiftCluster(self,disp):
        """
        Move everything a little so that cluster doesn't hit the boundaries
        """

        new_pos = self.pos + disp*np.random.rand()*3.0
        new_pos = wrap_pbc(new_pos,self.Lx)
        self.pos = new_pos

    def setShapeId(self,id):
        self.shape_id = id

    def setSimId(self,id):
        self.sim_id = id

    def getInfo(self):
        """
        Each row in the output corresponds to one cluster, each cluster is described by
        # simulation_id shape_id MaxDimension N_shape Avg-Degree EdgeFreq. Edge-Str(min mid max)
        """
        arr = np.array([
        self.sim_id,
        self.shape_id,
        self.max_dist,
        self.N_shape,
        self.average_degree,
        self.edge_frequency,
        self.avg_edg_str_min,
        self.avg_edg_str_mid,
        self.avg_edg_str_max
        ])
        return arr

    def assign_central_pos(self):
        box = np.array([self.Lx,self.Lx,self.Lx])
        unique_body = np.unique(self.body[self.body>-0.5])
        central_pos = []
        for ub in unique_body:
            pos_b = self.pos[self.body==ub]
            cm0 = com(pos_b,box)
            # com = np.average(pos_b,axis=0)
            central_pos.append(cm0)
        central_pos = np.array(central_pos)

        self.central_pos = central_pos



    def calculate_pair_distance(self,rp):
        """
        Average distance btw central particles of connected shapes for fractal dimension
        calculation
        """
        self.assign_central_pos()
        dists = []
        for edge in self.unique_edges:
            p1 = self.central_pos[unique_body==edge[0]]
            p2 = self.central_pos[unique_body==edge[1]]
            d = np.linalg.norm(p1-p2)
            dists.append(d)


        d_avg = np.average(dists)
        self.effective_radius = d_avg*0.5
        if(rp>0.0):
            self.effective_radius = rp

        all_dists = distance.pdist(central_pos)
        max_r = np.amax(all_dists)*0.5 + self.effective_radius
        N_beads  = len(central_pos)
        self.fake_df = np.log(N_beads*(self.effective_radius**3))/np.log(max_r)

        return self.effective_radius

    def get_pair_distance_distribution(self):
        """
        Average distance btw central particles of connected shapes for fractal dimension
        calculation
        """

        unique_body = np.unique(self.body[self.body>-0.5])
        central_pos = []
        for ub in unique_body:
            pos_b = self.pos[self.body==ub]
            com = np.average(pos_b,axis=0)
            # cm0 = com(pos0,box)
            central_pos.append(com)
        central_pos = np.array(central_pos)

        self.central_pos = central_pos
        dists = []
        for edge in self.unique_edges:
            p1 = central_pos[unique_body==edge[0]]
            p2 = central_pos[unique_body==edge[1]]
            d = np.linalg.norm(p1-p2)
            dists.append(d)

        dists = np.array(dists)

        return dists


    def get_mp_aggregate(self):
        return self.central_pos

    def get_fake_df(self):
        return self.fake_df

    def getAngles(self):
        """
        find every triple of central points
        """
        self.calculate_pair_distance(3.0)

        nodes = list(self.G_regular.nodes)
        N_nodes = len(nodes)
        triplets = []
        for start in nodes:
            for end in nodes:
                paths = list(nx.all_simple_paths(self.G_regular,start,end,cutoff=2))
                for path in paths:
                    if(len(path)==3):
                        triplets.append(path)


        angles = []
        for triplet in triplets:
            p1 = self.central_pos[nodes==triplet[0]][0]
            p2 = self.central_pos[nodes==triplet[1]][0]
            p3 = self.central_pos[nodes==triplet[2]][0]
            angle = calc_angle(p1,p2,p3)
            angles.append(angle)

        angles = np.array(angles)
        angles = np.unique(angles) # get rid of double counts
        return angles

    def get_pos(self):
        """
        Call this after centering the cluster otherwise trouble
        """
        return  self.pos


    def box_counting(self):
        pos = self.central_pos
        pos = pos/self.effective_radius

        box_x = np.array([np.max(pos[:,0])+0.5,np.min(pos[:,0])-0.5])
        box_y = np.array([np.max(pos[:,1])+0.5,np.min(pos[:,1])-0.5])
        box_z = np.array([np.max(pos[:,2])+0.5,np.min(pos[:,2])-0.5])

        bb = 0.05

        bin_x = np.linspace(box_x[1],box_x[0],num=int((box_x[0]-box_x[1])/bb)  )
        bin_y = np.linspace(box_y[1],box_y[0],num=int((box_y[0]-box_y[1])/bb)  )
        bin_z = np.linspace(box_z[1],box_z[0],num=int((box_z[0]-box_z[1])/bb)  )

        # print(box_x)
        # print(box_y)
        # print(box_z)
        # exit()

        t0 = time.time()
        points = []
        for x in bin_x:
            for y in bin_y:
                for z in bin_z:
                    points.append([x,y,z])

        points = np.array(points)

        pos_tree = KDTree(data=pos)
        grid_tree = KDTree(data=points)
        nn = grid_tree.query_ball_point(pos,r=0.5)
        counted_box = []
        for n in nn:
            counted_box.extend(n)
        counted_box = np.array(counted_box)
        unique_counted = np.unique(counted_box)
        N = len(unique_counted)
        df = np.log(N)/np.log(1.0/bb)
        print(len(unique_counted))
        print(df)
        exit()


        n_full_box = 0
        for n in nn:
            if len(n) > 0:
                n_full_box +=1
        tf = time.time()
        print(tf-t0)

        print(n_full_box)
        print(len(nn))
        print(len(points))
        exit()
        # x, y, z = np.meshgrid(bin_x, bin_y, bin_z)
        #
        # print(np.shape(x))
        # print(np.shape(y))
        # print(np.shape(z))
        #
        # print(x-y)

        x = np.arange(2)
        y = np.arange(3)
        z = np.arange(4)
        xx, yy, zz = np.meshgrid(x, y, z)
        # print(np.shape(xx))
        # print(np.shape(yy))
        # print(np.shape(zz))
        # print(xx)
        # print(yy)
        print(zz)

        exit()


############################################# rest not used ###############################################


    def calculate_edge_distance(self):
        """
        Calculate the average distance between the center of masses of each bridged
        shape pair and average it
        """
        box = np.array([self.L,self.L,self.L])
        dists = []
        for i,edge in enumerate(self.unique_edges):
            pos0 = self.pos[self.moleculeids==edge[0]]
            pos1 = self.pos[self.moleculeids==edge[1]]
            cm0 = com(pos0,box)
            cm1 = com(pos1,box)
            dists.append(np.linalg.norm(cm0-cm1))

        dists = np.array(dists)
        dists = np.sort(dists)
        print(dists)


    def plot_cluster_network(self):
        # subax1 = plt.subplot(121)
        # groups = set(nx.get_node_attributes(self.G_regular,'group').values())
        # mapping = dict(zip(sorted(groups),itertools.count()))
        # nodes = self.G_regular.nodes()
        # colors = [mapping[self.G_regular.nodes[n]['group']] for n in nodes]
        pos = nx.spring_layout(self.G_regular)
        print(len(pos))
        # ec = nx.draw_networkx_edges(self.G_regular, pos, alpha=0.8)
        # nc = nx.draw_networkx_nodes(self.G_regular, pos, nodelist=nodes, node_color=colors, node_size=20, cmap=plt.cm.jet)
        labels = nx.get_node_attributes(self.G_regular, 'id')
        labels =nx.draw_networkx_labels(self.G_regular, pos,labels=labels,verticalalignment='top',font_size=8,alpha=0.5)
        # plt.colorbar(nc)


        nx.draw(self.G_regular, with_labels=True, font_weight='bold')
        # labels = nx.get_edge_attributes(self.G_regular,'weight')
        plt.show()
        # nx.draw_networkx_edge_labels(self.G_regular, edge_labels=labels, font_weight='bold')
        # plt.show()
        exit()
        print("asdja")

    def plot_multigraph(self):

        """
        ornek bunu sil
        """
        G = nx.MultiGraph()
        edges = [
        [0,1,2],
        [0,1,3],
        [1,2,4]
        ]
        w = [1,2,4]
        G.add_weighted_edges_from(edges)

        pos = nx.spring_layout(G,k=1.0)
        pos[0] = [0.0,0.0]
        pos[1] = [1.0,1.0]
        pos[2] = [2.0,0.0]
        print(pos)
        nx.draw_networkx_nodes(G, pos, node_color = 'b', node_size = 400, alpha = 1)
        ax = plt.gca()
        for k,e in enumerate(G.edges):
            print(e)
            ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="-", color="0.5", linewidth=w[k],
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                ),
                                ),
                )
        plt.axis('off')
        plt.show()




        # exit()



    def nematic(self):
        """
        for stick id 132 . [120-23] avg
        """
        box = np.array([self.L,self.L,self.L])
        directions = []
        unique_mol_ids, counts = np.unique(self.moleculeids,return_counts=True)
        shape_mol_ids = unique_mol_ids[counts>5]
        # print(shape_mol_ids)
        for i,s_id in enumerate(shape_mol_ids):
            mol_pos = self.pos[self.moleculeids==s_id]
            mol_types = self.typeids[[self.moleculeids==s_id]]
            mol_pos0 = mol_pos[mol_types==0]
            cm = com(mol_pos0,box)
            dir = com(mol_pos[120:124],box) - cm
            dir = np.divide(dir,np.linalg.norm(dir))
            directions.append(dir)

        directions = np.array(directions)

        Q = np.zeros((3,3))

        for dir in directions:
            for i in range(3):
                for j in range(3):
                    if(i==j):
                        Q[i,j] = Q[i,j] + 1.5*dir[i]*dir[j] - 0.5
                    else:
                        Q[i,j] = Q[i,j] + 1.5*dir[i]*dir[j]

        e_vals,e_vects = np.linalg.eig(Q)
        director_index = np.argmax(e_vals)
        director = e_vects[:,director_index]

        op = 0
        for dir in directions:
             op = op + 0.5*(np.dot(dir,director)*np.dot(dir,director)*3.0 -1)

        op = op/len(directions)
        print(op)

        # print(e_vals)
        # print(e_vects)

        # exit()


# exit()
