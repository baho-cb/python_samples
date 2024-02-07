import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree as KDTree
import networkx as nx

def check_percolate_bonds(bonds,N,pts):

    """
    Construct networks from the bonds
    pick the largest connected network i.e. cluster
    Get the lattice positions for largest cluster
    Check if it spans the entire box in any dimension
    """
    G = nx.Graph()
    ids = np.arange(N*N)
    G.add_nodes_from(ids)
    G.add_edges_from(bonds)
    giant = max(nx.connected_components(G), key=len)
    largest_cluster = pts[list(giant)]
    # print(largest_cluster)


    # pts_np = np.array(pts)
    # tree = KDTree(data=pts_np, leafsize=12)
    # pairs = tree.sparse_distance_matrix(tree,cutoff+1.0)
    # dbscan = DBSCAN(eps=cutoff, min_samples=2, metric="precomputed", n_jobs=-1)
    # labels0 = dbscan.fit_predict(pairs)
    # largest_cluster = pts_np[labels0==0]
    row_check = np.unique(largest_cluster[:,0])
    col_check = np.unique(largest_cluster[:,1])
    isP = False
    if((len(row_check)==N) or (len(col_check)==N)):
        isP = True
    return isP

def adjust_neighbors_self_bond(lbo,rba,si,N_lattice):
    """
    When a self bond made at a lattice point, if the # of bonds reaches 4 then
    any other bond with another neighbor is impossible.
    """
    lbo_si = lbo[si]
    lbo_si = lbo_si[lbo_si>0.5]
    N_available = np.sum(lbo_si)
    if(N_available>0.5):
        ## do nothing
        pass
    elif(N_available==0):
        ## do
        for i in range(5):
            if(lbo[si,i]==0):
                lbo[si,i]=-1
        row = si // N_lattice
        col = si % N_lattice

        ### right neighbor
        if(col==(N_lattice-1)):
            right_index = si - N_lattice + 1
            lbo[right_index,2] = -1
            rba[right_index]-=1

        else:
            right_index = si + 1
            lbo[right_index,2] = -1
            rba[right_index]-=1
        ### upper neighbor
        if(row==0):
            upper_index = si + (N_lattice - 1)*N_lattice
            lbo[upper_index,3] = -1
            rba[upper_index]-=1
        else:
            upper_index = si - N_lattice
            lbo[upper_index,3] = -1
            rba[upper_index]-=1
        ### left_neighbor
        if(col==0):
            left_index = si + (N_lattice - 1)
            lbo[left_index,0] = -1
            rba[left_index]-=1
        else:
            left_index = si - 1
            lbo[left_index,0] = -1
            rba[left_index]-=1
        ### lower neighbor
        if(row==(N_lattice-1)):
            lower_index = si - (N_lattice - 1)*N_lattice
            lbo[lower_index,1] = -1
            rba[lower_index]-=1
        else:
            lower_index = si + N_lattice
            lbo[lower_index,1] = -1
            rba[lower_index]-=1
    else:
        print("Error 4556")
        exit()

    return lbo,rba


def adjust_neighbors_regular_bond(lbo,rba,si,N_lattice):
    """
    Similar to self bonds, the creation of a regular bond might render some others
    impossible if it is the 4th bond on a lattice (it is actually the same as the previous function)
    """
    lbo_si = lbo[si]
    lbo_si = lbo_si[lbo_si>0.5]
    N_available = np.sum(lbo_si)
    if(N_available>0.5):
        ## do nothing
        pass
    elif(N_available==0):
        ## do
        for i in range(5):
            print("kok")
            print(lbo)
            print(lbo[si,i])
            exit()
            if(lbo[si,i]==0):
                lbo[si,i]=-1
        row = si // N_lattice
        col = si % N_lattice

        ### right neighbor
        if(col==(N_lattice-1)):
            right_index = si - N_lattice + 1
            lbo[right_index,2] = -1
            rba[right_index]-=1

        else:
            right_index = si + 1
            lbo[right_index,2] = -1
            rba[right_index]-=1
        ### upper neighbor
        if(row==0):
            upper_index = si + (N_lattice - 1)*N_lattice
            lbo[upper_index,3] = -1
            rba[upper_index]-=1
        else:
            upper_index = si - N_lattice
            lbo[upper_index,3] = -1
            rba[upper_index]-=1
        ### left_neighbor
        if(col==0):
            left_index = si + (N_lattice - 1)
            lbo[left_index,0] = -1
            rba[left_index]-=1
        else:
            left_index = si - 1
            lbo[left_index,0] = -1
            rba[left_index]-=1
        ### lower neighbor
        if(row==(N_lattice-1)):
            lower_index = si - (N_lattice - 1)*N_lattice
            lbo[lower_index,1] = -1
            rba[lower_index]-=1
        else:
            lower_index = si + N_lattice
            lbo[lower_index,1] = -1
            rba[lower_index]-=1
    else:
        print("Error 4556")
        exit()

    return lbo,rba

def add_regular_bond(lbo,li1,dir2,nl):
    row = li1 // nl
    col = li1 % nl
    if(dir2==0):
        if(col==(nl-1)):
            li2 = li1 - nl
        else:
            li2 = li1 + 1
    if(dir2==1):
        if(row==0):
            li2 = li1 + (nl - 1)*nl
        else:
            li2 = li1 - nl
    if(dir2==2):
        if(col==0):
            li2 = li1 + (nl - 1)
        else:
            li2 = li1 - 1
    if(dir2==3):
        if(row==(nl-1)):
            li2 = li1 - (nl - 1)*nl
        else:
            li2 = li1 + nl

    lbo[li1,dir2] = 1
    lbo[li2,(dir2+2)%4] = 1
    return lbo,li2


"""
Nov 22 2023

Toy model for hysteresis on gel point

Percolation condition 1:
Is there a lattice point at every row or at every column in the largest cluster

This is a bit different than the previous 6 models. This one has self bonds that
actually matter. We can't predefine a total number of chains because we don't know
where the self bonds will occur (if they are not at boundary they will prevent
two non-self bonds)

It is easier to implement a system where we pick first the lattice point
and then find out the number of available bonds and pick one of those bonds. But
this might bias the bonding progression. We want every bond to have an equal chance.
To do this we need to know how many bonds are available in total at every step.
That requires a lot of bookkeeping.

BELOW IS IRRELEVANT
This is bond percolation not site percolation - No pbc
Total number of bonds :
2(n^2) - 2n (no pbc)
2(n^2) (with pbc)
2(n^2) + xn (with pbc & with loops)

"""

N_try = 2
rates = []
for ssdss in range(N_try):
    N_lattice = 5
    loop_rate = 0.1 ### every time we make a bond there is this chance of this loop
    ## being a self loop
    # np.random.seed(seed=113)


    Np = N_lattice**2
    lattice_ids = np.arange(Np)
    N_total_bonds = N_lattice*N_lattice*2 - 2*N_lattice

    p1 = np.arange(N_lattice)
    lattice_pts = np.zeros((N_lattice*N_lattice,2),dtype=int)
    for i,px in enumerate(p1):
        for j,py in enumerate(p1):
            lattice_pts[i*N_lattice+j] = np.array([px,py])


    ### No bond but can be : 0
    ### No bond but cannot be : -1
    ### Yes bond : 1
    lattice_bond_occupation = np.zeros((Np,5),dtype=int)
    total_bonds = np.sum(lattice_bond_occupation,axis=1)
    regular_bonds_available = 4 - total_bonds
    total_bonds_available = np.copy(regular_bonds_available)
    N_aval = np.sum(regular_bonds_available)
    lattice_bag_to_pick = np.repeat(lattice_ids,regular_bonds_available)
    all_bonds = []

    is_percolated = False
    steps = 0
    percolation_step = 0
    check_for_perco = 1
    while(N_aval > 0):
        p_self = np.random.rand()
        if(p_self<loop_rate):
            candidates = np.where(total_bonds_available>1.5)[0]
            if(len(candidates)>0.5):
                self_index = np.random.choice(candidates)
                lattice_bond_occupation[self_index,4] = lattice_bond_occupation[self_index,4] + 2
                lattice_bond_occupation,regular_bonds_available = adjust_neighbors_self_bond(lattice_bond_occupation,regular_bonds_available,self_index,N_lattice)

                all_bonds.append([self_index,self_index])
                total_bonds_available[self_index] = total_bonds_available[self_index] - 2
                regular_bonds_available[self_index] = regular_bonds_available[self_index] - 2
                regular_bonds_available[regular_bonds_available<0] = 0
                total_bonds_available[total_bonds_available<0] = 0
                N_aval = np.sum(regular_bonds_available)
                lattice_bag_to_pick = np.repeat(lattice_ids,regular_bonds_available)

        else:
            ### make a normal bond
            bag_index = np.random.randint(0,len(lattice_bag_to_pick)) # pick the first crosslinker
            li1 = lattice_bag_to_pick[bag_index]
            lbo1 = lattice_bond_occupation[li1,:4]
            # print(lbo1)
            # print(regular_bonds_available[li1])
            dir2 = np.where(lbo1==0)[0]
            dir2 = np.random.choice(dir2)

            lattice_bond_occupation,li2 = add_regular_bond(lattice_bond_occupation,li1,dir2,N_lattice)

            all_bonds.append([li1,li2])
            total_bonds_available[li1] = total_bonds_available[li1] - 1
            total_bonds_available[li2] = total_bonds_available[li2] - 1
            regular_bonds_available[li2] = regular_bonds_available[li2] - 1
            regular_bonds_available[li1] = regular_bonds_available[li1] - 1


            lattice_bond_occupation,regular_bonds_available = adjust_neighbors_regular_bond(lattice_bond_occupation,regular_bonds_available,li1,N_lattice)
            lattice_bond_occupation,regular_bonds_available = adjust_neighbors_regular_bond(lattice_bond_occupation,regular_bonds_available,li2,N_lattice)

            regular_bonds_available[regular_bonds_available<0] = 0
            total_bonds_available[total_bonds_available<0] = 0
            N_aval = np.sum(regular_bonds_available)

            lattice_bag_to_pick = np.repeat(lattice_ids,regular_bonds_available)

            if(check_for_perco==1):
                if(steps>N_lattice-2):
                    is_percolated = check_percolate_bonds(all_bonds,N_lattice,lattice_pts)
                if(is_percolated):
                    check_for_perco = 0
                    percolation_step = len(all_bonds)

        steps += 1
        # print(N_aval)

    """
    after the gel is made to completion, we simply destroy bonds one-by-one
    ad check for percolation after each step
    """
    N_b_total = len(all_bonds)

    is_p = True
    while(is_p):
        break_index = np.random.randint(0,len(all_bonds))
        del all_bonds[break_index]
        is_p = check_percolate_bonds(all_bonds,N_lattice,lattice_pts)
    rate = len(all_bonds)/N_b_total
    rates.append(rate)

rates = np.array(rates)
m = np.average(rates)
st = np.std(rates)
print('%.5f %.5f %d' %(m,st,N_try))


exit()
