import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as Rot


def matrix_to_axisangle(rotmax):
    r = Rot.from_dcm(rotmax)
    rv = r.as_rotvec() # vector is the axis you turn around, norm is the angle
    return rv

def check_overlap2(pts1,pts2,margin,L):
    """
    Two groups of points returns False if they are closer than margin
    in the box
    """
    pts1 = pts1 + L*0.5
    pts2 = pts2 + L*0.5
    tree1 = KDTree(data=pts1, leafsize=12, boxsize=L+0.0001)
    tree2 = KDTree(data=pts2, leafsize=12, boxsize=L+0.0001)
    n = tree2.count_neighbors(tree1, margin)
    if(n==0):
        answer = True
    else:
        answer = False
    return answer

def translate(pos,v,L):
    pos = pos + v
    pos = pbc(pos,L)
    return pos

def rotatePbc(pos,rot_max,L):
    """
    Rotate a rigid body while respecting periodic boundary conditions
    Assumes the first is the central particle ie. CofM
    """
    cm = np.copy(pos[0])
    pos = pos - cm
    pos = pbc(pos,L)
    pos = np.matmul(rot_max,np.transpose(pos))
    pos = np.transpose(pos)
    pos = pos + cm
    pos = pbc(pos,L)
    return pos

def random_quat():
    """
    Not sure the first or last term is the scalar
    probably last
    """
    rands = np.random.uniform(size=3)
    quat = np.array([np.sqrt(1.0-rands[0])*np.sin(2*np.pi*rands[1]),
            np.sqrt(1.0-rands[0])*np.cos(2*np.pi*rands[1]),
            np.sqrt(rands[0])*np.sin(2*np.pi*rands[2]),
            np.sqrt(rands[0])*np.cos(2*np.pi*rands[2])])
    return quat

def quat_to_matrix(quat):
    """
    Convert a quaternion (assuming last term is scalar)
    to a rotation matrix that rotates columns when multiplies
    from left.
    https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_rotations
    """
    i = quat[0]
    j = quat[1]
    k = quat[2]
    r = quat[3]
    matrix = np.zeros((3,3))
    matrix[0,0] = -1.0 + 2.0*i*i + 2.0*r*r
    matrix[0,1] = 2.0*(i*j-k*r)
    matrix[0,2] = 2.0*(i*k+j*r)
    matrix[1,0] = 2.0*(i*j+k*r)
    matrix[1,1] = -1.0 + 2.0*j*j + 2.0*r*r
    matrix[1,2] = 2.0*(j*k-i*r)
    matrix[2,0] = 2.0*(i*k-j*r)
    matrix[2,1] = 2.0*(j*k+i*r)
    matrix[2,2] = -1.0 + 2.0*k*k + 2.0*r*r
    return matrix

def quat_converter(quat):
    """
    HOOMD : scalar is first (ie default is 1,0,0,0)
    MCSim : scalar is last (ie default is 0,0,0,1)
    Converts mc_sim quats -> hoomd quats
    ovito also uses mc_sim quats
    """
    a,b,c,d = quat
    quat_new = np.array([d,a,b,c])
    return quat_new

def pbc(pos,L):
    pos = pos - ((pos - L*0.5)//L + 1)*L
    return pos

def rotate_rows(matrix,pos):
    """
    takes a position matrix (N,3)-> rows are positions
    and a rotation matrix to rotate the positions
    - simple r_mat *matmul* pos_mat rotates the columns only
    which is not what we want. It is possible to take
    transpose of row-wise position vector do the matmul
    and than retranspose it again to row wise to get the
    same result.
    """
    pos = np.matmul(matrix,np.transpose(pos))
    return np.transpose(pos)


def check_overlap(p,pts,margin,L):
    """
    Returns true if there is no overlap btw a point p and a  group of points pts
    Useful for placing noms
    """
    pts = np.array(pts)
    dists_v = pts - p
    dists_v = pbc(dists_v,L)
    dists = np.linalg.norm(dists_v,axis=1)
    min_dist = np.min(dists)
    if(min_dist>margin):
        answer = True
    else:
        answer = False
    return answer



def matrix_to_quat(m):
    """
    take a rotation matrix and return corresponding quat
    onenote 6.0
    !!! Different Quat Notation !!
    Unit quat is = [1,0,0,0]
    (before it was [0,0,0,1])
    This one unfortunately fails if the one of the euler angles are too close to
    zero(Not so sure about that) (quat returns NaN since inside the sqrt becomes
    negative) Scipy handles it better
    """
    q = np.zeros(4)
    q[0] = 0.5*np.sqrt(1.0+m[0,0]+m[1,1]+m[2,2])
    q[1] = (1.0/(4.0*q[0]))*(m[2,1]-m[1,2])
    q[2] = (1.0/(4.0*q[0]))*(m[0,2]-m[2,0])
    q[3] = (1.0/(4.0*q[0]))*(m[1,0]-m[0,1])
    return q

def matrix_to_quat2(m):
    """
    Try with scipy
    """
    r = Rot.from_dcm(m)
    q = r.as_quat()
    return q


def wrap_pbc(x, Box):
    delta = np.where(x > 0.5 * Box, x- Box, x)
    delta = np.where(delta <- 0.5 * Box, Box + delta, delta)
    return delta

def com(a,Box):
    """
    Calculates the center of mass of a group of particle that possibly
    crosses the periodic boundary conditions 
    """
    theta = np.divide(a + 0.5 * Box, Box)*np.multiply(2,np.pi)
    xi_average = np.average(np.cos(theta), axis = 0)
    zeta_average = np.average(np.sin(theta), axis = 0)
    theta_average = np.arctan2(-zeta_average,-xi_average) + np.pi
    com = np.multiply(Box,theta_average)/np.multiply(2,np.pi)-0.5 * Box
    return com

def getMOI(pos):
    """
    Copied from : https://scipython.com/book/chapter-6-numpy/problems/p65/the-moment-of-inertia-tensor/
    Inertia products ie. non-diagonal terms can be negative
    """
    masses = np.ones(len(pos))
    x, y, z = pos.T
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Iyz = -np.sum(masses * y * z)
    Ixz = -np.sum(masses * x * z)
    I = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    # print(I)
    # exit()
    return I
