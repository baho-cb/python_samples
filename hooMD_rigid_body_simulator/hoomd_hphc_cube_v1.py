import numpy as np
import sys, os
sys.path.insert(0,"/home/bargun2/Programs/azplugins_fork/build/release")
import hoomd
import azplugins
import hoomd
import hoomd.md
import azplugins
import gsd.hoomd
import argparse
import time
import gc
import datetime
from SimulatorHandler import SimulatorHandler

"""
December 30 - 2022

Randomly places rigid bodies (that are stored in ./shapes/. as gsd files)
in a box to run a MD simulation with it. Rigidity is maintained with rigidity
constraints of HOOMD, not with intra-particle harmonic bonds.

Uses the custom HalfPowerHalfCosine potential that is implemented with AZPlugins
won't work with regular HOOMD
"""

parser = argparse.ArgumentParser(description="Basic run")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-g', '--gpu', metavar="<int>", type=int, dest="gpu_id", required=True, help="0 or 1, -1 for cpu",default=0)
non_opt.add_argument('--N', metavar="<int>", type=int, dest="N_mp", required=True, help=" # of particles ")
non_opt.add_argument('--L', metavar="<float>", type=float, dest="box_size", required=True, help=" Box Size ")
non_opt.add_argument('--output', metavar="<dat>", type=str, dest="output", required=True, help="output file name ",default=0 )
non_opt.add_argument('--ts', metavar="<int>", type=int, dest="timesteps", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--dump', metavar="<int>", type=int, dest="dump_period", required=True, help="dump_period ",default=1 )
non_opt.add_argument('--kT', metavar="<float>", type=float, dest="kT", required=True, help="temperature ",default=1 )
non_opt.add_argument('--tau', metavar="<float>", type=float, dest="tau", required=True, help="tau -> for Nose Hoover ",default=1 )
non_opt.add_argument('--seed', metavar="<int>", type=int, dest="seed", required=True, help="seed for random particle placement ",default=1 )
non_opt.add_argument('--dt', metavar="<float>", type=float, dest="dt", required=True, help=" timestep ")

args = parser.parse_args()
gpu_id = args.gpu_id
N_mp = args.N_mp # N_mp is the # of rigid bodies
output = args.output
box_size = args.box_size
timesteps = args.timesteps
dump_period = args.dump_period
dt = args.dt
tau = args.tau
seed = args.seed
kT = args.kT
log_period = dump_period


shape = 'shapes/cube_v2.gsd'
notice_level = 2
context_initialize_str = "--gpu=" + str(gpu_id)

if(gpu_id < -0.5):
    context_initialize_str = "--mode=cpu"

hoomd.context.initialize(context_initialize_str)
sim = hoomd.context.SimulationContext()
hoomd.option.set_notice_level(notice_level)

handler = SimulatorHandler(shape)
handler.setBox(box_size,N_mp)
snapshot = handler.get_snap(sim)
system = hoomd.init.read_snapshot(snapshot)
pos  = handler.getRigidConstituents()

rigid = hoomd.md.constrain.rigid()
rigid.set_param('Re', types = ['Os']*len(pos), positions = pos)
rigid.validate_bodies()

g_rigid_centers = hoomd.group.rigid_center()

nl = hoomd.md.nlist.tree()

hphc = azplugins.pair.halfpairhalfcosine(r_cut=2.39, nlist=nl)
hphc.pair_coeff.set('Os','Os', pwr=2.5, depth=0.00035, pos=1.46375, width=3.3833)
hphc.pair_coeff.set(['Re'],['Os','Re'], pwr=2.5, depth=0.00035, pos=1.46375, width=3.3833, r_cut=False)

if not os.path.exists('./out/'):
    os.makedirs('./out/')

n =  './out/' + output +'.gsd'
nlog = './out/' + output +'.log'


hoomd.md.integrate.mode_standard(dt=dt,aniso=True)
nvt = hoomd.md.integrate.nvt(group=g_rigid_centers, kT=kT, tau=tau)
nvt.randomize_velocities(seed = np.random.randint(low=5,high=100))

analyzer = hoomd.analyze.log(filename=nlog, quantities=['potential_energy', 'kinetic_energy','translational_kinetic_energy','rotational_kinetic_energy','temperature'],
period=log_period, header_prefix='#', overwrite=False)
d = hoomd.dump.gsd(n, period=dump_period,dynamic=['attribute', 'momentum'], group=hoomd.group.all(), overwrite=False)
hoomd.run(timesteps)


exit(0)
