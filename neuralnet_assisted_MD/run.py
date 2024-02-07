import numpy as np
from Force import Force
from Sim import Sim
import argparse
import sys, os

"""
April 2022

This script sets the parameters for Neural-Net Assisted rigid-body MD simulations
of cubes in NVT (Nose-Hoover thermostat).

Can be run with:
python3 run.py --init ./init/init.gsd --out output_gsd --ts 20000 --dump 2000 --kT 0.3 --dt 0.005 --cont 0

./init/ : Contains the initial state gsd_files that the simulation will start from
./models/ : Contains the neural-nets
./out/ : Simulation output is saved in this folder as a gsd file

"""

parser = argparse.ArgumentParser(description="Runs NN-assisted MD")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('--init', metavar="<dat>", type=str, dest="init_gsd",
required=True, help=".gsd files " )
non_opt.add_argument('--out', metavar="<dat>", type=str, dest="output",
required=True, help=".gsd files " )
non_opt.add_argument('--ts', metavar="<int>", type=int, dest="timesteps", required=True, help="length of sim ",default=1 )
non_opt.add_argument('--dump', metavar="<int>", type=int, dest="dump_period", required=True, help="dump frequency ",default=1 )
non_opt.add_argument('--kT', metavar="<float>", type=float, dest="kT", required=True, help="temperature ",default=1 )
non_opt.add_argument('--dt', metavar="<float>", type=float, dest="dt", required=True, help="dt ",default=1 )
non_opt.add_argument('--cont', metavar="<int>", type=int, dest="is_cont", required=True, help="1 if continue, 0 if not(about thermostat variables) ",default=1 )
non_opt.add_argument('--accel', metavar="<dat>", type=str, dest="accel_file", required=False, help="if continue you need to get accelerations as well ",default='no_accel')

args = parser.parse_args()
init_gsd = args.init_gsd
output = args.output
timesteps = args.timesteps
dump_period = args.dump_period
kT = args.kT
dt = args.dt
is_cont = args.is_cont
accel_file = args.accel_file

if not os.path.exists('./out/'):
    os.makedirs('./out/')

outname =  './out/' + output +'.gsd'


sim = Sim()
tau = 1.0
N_list_cutoff = 6.30 + 2.0
N_list_every = 100
force_path = 'models/model'

sim.placeParticlesFromGsd(init_gsd)
sim.setForces(force_path)
sim.setNeighborList(N_list_cutoff,N_list_every)
sim.setDumpFreq(dump_period)
sim.setDump(outname)
sim.setkT(kT)
sim.setTau(tau)
sim.setdt(dt)
sim.setCutoff(6.30)
sim.setGradientMethod("forward")
sim.set_integrator(np.array([-0.0317711,0.108609,0.00814227,-0.014238]))
sim.set_m_exp_factor(1.00011)
sim.is_continue(is_cont)
sim.set_accelerations(accel_file)
sim.run(timesteps)
print("Simulation completed.")





































exit()
