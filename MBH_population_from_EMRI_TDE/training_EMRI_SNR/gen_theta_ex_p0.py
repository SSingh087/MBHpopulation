#!/usr/bin/env python

import numpy as np
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description="Generate data for training.")

parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--log10_M", type=float, nargs=2, required=True, help="log10(M)")
parser.add_argument("--a", type=float, nargs=2, required=True, help="a")
parser.add_argument("--e0", type=float, nargs=2, required=True, help="e0")
parser.add_argument("--Y0", type=float, nargs=2, required=True, help="Y0")
parser.add_argument("--dist", type=float, nargs=2, required=True, help="distance in Gpc. For conversion from z to Gpc refer https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.123531")
parser.add_argument("--T_SIGNAL", type=float, nargs=2, required=True, help="time in years")
parser.add_argument("--qS", type=float, nargs=2, required=False, help="Sky location polar angle in ecliptic coordinates. If not given the distribution is uniform between 0 to pi")
parser.add_argument("--phiS", type=float, nargs=2, required=False, help="Sky location azimuthal angle in ecliptic coordinates. If not given the distribution is uniform between 0 to 2*pi")
parser.add_argument("--qK", type=float, nargs=2, required=False, help="Initial BH spin polar angle in ecliptic coordinates. If not given the distribution is uniform between 0 to pi")
parser.add_argument("--phiK", type=float, nargs=2, required=False, help="Initial BH spin azimuthal angle in ecliptic coordinates. If not given the distribution is uniform between 0 to 2*pi")
parser.add_argument("--Phi_phi0", type=float, nargs=2, required=False, help="Phi_phi0")
parser.add_argument("--Phi_theta0", type=float, nargs=2, required=False, help="Phi_theta0")
parser.add_argument("--Phi_r0", type=float, nargs=2, required=False, help="Phi_r0")
parser.add_argument("--file_name", type=str, required=True, help="Output file name.")

args = parser.parse_args()

N = args.events

log10_M_min, log10_M_max = args.log10_M
a_min, a_max = args.a
e0_min, e0_max = args.e0
Y0_min, Y0_max = args.Y0
dist_min, dist_max = args.dist
T_SIGNAL_min, T_SIGNAL_max = args.T_SIGNAL

if args.qS is None :
    qS_min, qS_max = 0.1, np.pi * 0.99

if args.phiS is None :
    phiS_min, phiS_max = 0.1, 2 * np.pi * 0.99
    
if args.qK is None :
    qK_min, qK_max = 0.1, np.pi * 0.99
    
if args.phiK is None :
    phiK_min, phiK_max = 0.1, 2 * np.pi * 0.99
    
if args.Phi_phi0 is None :
    Phi_phi0_min, Phi_phi0_max = 0.1, 2 * np.pi * 0.99
    
if args.Phi_theta0 is None :
    Phi_theta0_min, Phi_theta0_max = 0.1, 2 * np.pi * 0.99
    
if args.Phi_r0 is None :
    Phi_r0_min, Phi_r0_max = 0.1, 2 * np.pi * 0.99
# Generate random values

data = [
    np.random.uniform(log10_M_min, log10_M_max, N),
    np.random.uniform(a_min, a_max, N),
    np.random.uniform(e0_min, e0_max, N),
    np.random.uniform(Y0_min, Y0_max, N),
    np.random.uniform(dist_min, dist_max, N),
    np.random.uniform(qS_min, qS_max, N),
    np.random.uniform(phiS_min, phiS_max, N),
    np.random.uniform(qK_min, qK_max, N),
    np.random.uniform(phiK_min, phiK_max, N),
    np.random.uniform(Phi_phi0_min, Phi_phi0_max, N),
    np.random.uniform(Phi_theta0_min, Phi_theta0_max, N),
    np.random.uniform(Phi_r0_min, Phi_r0_max, N),
    np.random.uniform(T_SIGNAL_min, T_SIGNAL_max, N),
]

data = np.asarray(data).T
print("data shape: ", data.shape)
                    
# Ensure the output directory exists
## check directory exists 

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'./data/PRE_TRAIN_DATA/{events_for_dir}_events/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the data
output_path = os.path.join(output_dir, args.file_name)
np.save(output_path, data)

print(f"Data successfully saved to {output_path}")