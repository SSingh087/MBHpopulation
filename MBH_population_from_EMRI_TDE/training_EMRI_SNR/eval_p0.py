#!/usr/bin/env python
"""Solve for the initial semi-latus rectum p0 that makes EMRIs plunge at a target time."""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from multiprocessing import Pool
from few.utils.utility import get_separatrix
from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral


parser = argparse.ArgumentParser(description="Generate data for training.")
parser.add_argument("--events", type=int, required=True, help="total events")
parser.add_argument("--MAX_SIGNAL_DURATION", type=float, required=True, help="total events")
parser.add_argument("--OBSERVING_WINDOW", type=float, required=True, help="total events")
args = parser.parse_args()

events_for_dir = f"{args.events:.0E}".replace("+0", "").replace(".0", "")
output_dir = f'./data/PRE_TRAIN_DATA/{events_for_dir}_events'


inspiral_kwargs = {
    "max_init_len": int(5e7),  # all of the trajectories will be well under max_init_len
}


traj = EMRIInspiral(func="pn5", enforce_schwarz_sep=True)

# Physical constants and tolerances
MIN_SEPARATION_FROM_SEPARATRIX = 0.5  # Minimum offset from separatrix for waveform stability
MAX_ADAPTIVE_OFFSET = 20.0  # Maximum adaptive offset for fallback

# MAX_SIGNAL_DURATION = args.MAX_SIGNAL_DURATION
def eval_p0s_T_gt_T_OBS(num_event):
    """
    Evaluate initial semi-latus rectum (p0) for T > T_obs regime.
    
    Strategy:
    1. Try progressively wider bound ranges
    2. Use physical separatrix offset for stability
    3. Fall back to adaptive offset based on mass ratio if solver fails
    """
    
    _T_INJECT_ = theta_ex_p0[num_event, -1]
    M = 10**theta_ex_p0[num_event, 0]
    mu = 10
    a = theta_ex_p0[num_event, 2]
    e0 = theta_ex_p0[num_event, 3]
    Y0 = theta_ex_p0[num_event, 4]

    # Calculate separatrix semi-latus rectum (physical boundary)
    # Reference: https://arxiv.org/pdf/1912.07609
    ps = get_separatrix(a, e0, Y0)
    
    log_mu_M = np.log10(mu/M)
    p0 = None
    attempt = 0
    
    # Strategy 1: Try to solve with progressively wider bounds
    bounds_sequence = [
        (ps + MIN_SEPARATION_FROM_SEPARATRIX + 1e-6, 20.),   # Near-separatrix regime
        (20., 50.),                                             # Mid-range regime
        (50., 100.),                                            # Far-range regime
    ]
    
    for bounds_low, bounds_high in bounds_sequence:
        try:
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, 
                           traj_args=[M, mu, a, e0, Y0], 
                           index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                           traj_kwargs=inspiral_kwargs, bounds=[bounds_low, bounds_high])
            print(f"Event {num_event} SUCCESS (attempt {attempt}): p0={p0:.4f}, ps={ps:.4f}, "
                  f"log10(mu/M)={log_mu_M:.2f}")
            return p0
        except Exception as e:
            attempt += 1
            continue
    
    # Strategy 2: If solver fails, use mass-ratio-adaptive fallback
    # Smaller mass ratios need larger offsets from separatrix
    if p0 is None:
        if -9 <= log_mu_M < -6:
            # Smaller mu: waveform stable with modest offset
            p0 = ps + MIN_SEPARATION_FROM_SEPARATRIX
            reason = "mass_ratio_small"
        elif -6 <= log_mu_M < -4:
            # Medium mu: need intermediate offset
            p0 = ps + 2.0
            reason = "mass_ratio_medium"
        else:
            # Larger mu: conservative large offset
            p0 = ps + MAX_ADAPTIVE_OFFSET
            reason = "mass_ratio_large"
        
        print(f"Event {num_event} FALLBACK ({reason}): p0={p0:.4f}, ps={ps:.4f}, "
              f"log10(mu/M)={log_mu_M:.2f}, e0={e0:.3f}, a={a:.3f}, Y0={Y0:.3f}")
    
    return p0




def eval_p0s_T_lt_T_OBS(num_event):
    """
    Evaluate initial semi-latus rectum (p0) for T < T_obs regime.
    
    For shorter observation windows, we add a small buffer (0.4 years)
    to avoid numerical issues with inspiral integration.
    
    Strategy is identical to T > T_obs case but with adjusted injection time.
    """
    
    # Add buffer to MAX_SIGNAL_DURATION to ensure numerical stability
    # in trajectory integration (see error comment in original code)
    _T_INJECT_ = args.MAX_SIGNAL_DURATION + 0.4 
    
    M = 10**theta_ex_p0[num_event, 0]
    mu = 10
    a = theta_ex_p0[num_event, 2]
    e0 = theta_ex_p0[num_event, 3]
    Y0 = theta_ex_p0[num_event, 4]

    # Calculate separatrix semi-latus rectum (physical boundary)
    # Reference: https://arxiv.org/pdf/1912.07609
    ps = get_separatrix(a, e0, Y0)

    log_mu_M = np.log10(mu/M)
    p0 = None
    attempt = 0
    
    # Strategy 1: Try to solve with progressively wider bounds
    bounds_sequence = [
        (ps + MIN_SEPARATION_FROM_SEPARATRIX + 1e-6, 20.),   # Near-separatrix regime
        (20., 50.),                                             # Mid-range regime
        (50., 100.),                                            # Far-range regime
    ]
    
    for bounds_low, bounds_high in bounds_sequence:
        try:
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, 
                           traj_args=[M, mu, a, e0, Y0], 
                           index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                           traj_kwargs=inspiral_kwargs, bounds=[bounds_low, bounds_high])
            print(f"Event {num_event} SUCCESS (attempt {attempt}): p0={p0:.4f}, ps={ps:.4f}, "
                  f"log10(mu/M)={log_mu_M:.2f}")
            return p0
        except Exception as e:
            attempt += 1
            continue
    
    # Strategy 2: If solver fails, use mass-ratio-adaptive fallback
    # Smaller mass ratios need larger offsets from separatrix
    if p0 is None:
        if -9 <= log_mu_M < -6:
            # Smaller mu: waveform stable with modest offset
            p0 = ps + MIN_SEPARATION_FROM_SEPARATRIX
            reason = "mass_ratio_small"
        elif -6 <= log_mu_M < -4:
            # Medium mu: need intermediate offset
            p0 = ps + 2.0
            reason = "mass_ratio_medium"
        else:
            # Larger mu: conservative large offset
            p0 = ps + MAX_ADAPTIVE_OFFSET
            reason = "mass_ratio_large"
        
        print(f"Event {num_event} FALLBACK ({reason}): p0={p0:.4f}, ps={ps:.4f}, "
              f"log10(mu/M)={log_mu_M:.2f}, e0={e0:.3f}, a={a:.3f}, Y0={Y0:.3f}")
    
    return p0


if __name__=="__main__":

    if args.MAX_SIGNAL_DURATION == args.OBSERVING_WINDOW:

        print("Evaluating p0 for T < T_obs")
        theta_ex_p0 = np.load(f'{output_dir}/theta_ex_p0_T_lt_T_obs.npy')
        events = len(theta_ex_p0)

        pool = Pool(32)
        p0s_T_max = pool.map(eval_p0s_T_lt_T_OBS, range(events))
        np.save(f'{output_dir}/p0_s_T_lt_T_obs.npy', np.array(p0s_T_max))

        # seq here is M, a, || p0 || e0 ....
        injection_params = np.hstack([theta_ex_p0[:, 0:2], np.array(p0s_T_max)[:, None], theta_ex_p0[:, 2:]])
        np.save(f'{output_dir}/injection_params_T_lt_T_obs.npy', injection_params)


    elif args.MAX_SIGNAL_DURATION > args.OBSERVING_WINDOW:
        print("Evaluating p0 for T > T_obs")
        theta_ex_p0 = np.load(f'{output_dir}/theta_ex_p0_T_gt_T_obs.npy')
        events = len(theta_ex_p0)

        pool = Pool(32)
        p0s_T_max = pool.map(eval_p0s_T_gt_T_OBS, range(events))
        np.save(f'{output_dir}/p0_s_T_gt_T_obs.npy', np.array(p0s_T_max))

        # seq here is M, a, || p0 || e0 ....
        injection_params = np.hstack([theta_ex_p0[:, 0:2], np.array(p0s_T_max)[:, None], theta_ex_p0[:, 2:]])
        np.save(f'{output_dir}/injection_params_T_gt_T_obs.npy', injection_params)

    else :
        raise ValueError('Oh Cmon, really bro !!')