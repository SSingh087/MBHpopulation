#!/usr/bin/env python

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
output_dir = f'/data/wiay/postgrads/shashwat/EMRI_data/PRE_TRAIN_DATA/{events_for_dir}_events'


inspiral_kwargs = {
    "max_init_len": int(5e7),  # all of the trajectories will be well under max_init_len
}


traj = EMRIInspiral(func="pn5", enforce_schwarz_sep=True)

# MAX_SIGNAL_DURATION = args.MAX_SIGNAL_DURATION
def eval_p0s_T_gt_T_OBS(num_event):
    
    _T_INJECT_ = theta_ex_p0[num_event, -1]
    M = 10**theta_ex_p0[num_event, 0]
    mu = 10**theta_ex_p0[num_event, 1]
    a = theta_ex_p0[num_event, 2]
    e0 = theta_ex_p0[num_event, 3]
    Y0 = theta_ex_p0[num_event, 4]

    # calculate when the mu plunges (the sepratrix) 
    # https://arxiv.org/pdf/1912.07609
    ps = get_separatrix(a, e0, Y0)

    log_mu_M = np.log10(mu/M)

    try:
        # Solve for log_mu_M E [-9,-4] 
        # the **try except** tree is to make sure solver 
        # finds a root before maximum iteration is reached
        
        try :
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, traj_args=[M, mu, a, e0, Y0], 
                            index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                            traj_kwargs=inspiral_kwargs, bounds=[ps + .1 + 1e-6, 20.])
                            #rtol=1e-9, xtol=1e-9)
        except : 
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, traj_args=[M, mu, a, e0, Y0], 
                            index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                            traj_kwargs=inspiral_kwargs, bounds=[20., 50.])
                            # rtol=1e-9, xtol=1e-9)
        print(num_event, "TRY", p0, ps, e0, a, Y0, _T_INJECT_)#np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)

    except:
        if -9 < log_mu_M <= -6:
            # usually p0 solves to be ~0.1 for 2-4 year span
            # this has been set to 0.5 because the waveform is 
            # not stable for positions close to ps.
            p0 = ps + 0.5
            print(num_event, "FAIL", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)
        else :
            try :
                # Solve for value of p0 which is outside the bounds. 
                # this will likely be a case for -3 < log_mu_M < -5
                p0 = get_p_at_t(traj_module=traj, t_out=MAX_SIGNAL_DURATION, traj_args=[M, mu, a, e0, Y0], 
                                index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                                traj_kwargs=inspiral_kwargs, bounds=[50., 100.])

                print(num_event, "TRYING AGAIN", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)
            except :
                # In case it fails set the p0 value to be large enough.
                # This should be long enough to accomodate plunge.
                p0 = ps + 20.
                print(num_event, "THAT'S IT", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)

    return p0


def eval_p0s_T_lt_T_OBS(num_event):
    
    _T_INJECT_ = args.MAX_SIGNAL_DURATION + 0.4 
    # 0.5 is added because of this error 
    # Traceback (most recent call last):
    #   File "/home/2673888s/EMRI_population/condor_script_1E5_vary_T_SIGNAL/get_traj_T_lt_T_obs.py", line 121, in <module>
    #     traj(M[num_event], mu[num_event], a[num_event],
    #   File "/home/2673888s/.conda/envs/fastemriwaveforms/lib/python3.12/site-packages/fastemriwaveforms-1.5.5-py3.12-linux-x86_64.egg/few/utils/baseclasses.py", line 693, in __call__
    #     out = self.get_inspiral(*args, **kwargs)
    #           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "/home/2673888s/.conda/envs/fastemriwaveforms/lib/python3.12/site-packages/fastemriwaveforms-1.5.5-py3.12-linux-x86_64.egg/few/trajectory/inspiral.py", line 266, in get_inspiral
    #     t, p, e, x, Phi_phi, Phi_theta, Phi_r = self.inspiral_generator(
    #                                             ^^^^^^^^^^^^^^^^^^^^^^^^
    #   File "src/inspiralwrap.pyx", line 68, in pyInspiral.pyInspiralGenerator.__call__
    # ValueError: Error: Initial length is too short. Inspiral requires more points. Need to raise max_init_len parameter for inspiral.
    
    M = 10**theta_ex_p0[num_event, 0]
    mu = 10**theta_ex_p0[num_event, 1]
    a = theta_ex_p0[num_event, 2]
    e0 = theta_ex_p0[num_event, 3]
    Y0 = theta_ex_p0[num_event, 4]

    # calculate when the mu plunges (the sepratrix) 
    # https://arxiv.org/pdf/1912.07609
    ps = get_separatrix(a, e0, Y0)

    log_mu_M = np.log10(mu/M)

    try:
        # Solve for log_mu_M E [-9,-4] 
        # the **try except** tree is to make sure solver 
        # finds a root before maximum iteration is reached
        
        try :
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, traj_args=[M, mu, a, e0, Y0], 
                            index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                            traj_kwargs=inspiral_kwargs, bounds=[ps + .1 + 1e-6, 20.])
                            #rtol=1e-9, xtol=1e-9)
        except : 
            p0 = get_p_at_t(traj_module=traj, t_out=_T_INJECT_, traj_args=[M, mu, a, e0, Y0], 
                            index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                            traj_kwargs=inspiral_kwargs, bounds=[20., 50.])
                            # rtol=1e-9, xtol=1e-9)
        print(num_event, "TRY", p0, ps, e0, a, Y0, _T_INJECT_)#np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)

    except:
        if -9 < log_mu_M <= -6:
            # usually p0 solves to be ~0.1 for 2-4 year span
            # this has been set to 0.5 because the waveform is 
            # not stable for positions close to ps.
            p0 = ps + 0.5
            print(num_event, "FAIL", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)
        else :
            try :
                # Solve for value of p0 which is outside the bounds. 
                # this will likely be a case for -3 < log_mu_M < -5
                p0 = get_p_at_t(traj_module=traj, t_out=MAX_SIGNAL_DURATION, traj_args=[M, mu, a, e0, Y0], 
                                index_of_p=3, index_of_a=2, index_of_e=4, index_of_x=5,
                                traj_kwargs=inspiral_kwargs, bounds=[50., 100.])

                print(num_event, "TRYING AGAIN", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)
            except :
                # In case it fails set the p0 value to be large enough.
                # This should be long enough to accomodate plunge.
                p0 = ps + 20.
                print(num_event, "THAT'S IT", p0, ps, e0, a, Y0, _T_INJECT_)# ps, np.log10(mu), np.log10(M), log_mu_M, e0, a, Y0)

    return p0


if __name__=="__main__":

    if args.MAX_SIGNAL_DURATION == args.OBSERVING_WINDOW:

        print("Evaluating p0 for T < T_obs")
        theta_ex_p0 = np.load(f'{output_dir}/theta_ex_p0_T_lt_T_obs.npy')
        events = len(theta_ex_p0)

        pool = Pool(32)
        p0s_T_max = pool.map(eval_p0s_T_lt_T_OBS, range(events))
        np.save(f'{output_dir}/p0_s_T_lt_T_obs.npy', np.array(p0s_T_max))

        # seq here is M, mu, a, || p0 || e0 ....
        injection_params = np.hstack([theta_ex_p0[:, 0:3], np.array(p0s_T_max)[:, None], theta_ex_p0[:, 3:]])
        np.save(f'{output_dir}/injection_params_T_lt_T_obs.npy', injection_params)


    elif args.MAX_SIGNAL_DURATION > args.OBSERVING_WINDOW:
        print("Evaluating p0 for T > T_obs")
        theta_ex_p0 = np.load(f'{output_dir}/theta_ex_p0_T_gt_T_obs.npy')
        events = len(theta_ex_p0)

        pool = Pool(32)
        p0s_T_max = pool.map(eval_p0s_T_gt_T_OBS, range(events))
        np.save(f'{output_dir}/p0_s_T_gt_T_obs.npy', np.array(p0s_T_max))

        # seq here is M, mu, a, || p0 || e0 ....
        injection_params = np.hstack([theta_ex_p0[:, 0:3], np.array(p0s_T_max)[:, None], theta_ex_p0[:, 3:]])
        np.save(f'{output_dir}/injection_params_T_gt_T_obs.npy', injection_params)

    else :
        raise ValueError('Oh Cmon, really bro !!')