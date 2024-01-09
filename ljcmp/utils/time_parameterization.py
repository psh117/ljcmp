import numpy as np

import toppra as ta
import toppra.algorithm as algo

def time_parameterize(q_path, model_info, hz=20):
    path = q_path
    ss = np.linspace(0, 1, path.shape[0])
    path = ta.SplineInterpolator(ss, path)

    vel_lim = np.array(model_info['hardware']['joint_vel_limit'])
    effor_lim = np.array(model_info['hardware']['joint_effort_limit'])
    pc_vel = ta.constraint.JointVelocityConstraint(vel_lim)
    pc_acc = ta.constraint.JointAccelerationConstraint(effor_lim)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()

    duration = jnt_traj.duration

    ts_sample = np.linspace(0, duration, int(round(duration * hz)))
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)

    return duration, qs_sample, qds_sample, qdds_sample, ts_sample