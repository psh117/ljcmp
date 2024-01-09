import numpy as np
from srmt.constraints.constraints import OrientationConstraint, MultiChainConstraint, MultiChainFixedOrientationConstraint
from srmt.utils.transform_utils import get_transform, get_pose
from srmt.planning_scene.planning_scene_tools import add_shelf, add_table

from termcolor import colored
from math import cos, sin, pi
from scipy.spatial.transform import Rotation as R

import yaml

from ljcmp.utils.cont_reader import ContinuousGraspCandidates

def generate_environment(exp_name):
    """_summary_

    Args:
        exp_name (str): experiment name

    Returns:
        constraint: constraint
        model_info: model information
        c: default condition value
        update_scene_from_yaml: update scene from yaml function
        set_constraint: constraint set function
    """
    
    if 'panda_orientation' in exp_name:
        return generate_environment_panda_orientation(exp_name)

    if 'panda_dual' in exp_name:
        return generate_environment_panda_dual(exp_name)

    if 'panda_triple' in exp_name:
        return generate_environment_panda_triple(exp_name)

def generate_environment_panda_orientation(exp_name):
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

    R_offset = np.zeros((3,3))
    R_offset[2,0] = 1.0
    R_offset[1,1] = -1.0
    R_offset[0,2] = 1.0
    constraint = OrientationConstraint(arm_names=model_info['arm_names'],
                                        arm_dofs=model_info['arm_dofs'],
                                        base_link=model_info['base_link'],
                                        axis=0,
                                        orientation_offset=R_offset,
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'],
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'])
    pc = constraint.planning_scene

    shelf_1_pos = np.array([0.8,0.0,0.75])
    add_shelf(pc, shelf_1_pos, 0, -1.572, 0.7, 0.5, 1.5, 0.02, 4, 'shelf_1')

    num_obs = 3
    start_pos_base = np.array([0.65, 0.0, 0.88]) 
    while True:
        q = np.random.uniform(constraint.lb, constraint.ub)
        r = constraint.solve_ik(q, start_pos_base)
        if r is False:
            continue
        if (q < constraint.lb).any() or (q > constraint.ub).any():
            continue
        if constraint.planning_scene.is_valid(q) is False:
            continue
        break

    pc.display(q)
    pc.add_cylinder('start', 0.1, 0.03, start_pos_base, [0,0,0,1])
    pc.attach_object('start', 'panda_hand',[])

    def update_scene_from_yaml(scene_data):
        for i in range(num_obs):
            pc.add_box('obs{}'.format(i), scene_data['obs{}'.format(i)]['dim'], scene_data['obs{}'.format(i)]['pos'], [1,0,0,0])

    return constraint, model_info, None, update_scene_from_yaml, None, q


def generate_environment_panda_dual(exp_name):
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

    if exp_name == 'panda_dual':
        constraint = MultiChainConstraint(arm_names=model_info['arm_names'],
                                            arm_dofs=model_info['arm_dofs'],
                                            base_link=model_info['base_link'],
                                            ee_links=model_info['ee_links'],
                                            hand_names=model_info['hand_names'],
                                            hand_joints=model_info['hand_joints'],
                                            hand_open=model_info['hand_open'],
                                            hand_closed=model_info['hand_closed'])
        
    elif exp_name == 'panda_dual_orientation':
        constraint = MultiChainFixedOrientationConstraint(arm_names=model_info['arm_names'],
                                            arm_dofs=model_info['arm_dofs'],
                                            axis=2,
                                            base_link=model_info['base_link'],
                                            ee_links=model_info['ee_links'],
                                            hand_names=model_info['hand_names'],
                                            hand_joints=model_info['hand_joints'],
                                            hand_open=model_info['hand_open'],
                                            hand_closed=model_info['hand_closed'])
    else:
        raise NotImplementedError

    pc = constraint.planning_scene
    def set_constraint(c):
        d1, d2, theta = c
        l = d1 + 2*d2*cos(theta)
        ly = l * sin(theta)
        lz = l * cos(theta)
        
        dt = pi - 2 * theta
        chain_pos = np.array([0.0, ly, lz])
        chain_rot = np.array([[1, 0, 0], [0, cos(dt), -sin(dt)], [0, sin(dt), cos(dt)]])
        chain_quat = R.from_matrix(chain_rot).as_quat()

        t1 = np.concatenate([chain_pos, chain_quat])
        constraint.set_chains([t1])
        pc.detach_object('tray', 'panda_2_hand_tcp')

        constraint.set_early_stopping(True)
        
        l_obj_z = d2 + d1/2 * cos(theta)
        l_obj_y = d1/2 * sin(theta)
        ee_to_obj_pos = np.array([0.0, l_obj_y, l_obj_z])
        obj_dt = -(pi/2 + theta)
        ee_to_obj_rot = np.array([[1, 0, 0], [0, cos(obj_dt), -sin(obj_dt)], [0, sin(obj_dt), cos(obj_dt)]])
        ee_to_obj_quat = R.from_matrix(ee_to_obj_rot).as_quat()

        q = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4, 0, 0, 0, -pi/2, 0, pi/2, pi/4])
        pos, quat = constraint.forward_kinematics('panda_arm_2', q[:7])
        T_0g = get_transform(pos, quat)
        T_go = get_transform(ee_to_obj_pos, ee_to_obj_quat)
        T_0o = np.dot(T_0g, T_go)
        np.set_printoptions(precision=5, suppress=True)
        obj_pos, obj_quat = get_pose(T_0o)

        pc.add_box('tray', [d1 * 3/4, d1, 0.01], obj_pos, obj_quat)
        pc.update_joints(q)
        pc.attach_object('tray', 'panda_2_hand_tcp', [])
        constraint.set_grasp_to_object_pose(go_pos=ee_to_obj_pos, go_quat=ee_to_obj_quat)
        return q


    c = np.array([0.3, 0.05, 0.9],dtype=np.float32)

    q_init = set_constraint(c)

    add_table(pc, 'table_1', [0.5,-0.6,0.3], 0, 0, 0.5, 0.9, 0.6, 0.05)
    add_table(pc, 'table_2', [0.5,0.6,0.3], 0, 0, 0.5, 0.9, 0.6, 0.05)

    num_obs = 3

    def update_scene_from_yaml(scene_data):
        for i in range(num_obs):
            pc.add_box('obs{}'.format(i), scene_data['obs{}'.format(i)]['dim'], scene_data['obs{}'.format(i)]['pos'], [1,0,0,0])

    return constraint, model_info, c, update_scene_from_yaml, set_constraint, q_init

def generate_environment_panda_triple(exp_name):
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)
    constraint = MultiChainConstraint(arm_names=model_info['arm_names'], 
                                        base_link=model_info['base_link'], 
                                        arm_dofs=model_info['arm_dofs'],
                                        ee_links=model_info['ee_links'],
                                        hand_names=model_info['hand_names'], 
                                        hand_joints=model_info['hand_joints'],
                                        hand_open=model_info['hand_open'],
                                        hand_closed=model_info['hand_closed'],
                                        planning_scene_name=model_info['planning_scene_name']) 
    cgc = ContinuousGraspCandidates(file_name='model/{exp_name}/{cont_grasp}'.format(exp_name=exp_name, cont_grasp=model_info['cont_grasp']))

    pc = constraint.planning_scene


    chair_pos = np.array([0.69, 0.44, 1.19])
    chair_quat = np.array([0.9238795, -0.3826834, 0, 0])
    pc.add_mesh('chair',model_info['mesh'], chair_pos, chair_quat) # X-180, Z-45 Euler

    q_init = np.array([-0.12904, 0.173413, -0.390121, -1.30219, 0.0913822, 1.36203, 1.03038, 
                    -1.53953, -1.64972, 2.00178, -2.66883, 0.633282, 3.66834, 0.562251, 
                    -0.790644, -1.40522, 1.81529, -2.61019, -0.242376, 2.49991, 1.26293])
    p_cg, q_cg = cgc.get_global_grasp(model_info['c_idx'][2], 0.5, chair_pos, chair_quat)
    r, q = constraint.solve_arm_ik('panda_left', q_init[0:7], p_cg, q_cg)
    q_init[0:7] = q
    pc.update_joints(q_init)
    pc.attach_object('chair', 'panda_left_hand', [])

    def set_constraint(y):
        # y: top right left ...
        constraint.planning_scene.detach_object('chair', 'panda_left_hand')
        p1, q1 = cgc.get_relative_transform(model_info['c_idx'][2], y[2], model_info['c_idx'][1], y[1])
        p2, q2 = cgc.get_relative_transform(model_info['c_idx'][2], y[2], model_info['c_idx'][0], y[0])
        p_cg, q_cg = cgc.get_global_grasp(model_info['c_idx'][2], y[2], chair_pos, chair_quat)
        t1 = np.concatenate([p1, q1])
        t2 = np.concatenate([p2, q2])
        constraint.set_chains([t1, t2])

        while True:
            r, q = constraint.solve_arm_ik('panda_left', q_init[0:7], p_cg, q_cg)
            if r: break

        q_init[0:7] = q
        pos, quat = cgc.get_grasp(model_info['c_idx'][2], y[2])
        T_og = get_transform(pos, quat)
        T_go = np.linalg.inv(T_og)
        constraint.set_grasp_to_object_pose(T_go=T_go)
        pc.update_object_pose('chair', chair_pos, chair_quat) # X-180, Z-45 Euler
        pc.update_joints(q_init)
        pc.attach_object('chair', 'panda_left_hand', [])

    c = np.array([0.3, 0.6, 0.3])

    set_constraint(c)

    pc.add_box('sub_table', np.array([0.37, 0.22, 0.165]), np.array([0.69, -0.04, 1.0826]), np.array([0, 0, 0, 1]))
    pc.add_box('sub_table2', np.array([0.22, 0.17, 0.165]), np.array([0.465, -0.505, 1.0826]), np.array([0, 0, 0, 1]))
    pc.add_box('sub_table3', np.array([0.17, 0.22, 0.165]), np.array([0.595, 0.355, 1.0826]), np.array([0, 0, 0, 1]))
    pc.add_box('sub_table4', np.array([0.22, 0.22, 0.165]), np.array([0.42, 0.1, 1.0826]), np.array([0, 0, 0, 1]))

    num_obs = 3

    def update_scene_from_yaml(scene_data):
        for i in range(num_obs):
            pc.add_box('obs{}'.format(i), 
                    scene_data['obs{}'.format(i)]['dim'], 
                    scene_data['obs{}'.format(i)]['pos'], [1,0,0,0])
            
    return constraint, model_info, c, update_scene_from_yaml, set_constraint, q_init