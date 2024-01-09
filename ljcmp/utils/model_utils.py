
import torch
import os
import yaml
import tqdm
import copy
import time
import numpy as np
import pickle
import networkx as nx

from ljcmp.models import TSVAE
from ljcmp.models.validity_network import VoxelValidityNet

from ljcmp.planning.sample_region import RegionSampler, LatentRegionSampler

from ljcmp.planning.constrained_bi_rrt import SampleBiasedConstrainedBiRRT
from ljcmp.planning.precomputed_roadmap import PrecomputedRoadmap, PrecomputedGraph
from ljcmp.planning.constrained_bi_rrt_latent_jump import ConstrainedLatentBiRRT
from ljcmp.utils.time_parameterization import time_parameterize

from scipy.linalg import null_space

from termcolor import colored

import multiprocessing as mp

def load_model(exp_name, model_info, load_validity_model=False):
    constraint_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name, 
                                                                    model_path=model_info['constraint_model']['path'])
    
    model_type = constraint_model_path.split('.')[-1]
    tag = model_info['constraint_model']['tag']

    if model_type == 'pt':
        constraint_model = torch.load(constraint_model_path)

    elif model_type == 'ckpt':
        constraint_model_checkpoint = torch.load(constraint_model_path)
        constraint_model_state_dict = constraint_model_checkpoint['state_dict']
        constraint_model = TSVAE(x_dim=model_info['x_dim'], 
                                 h_dim=model_info['constraint_model']['h_dim'], 
                                 z_dim=model_info['z_dim'], 
                                 c_dim=model_info['c_dim'], 
                                 null_augment=False)
        
        for key in list(constraint_model_state_dict):
            constraint_model_state_dict[key.replace("model.", "")] = constraint_model_state_dict.pop(key)

        constraint_model.load_state_dict(constraint_model_state_dict)
        # save pt
        os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
        torch.save(constraint_model, 'model/{exp_name}/weights/{tag}/constraint_model.pt'.format(exp_name=exp_name, tag=tag))

    else:
        raise NotImplementedError
    
    constraint_model.eval()

    tag = model_info['voxel_validity_model']['tag']
    validity_model_path = 'model/{exp_name}/{model_path}'.format(exp_name=exp_name, 
                                                                  model_path=model_info['voxel_validity_model']['path'])
    validity_model = VoxelValidityNet(z_dim=model_info['z_dim'], 
                                      c_dim=model_info['c_dim'], 
                                      h_dim=model_info['voxel_validity_model']['h_dim'],
                                      voxel_latent_dim=model_info['voxel_validity_model']['voxel_latent_dim'])

    if load_validity_model:
        validity_model_type = validity_model_path.split('.')[-1]
        if validity_model_type == 'pt':
            validity_model = torch.load(validity_model_path)

        elif validity_model_type == 'ckpt':
            validity_model_checkpoint = torch.load(validity_model_path)
            validity_model_state_dict = validity_model_checkpoint['state_dict']
            validity_model_state_dict_z_model = {}
            for key in list(validity_model_state_dict):
                if key.startswith('model.'):
                    validity_model_state_dict_z_model[key.replace("model.", "")] = validity_model_state_dict.pop(key)

            validity_model.load_state_dict(validity_model_state_dict_z_model)
            # save pt
            os.makedirs('model/{exp_name}/weights/{tag}'.format(exp_name=exp_name, tag=tag), exist_ok=True)
            torch.save(validity_model, 'model/{exp_name}/weights/{tag}/voxel_validity_model.pt'.format(exp_name=exp_name, tag=tag))

        else:
            raise NotImplementedError

        validity_model.threshold = model_info['voxel_validity_model']['threshold']
    else:
        validity_model.threshold = 0.0

    validity_model.eval()

    return constraint_model, validity_model


def generate_constrained_config(constraint_setup_fn, exp_name,
                                workers_seed_range=range(0,2), dataset_size=30000, 
                                samples_per_condition=10,
                                save_top_k=1, save_every=1000, timeout=1.0,
                                fixed_condition=None,
                                display=False, display_delay=0.5):
    save_dir = f'dataset/{exp_name}/manifold/'
    model_info = yaml.load(open('model/{exp_name}/model_info.yaml'.format(exp_name=exp_name), 'r'), Loader=yaml.FullLoader)

    dataset_size_per_worker = dataset_size // len(workers_seed_range)
    
    def generate_constrained_config_worker(seed, pos):
        np.random.seed(seed)

        save_dir_local = os.path.join(save_dir, str(seed))
        os.makedirs(save_dir_local, exist_ok=True)

        tq = tqdm.tqdm(total=dataset_size_per_worker, position=pos,desc='Generating dataset for seed {}'.format(seed))
        q_dataset = []
        jac_dataset = []
        null_dataset = []
        constraint, set_constraint_by_condition = constraint_setup_fn()
        pc = constraint.planning_scene

        if fixed_condition is not None:
            set_constraint_by_condition(fixed_condition)
            c = fixed_condition

        while len(q_dataset) < dataset_size_per_worker:
            if fixed_condition is None:
                c = np.random.uniform(model_info['c_lb'], model_info['c_ub'])
                set_constraint_by_condition(c)

            q = constraint.sample_valid(pc.is_valid, timeout=timeout)
            
            if q is False:
                continue

            if (q > constraint.ub).any() or (q < constraint.lb).any():
                continue
            
            jac = constraint.jacobian(q)
            null = null_space(jac)
            q_dataset.append(np.concatenate((c,q)))
            jac_dataset.append(jac)
            null_dataset.append(null)
            
            tq.update(1)

            for _ in range(samples_per_condition-1):
                q = constraint.sample_valid(pc.is_valid, timeout=timeout)

                if q is False:
                    continue

                if (q > constraint.ub).any() or (q < constraint.lb).any():
                    continue

                jac = constraint.jacobian(q)
                null = null_space(jac)
                q_dataset.append(np.concatenate((c,q)))
                jac_dataset.append(jac)
                null_dataset.append(null)
                
                tq.update(1)

                if display:
                    pc.display(q)
                    time.sleep(display_delay)
                
                if save_every > 0:
                    if len(q_dataset) % save_every == 0:
                        current_len = len(q_dataset)
                        delete_len = current_len - save_every*save_top_k
                        try:
                            np.save(f'{save_dir_local}/data_{seed}_{current_len}.npy', np.array(q_dataset))
                            np.save(f'{save_dir_local}/null_{seed}_{current_len}.npy', np.array(null_dataset))
                            
                            if delete_len > 0:
                                os.remove(f'{save_dir_local}/data_{seed}_{delete_len}.npy')
                                os.remove(f'{save_dir_local}/null_{seed}_{delete_len}.npy')
                        except:
                            print('save failed')
                    break

        np.save(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy', np.array(q_dataset[:dataset_size_per_worker]))
        np.save(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy', np.array(null_dataset[:dataset_size_per_worker]))
        tq.close()

    p_list = []
    for pos, seed in enumerate(workers_seed_range):
        p = mp.Process(target=generate_constrained_config_worker, args=(seed, pos))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    print('Merge dataset')

    data_list = []
    null_list = []
    for seed in workers_seed_range:
        save_dir_local = os.path.join(save_dir, str(seed))
        data_list.append(np.load(f'{save_dir_local}/data_{seed}_{dataset_size_per_worker}.npy'))
        null_list.append(np.load(f'{save_dir_local}/null_{seed}_{dataset_size_per_worker}.npy'))
    
    data = np.concatenate(data_list)
    null = np.concatenate(null_list)

    if fixed_condition is not None:
        np.save(f'{save_dir}/data_fixed_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_fixed_{dataset_size}.npy', null)
    else:
        np.save(f'{save_dir}/data_{dataset_size}.npy', data)
        np.save(f'{save_dir}/null_{dataset_size}.npy', null)



    print('Done')

def generate_scene_config(constraint, constraint_model, model_info, condition, update_scene_from_yaml, start=0, end=500, config_size=100):
    save_dir = f"dataset/{model_info['name']}/scene_data"
    os.makedirs(save_dir, exist_ok=True)

    tq_scene = tqdm.tqdm(range(start, end))
    for cnt in tq_scene:
        tq_scene.set_description('scene: {:04d}'.format(cnt))
        save_dir_local = '{}/scene_{:04d}'.format(save_dir, cnt)
        if os.path.exists(f'{save_dir_local}/scene.yaml'):

            scene_data = yaml.load(open(f'{save_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
            update_scene_from_yaml(scene_data)
            invalid_by_projection = 0
            invalid_by_out_of_range = 0
            invalid_by_collision = 0
            valid_sets = []
            invalid_sets = []
            tq = tqdm.tqdm(total=config_size, leave=False)
            
            while len(valid_sets) < config_size or len(invalid_sets) < config_size:
                xs, zs = constraint_model.sample(100)    
                for x, z in zip(xs, zs):
                    r = constraint.project(x)

                    if r is False:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_projection += 1
                        continue
                    
                    if (x < constraint.lb).any() or (x > constraint.ub).any():
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z,x,condition))
                        invalid_by_out_of_range += 1
                        continue

                    r = constraint.planning_scene.is_valid(x)
                    if r:
                        if len(valid_sets) < config_size:
                            valid_sets.append((z, x, condition))
                            tq.update(1)
                    else:
                        if len(invalid_sets) < config_size:
                            invalid_sets.append((z, x, condition))
                        invalid_by_collision += 1
                    
                    tq.set_description(f'valid: {len(valid_sets)}, invalid: {len(invalid_sets)} (P: {invalid_by_projection}, R: {invalid_by_out_of_range}, C: {invalid_by_collision})')
            tq.close()
            save_dir_local_tag = '{}/{}'.format(save_dir_local, model_info['constraint_model']['tag'])
            os.makedirs(save_dir_local_tag, exist_ok=True)
            pickle.dump({'valid_set':valid_sets, 'invalid_set':invalid_sets}, open(f'{save_dir_local_tag}/config.pkl', 'wb'))
                    
        else:
            print('scene {} not found'.format(cnt))
            break
            

def benchmark(exp_name, model_info, method, update_scene_from_yaml, 
              constraint, device='cpu', condition=None, max_time=500.0,
              use_given_start_goal=False, debug=False, display=False,
              load_validity_model=True, 
              trials=1, test_scene_start_idx=500, num_test_scenes=100):
    """benchmark function

    Args:
        exp_name (str): experiment name
        model_info (dict): model information
        method (str): method name (e.g. 'latent_rrt', 'latent_rrt_latent_jump', 'sampling_rrt', and 'precomputed_graph_rrt')
        update_scene_from_yaml (function): function to update scene from yaml file
        constraint (ConstraintBase): constraint
        device (str, optional): device. Defaults to 'cpu'.
        condition (dict, optional): condition. Defaults to None.
        max_time (float, optional): maximum planning time. Defaults to 500.0.
        use_given_start_goal (bool, optional): use given start and goal. Defaults to False.
        debug (bool, optional): debug mode. Defaults to False.
        display (bool, optional): display mode. Defaults to False.
        load_validity_model (bool, optional): load validity model. Defaults to True.
        trials (int, optional): number of trials. Defaults to 1.
        test_scene_start_idx (int, optional): test scene start index. Defaults to 500.
        num_test_scenes (int, optional): number of test scenes. Defaults to 100.
    """
    # ready for model
    constraint_model, validity_model = load_model(exp_name, model_info, 
                                                load_validity_model=load_validity_model)
    
    constraint_model.to(device=device)
    validity_model.to(device=device)

    if condition is not None:
        constraint_model.set_condition(condition)
        validity_model.set_condition(condition)

    # warm up
    z_dim = model_info['z_dim']
    z = torch.normal(mean=torch.zeros([constraint_model.default_batch_size, z_dim]), 
                    std=torch.ones([constraint_model.default_batch_size, z_dim])).to(device=device)
    _ = validity_model(z)


    if 'precomputed_roadmap' in method:
        tag = model_info['precomputed_roadmap']['tag']

        precomputed_roadmap_path = os.path.join('model', 
                                                exp_name, 
                                                model_info['precomputed_roadmap']['path'])
        
        precomputed_roadmap = nx.read_gpickle(precomputed_roadmap_path)

        print(colored('precomputed_roadmap tag: ', 'green'), tag)
        print(colored('precomputed_roadmap path: ', 'green'), precomputed_roadmap_path)
        print(colored('precomputed_roadmap nodes: ', 'green'), len(precomputed_roadmap.nodes))
        print(colored('precomputed_roadmap edges: ', 'green'), len(precomputed_roadmap.edges))

    if 'precomputed_graph' in method:
        tag = model_info['precomputed_graph']['tag']

        precomputed_graph_path = os.path.join('dataset',
                                               exp_name,
                                               model_info['precomputed_graph']['path'])
        
        configs = np.load(precomputed_graph_path)
        
        planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
        planner.from_configs(configs[:, model_info['c_dim']:])
        
        precomputed_graph = planner.graph
        
    # benchmark
    scene_dir = f'dataset/{exp_name}/scene_data'

    test_range = range(test_scene_start_idx, test_scene_start_idx + num_test_scenes)
    
    test_times = []
    test_paths = []
    test_path_lenghts = []
    test_paths_z = [] # only for latent_rrt
    test_path_refs = [] # only for latent_rrt
    test_suc_cnt = 0
    test_cnt = 0

    print(colored('test_range: ', 'green'), test_range)

    tq = tqdm.tqdm(test_range, position=0)

    for i in tq:
        scene_dir_local = '{}/scene_{:04d}'.format(scene_dir, i)
        if not os.path.exists(f'{scene_dir_local}/scene.yaml'):
            print(f'{scene_dir_local}/scene.yaml not exist')
            break

        scene_data = yaml.load(open(f'{scene_dir_local}/scene.yaml', 'r'), Loader=yaml.FullLoader)
        update_scene_from_yaml(scene_data)

        scene = yaml.load(open(os.path.join(scene_dir_local,  'scene.yaml'), 'r'), Loader=yaml.FullLoader)
        voxel = np.load(os.path.join(scene_dir_local, 'voxel.npy')).flatten()

        start_q = np.loadtxt(os.path.join(scene_dir_local, 'start_q.txt'))
        goal_q = np.loadtxt(os.path.join(scene_dir_local, 'goal_q.txt'))
        validity_model.set_voxel(voxel)

        start_pose = np.array(scene['start_pose'])
        goal_pose = np.array(scene['goal_pose'])

        inner_tq = tqdm.tqdm(range(trials), position=1, leave=False)

        latent_jump = False
        if 'latent_jump' in method:
            latent_jump = True

        for trial in inner_tq:
            if 'latent_rrt' in method:
                if use_given_start_goal:
                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=latent_jump)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)

                    planner = ConstrainedLatentBiRRT(constraint_model, validity_model, constraint, latent_jump=latent_jump,
                                                    start_region_fn=lrs_start.sample, 
                                                    goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q'] / model_info['planning']['alpha']
                planner.max_distance_q = model_info['planning']['max_distance_q']
                planner.off_manifold_threshold = model_info['planning']['off_manifold_threshold']
                planner.p_q_plan = model_info['planning']['p_q_plan']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.max_latent_jump_trials = model_info['planning']['max_latent_jump_trials']
                planner.debug = debug
                r, z_path, q_path, path_ref = planner.solve(max_time=max_time)
                

            elif method == 'sampling_rrt':
                if use_given_start_goal:
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model, constraint=constraint)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    lrs_start = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_start.set_target_pose(start_pose)
                    lrs_goal = LatentRegionSampler(constraint_model, constraint, validity_model)
                    lrs_goal.set_target_pose(goal_pose)
                    planner = SampleBiasedConstrainedBiRRT(state_dim=model_info['x_dim'], model=constraint_model, constraint=constraint,
                                                        start_region_fn=lrs_start.sample, 
                                                        goal_region_fn=lrs_goal.sample)
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.qnew_threshold = model_info['planning']['qnew_threshold']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                planner.debug = debug
                r, q_path = planner.solve(max_time=max_time)

            elif method == 'precomputed_roadmap_prm':
                if use_given_start_goal:
                    planner = PrecomputedRoadmap(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_roadmap)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                r, q_path = planner.solve(max_time=max_time)


            elif method == 'precomputed_graph_rrt':
                if use_given_start_goal:
                    planner = PrecomputedGraph(state_dim=model_info['x_dim'], constraint=constraint)
                    planner.set_graph(graph=precomputed_graph)
                    planner.set_start(start_q)
                    planner.set_goal(goal_q)
                else:
                    raise NotImplementedError
                
                planner.debug = debug
                planner.max_distance = model_info['planning']['max_distance_q']
                planner.delta = model_info['planning']['delta']
                planner.lambda1 = model_info['planning']['lambda']
                r, q_path = planner.solve(max_time=max_time)
            
            else:
                raise NotImplementedError

            if debug:
                print('planning time', planner.solved_time)

            test_cnt += 1
            
            if r is True:
                path_length = np.array([np.linalg.norm(q_path[i+1] - q_path[i]) for i in range(len(q_path)-1)]).sum()
                test_suc_cnt += 1
                solved_time = planner.solved_time
            else:
                if debug:
                    print('failed to find a path')
                q_path = None
                z_path = None
                path_ref = None

                solved_time = -1.0
                path_length = -1.0

            test_paths.append(q_path)
            test_times.append(solved_time)
            test_path_lenghts.append(path_length)

            if 'latent_rrt' in method:
                test_paths_z.append(z_path)
                test_path_refs.append(path_ref)

            mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
            mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
            
            tq.set_description('test suc rate: {:.3f}, avg time: {:.3f}, avg path length: {:.3f}'.format(test_suc_cnt/test_cnt, mean_test_times, mean_test_path_lenghts))

            if display:
                hz = 20
                duration, qs_sample, qds_sample, qdds_sample, ts_sample = time_parameterize(q_path, model_info, hz=hz)
                
                if debug:
                    print('duration', duration)

                for q in qs_sample:
                    slow_down = 3.0
                    constraint.planning_scene.display(q)
                    time.sleep(1.0/hz * slow_down)
                
    test_paths_cartesian = []
    for path in test_paths:
        if path is None:
            test_paths_cartesian.append(None)
            continue
        
        path_cartesian = []
        for q in path:
            cur_idx = 0
            cartesian_vector = []
            for arm_name, dof in zip(constraint.arm_names, constraint.arm_dofs):
                pos, quat = constraint.forward_kinematics(arm_name, q[cur_idx:cur_idx+dof])
                cur_idx += dof
                cartesian_vector.append(np.concatenate([pos, quat]))
                
            cartesian_vector = np.concatenate(cartesian_vector)
            path_cartesian.append(cartesian_vector)
        test_paths_cartesian.append(path_cartesian)


    mean_test_times = np.mean(test_times, where=np.array(test_times) > 0)
    std_test_times = np.std(test_times, where=np.array(test_times) > 0)
    mean_test_path_lenghts = np.mean(test_path_lenghts, where=np.array(test_path_lenghts) > 0)
    std_test_path_lenghts = np.std(test_path_lenghts, where=np.array(test_path_lenghts) > 0)

    ret =  {'experiment_name': exp_name,
            'model_tag_name': model_info['constraint_model']['tag'],
            'method': method,
            'use_given_start_goal': use_given_start_goal,
            'max_time': max_time,

            # test scene info
            'test_scene_start_idx': test_scene_start_idx,
            'test_scene_cnt': num_test_scenes,
            
            # result overview
            'test_cnt': test_cnt, 
            'test_suc_cnt': test_suc_cnt, 
            'success_rate': test_suc_cnt/test_cnt,
            'mean_test_times': mean_test_times,
            'std_test_times': std_test_times,
            'mean_test_path_lenghts': mean_test_path_lenghts,
            'std_test_path_lenghts': std_test_path_lenghts,

            # result details
            'test_times': test_times, 
            'test_paths': test_paths, 
            'test_path_lenghts': test_path_lenghts, 
            'test_paths_cartesian': test_paths_cartesian}
    
    if 'latent_rrt' in method:
        ret['test_paths_z'] = test_paths_z
        ret['test_path_refs'] = test_path_refs

    return ret