
import time
import numpy as np

from srmt.constraints.constraints import ConstraintBase

from ljcmp.models.latent_model import LatentModel, LatentValidityModel
from ljcmp.planning.motion_trees import LatentMotionTree
from ljcmp.planning.distance_functions import distance_z, distance_q
from ljcmp.planning.status_description import GrowStatus

from scipy.stats import norm
from scipy.special import ndtri
import copy
import random

class ConstrainedLatentBiRRT():
    def __init__(self, model : LatentModel, validity_model : LatentValidityModel=None, constraint : ConstraintBase=None, latent_dim=2, state_dim=3, validity_fn=None, latent_jump=True, start_region_fn=None, goal_region_fn=None) -> None:
        self.learned_model = model
        self.validity_model = validity_model

        self.constraint=constraint
        self.start_tree = LatentMotionTree(name='start_tree', model=model, multiple_roots=True if start_region_fn is not None else False)
        self.goal_tree = LatentMotionTree(name='goal_tree', model=model, multiple_roots=True if goal_region_fn is not None else False)
        self.start_q = None
        self.goal_q = None
        self.sampled_goals = []
        self.next_goal_index = 0
        self.distance_btw_trees = float('inf')
        self.distance_btw_trees_q = float('inf')
        self.use_latent_jump = latent_jump
        self.max_latent_jump_trials = 1
        self.solved = False
        self.off_manifold_threshold = 1.0
        self.p_q_plan = 0.001
        self.validity_fn = validity_fn
        if validity_fn is None:
            self.planning_scene = constraint.planning_scene

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        self.ub = constraint.ub
        self.lb = constraint.lb

        self.p_sample = 0.02
        self.region_timeout = 0.1

        self.start_region_fn = start_region_fn
        self.goal_region_fn = goal_region_fn
        
        # properties
        self.max_distance = 0.1 
        self.max_distance_q = 0.5
        self.solved_time = -1.0

        self.delta = 0.05
        self.lambda1 = 5 # curvature length ratio

        self.debug = False

    def print(self, *args):
        if self.debug:
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print(*args)
    

    def check_tree_validity(self, tree : LatentMotionTree, path_z, nodes):
        path_q = self.learned_model.to_state(path_z)
        header_changes = [] # [child, new parent node]
        aborted = False

        for i in range(len(path_z)):
            node = nodes[i]
            if node.checked is True:
                if node.valid is True:
                    path_q[i] = node.q
                    continue 

            else:   # if not checked
                manifold_distances = self.constraint.function(path_q[i])
                manifold_distance = np.linalg.norm(manifold_distances)
                if manifold_distance > self.off_manifold_threshold:
                    aborted = True
                    break
                
                r = self.constraint.project(path_q[i])
                if r is False:
                    aborted = True
                    break

                node.q = path_q[i]
                node.projected = True
                if self.is_valid(path_z[i], q=path_q[i], approx=False) is True:
                    node.q = path_q[i] # updated q (after projection)
                    dist_q = distance_q(nodes[i-1].q, node.q)

                    if dist_q > self.max_distance_q:
                        num_node_int = int(dist_q / self.max_distance_q)

                        q_ints = []
                        for j in range(num_node_int):
                            q_int, r = self.interpolate_q(nodes[i-1].q, node.q, (j+1)/(num_node_int+1))
                            if r is False or self.is_valid(q=q_int, approx=False) is False:
                                self.print('invalid path at', i, 'due to interpolation')
                                aborted = True
                                break
                            gs, _ = self.check_motion(nodes[i-1].q, q_int, end_checked=True)
                            if gs is GrowStatus.TRAPPED:
                                self.print('check_motion failed invalid path at', i, 'due to collision')
                                aborted = True
                                break
                            q_ints.append(q_int)
                        if aborted is True:
                            break
                        last_node = nodes[i-1]
                        z_ints = self.learned_model.to_latent(q_ints)
                        for q_int, z_int in zip(q_ints, z_ints):
                            last_node = tree.add_node(last_node, q=q_int, z=z_int, valid=True, checked=True, projected=True, ref='i')
                        header_changes.append([node, last_node]) 
                    else:
                        gs, _ = self.check_motion(nodes[i-1].q, node.q, end_checked=True)
                        if gs is GrowStatus.TRAPPED:
                            self.print('check_motion failed invalid path at', i, 'due to collision')
                            aborted = True
                            break

                    node.checked = True
                    node.valid = True
                    tree.q_nodes.append(node)
                else: # if not valid
                    # print(f'invalid path at {i}')
                    aborted = True
                    break
                
        for child, new_parent in header_changes:
            child.parent = new_parent
        
        if aborted:
            tree.delete_node(node)
            return False, nodes[i-1]
        
        return True, nodes[-1]

    def is_valid(self,z=None,q = None, approx=True):
        """if q is given, then, q should be projected
        """
        if approx:
            if self.validity_model is None:
                return True
            return self.validity_model.is_valid_estimated(z[None,:])[0]
            
        if q is None:
            q = self.learned_model.to_state(z)
            r = self.constraint.project(q)
            if r is False: 
                return False
        
        if (q <self.lb).any() or (q > self.ub).any():
            return False
        
        if self.validity_fn is not None:
            return self.validity_fn(q)

        return self.planning_scene.is_valid(q)

    def check_motion(self, start, end, end_checked=False):
        """check motion from start to end

        Args:
            start (numpy.array): start node
            end (numpy.array): end node

        Returns:
            GrowStatus: grow status
            list: geodesic states
        """
        if end_checked is False:
            if self.is_valid(q=end,approx=False) is False:
                return GrowStatus.TRAPPED, None
            if self.constraint.is_satisfied(end) is False:
                return GrowStatus.TRAPPED, None
        
        grow_status, geodesic_states, step_dists, distance_traveled = self.discrete_geodesic(start, end, True)

        return grow_status, geodesic_states
    
    
    def discrete_geodesic(self, start, end, validity_check_in_interpolation = True):
        """discrete geodesic from start to end

        Args:
            start (numpy.array): start node
            end (numpy.array): end node
            validity_check_in_interpolation (bool, optional): whether to check validity in interpolation. Defaults to True.

        Returns:
            GrowStatus: grow status
            list: geodesic states
            list: step distances
            float: total distance
        """
        tolerance = self.delta
        geodesic = [start]

        dist = distance_q(start, end)
        if dist <= tolerance:
            geodesic.append(end)
            step_dists = [dist]
            total_dist = dist
            return GrowStatus.REACHED, geodesic, step_dists, total_dist

        max = dist * self.lambda1

        previous = start
        step_dists = []
        total_dist = 0
        status = GrowStatus.TRAPPED
        while True:
            scratch, r = self.interpolate_q(previous, end, self.delta / dist)
            # r = self.constraint.project(scratch)

            if r is False: # project failed
                break

            if validity_check_in_interpolation and not self.is_valid(q=scratch, approx=False): # not valid (check only if not interpolating)
                break
            
            new_dist = distance_q(scratch, end)
            if new_dist >= dist: # went to backward
                break

            step_dist = distance_q(previous, scratch)
            if step_dist > self.lambda1 * self.delta: # too much deviation
                break
                
            total_dist += step_dist
            if total_dist > max:
                break

            step_dists.append(step_dist)

            dist = new_dist
            previous = scratch
            geodesic.append(copy.deepcopy(scratch))
            status = GrowStatus.ADVANCED
            
            if dist <= tolerance:
                return GrowStatus.REACHED, geodesic, step_dists, total_dist

        return status, geodesic, step_dists, total_dist

    def set_start(self, q):
        q = q.astype(np.double)
        self.start_q = q
        z = self.learned_model.to_latent(q)
        
        if self.start_tree.multiple_roots:
            self.start_tree.add_root_node(z, q)
        else:
            self.start_tree.set_root(z, q)

    def set_goal(self, q):
        q = q.astype(np.double)
        self.goal_q = q
        z = self.learned_model.to_latent(q)
        
        if self.goal_tree.multiple_roots:
            self.goal_tree.add_root_node(z, q)
        else:
            self.goal_tree.set_root(z, q) 

    def solve(self, max_time=10.0):
        self.start_time = time.time()
        is_start_tree = True
        self.print(self.distance_btw_trees)
        self.print(self.distance_btw_trees_q)
        self.print ('[z] Estimated distance to go: {0}'.format(self.distance_btw_trees))
        self.print ('[q] Estimated distance to go: {0}'.format(self.distance_btw_trees_q))
        self.terminate = False

        if self.start_q is None:
            if self.start_tree.multiple_roots:
                start_q = self.start_region_fn()
                self.set_start(start_q)
            else:
                raise ValueError('start_q is None, but start_tree.multiple_roots is False') 

        if self.goal_q is None:
            if self.goal_tree.multiple_roots:
                goal_q = self.goal_region_fn()
                self.set_goal(goal_q)
            else:
                raise ValueError('goal_q is None, but goal_tree.multiple_roots is False')

        while self.terminate is False: 
            if (time.time() - self.start_time) > max_time:
                self.print('timed out')
                self.terminate = True
                break

            if self.start_region_fn is not None:
                if self.start_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        start_q = self.start_region_fn(timeout=self.region_timeout)
                        if self.start_q is not None:
                            self.set_start(start_q)

                            self.print('added start', start_q)
            
            if self.goal_region_fn is not None:
                if self.goal_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        goal_q = self.goal_region_fn(timeout=self.region_timeout)
                        if goal_q is not None:
                            self.set_goal(goal_q)
                            self.print('added goal', goal_q)


            if is_start_tree:
                cur_tree = self.start_tree
                other_tree = self.goal_tree
            else:
                cur_tree = self.goal_tree
                other_tree = self.start_tree
            
            z_rand = self.random_sample()
            r, z_des, z_node_cur = self.grow(cur_tree, z_rand)

            if r != GrowStatus.TRAPPED:
                z_rand = copy.deepcopy(z_des)

                r, z_des, z_node = self.grow(other_tree, z_rand)
                
                if r == GrowStatus.TRAPPED:
                    continue

                while r == GrowStatus.ADVANCED:
                    if time.time() - self.start_time > max_time:
                        self.print('timed out')
                        self.terminate = True
                        break
                    
                    r, z_des, x_node = self.grow(other_tree, z_rand)
                
                if self.terminate:
                    break
                    
                z_near_other, z_other_node = other_tree.get_nearest(z_rand)
                new_dist = distance_z(z_rand, z_near_other)
                if new_dist < self.distance_btw_trees:
                    self.distance_btw_trees = new_dist
                    self.print ('[z] Estimated distance to go: {0}'.format(new_dist))

                if r == GrowStatus.REACHED:
                    reached_time = time.time()
                    elapsed = reached_time - self.start_time
                    self.print ('solution reached elapsed_time:',elapsed)
                    self.print ('validity check...')
                    cur_path_z, cur_nodes = cur_tree.get_path(z_node_cur)
                    cur_path_ref, _ = cur_tree.get_path_ref(z_node_cur)
                    
                    result, last_node_cur = self.check_tree_validity(cur_tree, cur_path_z, cur_nodes)
                    
                    if result is False:
                        if self.use_latent_jump:
                            res, node = self.latent_jump(cur_tree, other_tree, last_node_cur)
                            if res == GrowStatus.REACHED:
                                z_node_cur = node
                            else:
                                continue
                        else:
                            continue
                    other_path_z, other_nodes = other_tree.get_path(z_other_node)
                    other_path_ref, _ = other_tree.get_path_ref(z_other_node)
                    
                    result, last_node_other = self.check_tree_validity(other_tree, other_path_z, other_nodes)

                    if result is False:
                        if self.use_latent_jump:
                            res, node = self.latent_jump(other_tree, cur_tree, last_node_other)
                            if res == GrowStatus.REACHED:
                                z_other_node = node
                            else:
                                continue
                        else:
                            continue
                    self.end_time = time.time()
                    elapsed = self.end_time - self.start_time
                    validity_elapsed = self.end_time - reached_time
                    self.print ('[z] found a solution! elapsed_time:',elapsed)
                    self.print ('validity check elapsed_time:',validity_elapsed)
                    self.solved_time = elapsed
                    cur_path_z, cur_nodes = cur_tree.get_path(z_node_cur)
                    cur_path_q, _ = cur_tree.get_path_q(z_node_cur)
                    cur_path_ref, _ = cur_tree.get_path_ref(z_node_cur)
                    other_path_z, other_nodes = other_tree.get_path(z_other_node)
                    other_path_q, _ = other_tree.get_path_q(z_other_node)
                    other_path_ref, _ = other_tree.get_path_ref(z_other_node)

                    self.print('original_path1\n', z_node_cur.path)
                    self.print('original_path2\n', z_other_node.path)
                    
                    if is_start_tree:
                        q_path = np.concatenate((cur_path_q, np.flip(other_path_q,axis=0)), axis=0)
                        z_path = np.concatenate((cur_path_z, np.flip(other_path_z,axis=0)), axis=0)
                        ref_path = np.concatenate((cur_path_ref, np.flip(other_path_ref,axis=0)), axis=0)
                    else:
                        q_path = np.concatenate((other_path_q, np.flip(cur_path_q,axis=0)), axis=0)
                        z_path = np.concatenate((other_path_z, np.flip(cur_path_z,axis=0)), axis=0)
                        ref_path = np.concatenate((other_path_ref, np.flip(cur_path_ref,axis=0)), axis=0)

                    self.terminate = True
                    return True, z_path, q_path, ref_path

            if self.use_latent_jump:
                if self.p_q_plan > np.random.rand():
                    q_rand = self.random_sample_q()
                    self.grow_q(cur_tree, q_rand)

            is_start_tree = not is_start_tree

        return False, None, None, None

    def latent_jump(self, tree_a, tree_b, node_last):
        q_near, q_near_node = tree_b.get_nearest_q(node_last.q)
        res, q_new, node = self.grow_q(tree_a, q_near, node_last)
        if self.debug:
            self.print ('[z] latent jump')
        
        num_latent_jump_trials = 0
        while res != GrowStatus.TRAPPED:
            num_latent_jump_trials += 1

            if num_latent_jump_trials > self.max_latent_jump_trials:
                return res, node
            
            if res == GrowStatus.REACHED:
                return res, node

            z = self.learned_model.to_latent(q_new)
            q_recon = self.learned_model.to_state(z)

            # arrived to new latent region
            e = np.linalg.norm(q_recon - q_new)
            if e < self.off_manifold_threshold:
                return res, node

            res, q_new, node = self.grow_q(tree_a, q_near, node_last)
            node_last = node

        # If it trapped, add stochastic extension once 
        q_rand = self.random_sample_q()
        res, q_new, node = self.grow_q(tree_a, q_rand)
        return res, node
            
    def random_sample(self):
        if self.validity_model is not None:
            z = self.learned_model.sample_with_estimated_validity(1, self.validity_model)[0]
            return z

        z = np.random.normal(0, 1, self.latent_dim)
        
        return z

    def grow(self, tree, z):
        z_near, z_near_node = tree.get_nearest(z)
        reach = True
        z_des = z

        dist = distance_z(z_near, z)
        if dist > self.max_distance:
            z_int = self.interpolate(z_near, z, self.max_distance / dist) # TODO: check ratio

            z_des = z_int
            reach = False

        if self.is_valid(z_des) is False:
            return GrowStatus.TRAPPED, z_des, 0
        
        z_node = tree.add_node(z_near_node, z_des, ref='z')
        
        if reach:
            return GrowStatus.REACHED, z_des, z_node
        
        return GrowStatus.ADVANCED, z_des, z_node

    def interpolate(self, z_from, z_to, ratio):
        cdf_zs = norm.cdf([z_from, z_to]) 
        cdf_int = cdf_zs[0] * (1-ratio) + cdf_zs[1] * (ratio)
        z_int = ndtri(cdf_int)

        return z_int

    def random_sample_q(self):
        q = np.random.uniform(self.lb, self.ub)
        return q

    def grow_q(self, tree, q, given_q_near_node=None):
        if given_q_near_node is None:
            q_near, q_near_node = tree.get_nearest_q(q)
        else:
            q_near, q_near_node = given_q_near_node.q, given_q_near_node

        reach = True

        dist = distance_q(q_near, q)
        if dist > self.max_distance:
            q_int, r = self.interpolate_q(q_near, q, self.max_distance / dist) # TODO: check ratio
            
            if r == GrowStatus.TRAPPED:
                self.print('projection failed')
                return GrowStatus.TRAPPED, q_near, 0
            
            grow_status, geodesic_states = self.check_motion(q_near, q_int)

            if grow_status != GrowStatus.REACHED:
                return GrowStatus.TRAPPED, q_near, 0
                
            q_des = q_int
            reach = False
        else:
            q_pro = copy.deepcopy(q)
            r = self.constraint.project(q_pro)
            if r is False:
                return GrowStatus.TRAPPED, q_near, 0
            q_des = q_pro

        dist_after_projection = distance_q(q_des, q)
        if dist_after_projection >= dist:
            # went to backward...
            self.print('went to backward...')
            return GrowStatus.TRAPPED, q_des, 0

        if self.is_valid(q=q_des,approx=False) is False:
            return GrowStatus.TRAPPED, q_des, 0

        node = tree.add_node(q_near_node, q=q_des, 
                            checked=True, valid=True, projected=True, ref='q', z=self.learned_model.to_latent(q_des[None,:])[0])
        
        if reach:
            return GrowStatus.REACHED, q_des, node
        
        return GrowStatus.ADVANCED, q_des, node

    def interpolate_q(self, q_from, q_to, ratio):
        # returning q_from means interpolation failed
        q_int = (q_to-q_from) * ratio + q_from
        r = self.constraint.project(q_int)
        
        if r is False:
            return q_from, False
        return q_int, True
    