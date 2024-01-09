import numpy as np

import time
import numpy as np

from ljcmp.models.latent_model import LatentModel

from srmt.constraints.constraints import ConstraintBase

from ljcmp.planning.motion_trees import MotionTree
from ljcmp.planning.distance_functions import distance_q
from ljcmp.planning.status_description import GrowStatus

import copy
import rospy

import random

class ConstrainedBiRRT:
    def __init__(self, state_dim=3, constraint: ConstraintBase = None, validity_fn=None, start_region_fn=None, goal_region_fn=None) -> None:
        self.constraint=constraint
        self.start_tree = MotionTree(name='start_tree', multiple_roots=True if start_region_fn is not None else False, constraint=constraint)
        self.goal_tree = MotionTree(name='goal_tree', multiple_roots=True if goal_region_fn is not None else False, constraint=constraint)
        self.start_q = None
        self.goal_q = None
        self.sampled_goals = []
        self.next_goal_index = 0
        self.distance_btw_trees = float('inf')
        self.state_dim = state_dim
        self.ub = constraint.ub
        self.lb = constraint.lb
        self.solved_time = -1.0
        self.p_sample = 0.02
        self.validity_fn = validity_fn
        if validity_fn is None:
            self.planning_scene = constraint.planning_scene

        self.start_region_fn = start_region_fn
        self.goal_region_fn = goal_region_fn
        
        # properties
        self.max_distance = 0.5
        self.delta = 0.05
        self.lambda1 = 5 # curvature length ratio

        self.debug = True

    def print(self, *args):
        if self.debug:
            print(*args)
    
    def is_valid(self,q):
        if (q <self.lb).any() or (q > self.ub).any():
            return False
        
        if self.validity_fn is not None:
            return self.validity_fn(q)

        r = self.planning_scene.is_valid(q)
        
        return r
        
    def check_motion(self, start, end):
        """check motion from start to end

        Args:
            start (numpy.array): start node
            end (numpy.array): end node

        Returns:
            GrowStatus: grow status
            list: geodesic states
        """
        if self.is_valid(end) is False:
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

            if r is False: # project failed
                break

            if validity_check_in_interpolation and not self.is_valid(scratch): # not valid (check only if not interpolating)
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
        self.start_q = q

        if self.start_tree.multiple_roots:
            self.start_tree.add_root_node(q)
        else:
            self.start_tree.set_root(q)

    def set_goal(self, q):
        self.goal_q = q

        if self.goal_tree.multiple_roots:
            self.goal_tree.add_root_node(q)
        else:
            self.goal_tree.set_root(q)

    def solve(self, max_time=10.0):
        start_time = time.time()
        solved = False
        is_start_tree = True
        self.distance_btw_trees = float('inf')
        terminate = False

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

        while terminate is False:
            if (time.time() - start_time) > max_time:
                self.print('timed out')
                break
            
            # CBiRRT2
            if self.start_region_fn is not None:
                if self.start_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        start_q = self.start_region_fn()
                        self.start_tree.add_root_node(start_q)

                        self.print('added start', start_q)
            
            if self.goal_region_fn is not None:
                if self.goal_tree.multiple_roots:
                    if random.random() < self.p_sample:
                        goal_q = self.goal_region_fn()
                        self.goal_tree.add_root_node(goal_q)

                        self.print('added goal', goal_q)

            if is_start_tree:
                cur_tree = self.start_tree
                other_tree = self.goal_tree
            else:
                cur_tree = self.goal_tree
                other_tree = self.start_tree
            
            # sample
            q_rand = self.random_sample()
            r = self.constraint.project(q_rand)
            if r is False:
                continue

            r, q_des, q_idx_cur = self.grow(cur_tree, q_rand)

            if r != GrowStatus.TRAPPED:
                if r != GrowStatus.REACHED:
                    q_rand = copy.deepcopy(q_des)

                r, q_des, q_idx = self.grow(other_tree, q_rand)
                
                if r == GrowStatus.TRAPPED:
                    continue
                
                while r == GrowStatus.ADVANCED:
                    if rospy.is_shutdown():
                        raise (KeyboardInterrupt)
                    r, q_des, q_idx = self.grow(other_tree, q_rand)
                    
                q_near_other, q_idx_other = other_tree.get_nearest(q_rand)
                new_dist = distance_q(q_rand, q_near_other)
                if new_dist < self.distance_btw_trees:
                    self.distance_btw_trees = new_dist
                    self.print ('Estimated distance to go: {0}'.format(new_dist))

                if r == GrowStatus.REACHED:
                    
                    end_time = time.time()
                    elapsed = end_time - start_time
                    self.print ('found a solution! elapsed_time:',elapsed)
                    self.solved_time = elapsed

                    cur_path = cur_tree.get_path(q_idx_cur)
                    other_path = other_tree.get_path(q_idx_other)

                    if is_start_tree:
                        q_path = cur_path + list(reversed(other_path))
                    else:
                        q_path = other_path + list(reversed(cur_path))

                    q_path = np.array(q_path)
                    
                    return True, q_path
            is_start_tree = not is_start_tree
        return False, None

    def enforce_bounds(self, q):
        return np.clip(q, self.lb, self.ub)
    
    def random_sample(self):
        q = np.random.uniform(self.lb, self.ub)
        r = self.constraint.project(q)
        if r is False:
            return self.random_sample()
        
        return self.enforce_bounds(q)

    def grow(self, tree, q):
        q_near, q_near_idx = tree.get_nearest(q)
        reach = True

        dist = distance_q(q_near, q)
        if dist > self.max_distance:
            q_int, r = self.interpolate(q_near, q, self.max_distance / dist) # TODO: check ratio
            
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
            self.constraint.project(q_pro)
            q_des = q_pro

        dist_after_projection = distance_q(q_des, q)
        if dist_after_projection >= dist:
            # went to backward...
            self.print('went to backward...')
            return GrowStatus.TRAPPED, q_des, 0

        if self.is_valid(q_des) is False:
            return GrowStatus.TRAPPED, q_des, 0
        
        q_idx = tree.add_node(q_near_idx, q_des)
        
        if reach:
            return GrowStatus.REACHED, q_des, q_idx
        
        return GrowStatus.ADVANCED, q_des, q_idx

    def interpolate(self, q_from, q_to, ratio):
        # returning q_from means interpolation failed
        status, geodesic, step_dists, total_dist = self.discrete_geodesic(q_from, q_to, validity_check_in_interpolation=False)

        if status != GrowStatus.REACHED:
            return q_from, False
        
        q_int = self.geodesic_interpolate(geodesic, ratio)

        r = self.constraint.project(q_int)
        
        if r is False:
            return q_from, False
        return q_int, True
    
    def interpolate_q(self, q_from, q_to, ratio):
        # returning q_from means interpolation failed
        q_int = (q_to-q_from) * ratio + q_from
        r = self.constraint.project(q_int)
        
        if r is False:
            return q_from, False
        return q_int, True
    

    def geodesic_interpolate(self, geodesic, ratio):
        n = len(geodesic)
        dists = np.zeros(n)
        dists[0] = 0
        for i in range(1, n):
            dists[i] = dists[i-1] + distance_q(geodesic[i-1], geodesic[i])

        total_dist = dists[-1]
        target_dist = ratio * total_dist

        for i in range(1, n):
            if dists[i] >= target_dist:
                break

        if i == n-1:
            return geodesic[-1]
        t1 = dists[i] / total_dist - ratio
        if i < n-1:
            t2 = dists[i+1] / total_dist - ratio
        else:
            t2 = 1.0
        
        if t1 < t2:
            return geodesic[i]
        else:
            return geodesic[i+1]

class SampleBiasedConstrainedBiRRT(ConstrainedBiRRT):
    def __init__(self, model : LatentModel, state_dim=3,  constraint: ConstraintBase = None, validity_fn=None, start_region_fn=None, goal_region_fn=None) -> None:

        super().__init__(state_dim, constraint, validity_fn=validity_fn, start_region_fn=start_region_fn, goal_region_fn=goal_region_fn)
        
        self.config_queue = []
        self.max_config_queue_size = 1000
        self.left_config_queue_size = 0
        self.learned_model = model
        self.qnew_threshold = 5e-1

    def generate_samples(self):
        self.config_queue, _ = self.learned_model.sample(num_samples=self.max_config_queue_size)
        self.left_config_queue_size = len(self.config_queue)

    def random_sample(self):
        while True:
            if self.left_config_queue_size == 0:
                self.generate_samples()

            q = self.config_queue[self.left_config_queue_size-1] # N-1, N-2, ..., 0
            self.left_config_queue_size -= 1

            # if qnew satisfies constraints then
            if np.linalg.norm(self.constraint.function(q)) < self.qnew_threshold :
                # return qnew
                r = self.constraint.project(q)
                if r:
                    if (q <self.lb).any() or (q > self.ub).any():
                        continue

                    return q
