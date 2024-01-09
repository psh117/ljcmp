import numpy as np

import time
import numpy as np
import scipy.io

from enum import Enum

import networkx as nx

from ljcmp.models.latent_model import LatentModel

from srmt.constraints.constraints import ConstraintBase
from srmt.planning_scene.planning_scene import PlanningScene

from ljcmp.planning.distance_functions import distance_q
from ljcmp.planning.status_description import GrowStatus
from ljcmp.planning.abstract_planner import Planner

import copy

import random
import math

class ConstrainedPRM(Planner):
    def __init__(self, constraint : ConstraintBase, state_dim, start_region_fn=None, goal_region_fn=None, star=False) -> None:
        super().__init__(start_region_fn, goal_region_fn)

        self.planning_scene = constraint.planning_scene
        self.constraint=constraint

        self.max_random_bounce_steps = 5
        self.roadmap_build_time = 0.05
        self.k_nearest_neighbors = 10
        self.r_nearest_neighnors = 0.5

        self.state_dim = state_dim

        self.graph = nx.Graph()
        self.start_list = []
        self.goal_list = []

        self.delta = 0.05
        self.lambda1 = 5 # curvature length ratio

        self.v_start = self.add_node(q=None, weight=0.0, type='start')
        self.v_goal = self.add_node(q=None, weight=0.0, type='goal')    
        self.edge_paths = {}

        self.star = star
        self.k_prm_const = math.e + math.e / self.state_dim

        self.p_start_sample = 0.05
        self.p_goal_sample = 0.05

    def set_graph(self, graph):
        self.graph = copy.deepcopy(graph)

        self.v_start = self.add_node(q=None, weight=0.0, type='start')
        self.v_goal = self.add_node(q=None, weight=0.0, type='goal')    
    
    def add_node(self, q, weight=1.0, type='int'):
        node_cnt = len(self.graph.nodes)
        self.graph.add_node(node_cnt, q=q, weight=weight, type=type, total_connection_attempts=1, successful_connection_attempts=0)

        return node_cnt
    
    def set_start(self, q):
        self.add_start(q)

    def set_goal(self, q):
        self.add_goal(q)

    def add_start(self, q):
        v = self.add_node(q, weight=0.0, type='start')
        self.graph.add_edge(v, self.v_start, weight=0.0, path=[q])
        self.start_list.append(v)

    def add_goal(self, q):
        v = self.add_node(q, weight=0.0, type='goal')
        self.graph.add_edge(v, self.v_goal, weight=0.0, path=[q])
        self.goal_list.append(v)

    def solve(self, max_time=10, star=False):
        """solve

        Args:
            max_time (int or list, optional): time limits. Defaults to 10.
            star (bool, optional): optimal planning. Defaults to False.

        Returns:
            results (bool), path (np.array), path_vertices (list), length: planning results, path, path vertices, path length
        """
        self.star = star
        self.best_cost = np.inf
        self.iteration = 0
        start_time = time.time()
        cnt = 0
        single_time_limit = False

        if len(self.start_list) == 0:
            self.add_start(self.start_region_fn())
        if len(self.goal_list) == 0:
            self.add_goal(self.goal_region_fn())
            
        if type(max_time) is not list:
            max_time = [max_time]
            single_time_limit = True
        
        r_list = []
        path_list = []
        path_node_list = []
        length_list = []

        for time_limit_now in max_time:
            while time.time() - start_time < time_limit_now:
                self.grow(2 * self.roadmap_build_time)
                self.expand(self.roadmap_build_time)
                cnt += 1

                if not star:
                    if cnt % 3 == 0:
                        r, path, length = self.check_for_solution()
                        self.solved_time = time.time() - start_time
                        if r:
                            import pdb; pdb.set_trace()
                            return r, self.get_solution(path), length

            if star:
                r, path, length = self.check_for_solution()
                if r:
                    self.solved_time = time.time() - start_time
                    if single_time_limit:
                        return r, self.get_solution(path), length
                
                    r_list.append(r)
                    path_list.append(self.get_solution(path))
                    path_node_list.append(path)
                    length_list.append(length)
                else:
                    r_list.append(False)
                    path_list.append(None)
                    path_node_list.append(None)
                    length_list.append(0.0)

        if single_time_limit:    
            return False, None, None, 0.0
    
        return r_list, path_list, path_node_list, length_list

    def grow(self, time_limit=10):
        cur_time = time.time()
        while time.time() - cur_time < time_limit:
            self.iteration += 1

            if self.start_region_fn is not None:
                if random.random() < self.p_start_sample:
                    q = self.start_region_fn()
                    self.add_start(q)
                    self.add_milestone(self.start_list[-1])

            if self.goal_region_fn is not None:
                if random.random() < self.p_goal_sample:
                    q = self.goal_region_fn()
                    self.add_goal(q)
                    self.add_milestone(self.goal_list[-1])
        
            q = self.random_sample()
            r = self.constraint.project(q)
            if r is False:
                continue
            if self.is_valid(q):
                v = self.add_node(q) 
                self.add_milestone(v)

    def expand(self, time_limit=10):
        cur_time = time.time()
        
        while time.time() - cur_time < time_limit:
            self.iteration += 1
            v = random.choice(list(self.graph.nodes))
            states = self.random_bounce_motion(self.graph.nodes[v]['q'], self.max_random_bounce_steps)
            
            if (len(states) == 0):
                continue

            v_last = self.add_node(q=states[-1][-1]) 
            self.add_milestone(v_last) 
            
            for state in states[:-1]:
                v_s = self.add_node(q=state[-1])
                self.graph.add_edge(v, v_s, weight=self.motion_cost(v,v_s), path=state)
                v = v_s

            self.graph.add_edge(v, v_last, weight=self.motion_cost(v,v_last), path=states[-1]) 


    def motion_cost(self, v1, v2):
        return distance_q(self.graph.nodes[v1]['q'], self.graph.nodes[v2]['q'])
    
    def add_milestone(self, v):
        neighnors = self.get_nearest_neighbors(self.graph.nodes[v]['q']) # connection strategy
        for v_n in neighnors:
            self.graph.nodes[v]['total_connection_attempts'] += 1
            self.graph.nodes[v_n]['total_connection_attempts'] += 1
            r, geodesic_states = self.check_motion(self.graph.nodes[v_n]['q'], self.graph.nodes[v]['q'])
            
            if r != GrowStatus.REACHED:
                continue

            self.graph.nodes[v]['successful_connection_attempts'] += 1
            self.graph.nodes[v_n]['successful_connection_attempts'] += 1
            self.graph.add_edge(v, v_n, weight=self.motion_cost(v,v_n), path=geodesic_states)

        return v
    
    def get_nearest_neighbors(self, q):
        return self.get_k_nearest_neighbors(q)
    
    def get_k_nearest_neighbors(self, q):
        if self.star:
            k = int(self.k_prm_const * math.log(self.graph.number_of_nodes()))
        else:
            k = self.k_nearest_neighbors
        # start and goal are always connected
        dists = [distance_q(q, self.graph.nodes[n]['q']) for n in self.graph.nodes]
        sorted_idx = np.argsort(dists)
        return sorted_idx.tolist()[:k]
    
    def get_r_nearest_neighbors(self, q):
        r = self.r_nearest_neighnors
        nn = [i for i, n in enumerate(self.graph.nodes) if distance_q(q, self.graph.nodes[n]['q']) < r]
        return nn

    def sample_valid(self):
        while True:
            q = self.random_sample()
            r = self.constraint.project(q)
            
            if r and self.is_valid(q):
                return q
            
    def random_sample(self):
        return self.constraint.sample()
    
    def random_bounce_motion(self, start, steps):
        """random bounce motion from start

        Args:
            start (numpy.array): start node
            steps (int): number of steps

        Returns:
            list: generated bounce states
        """

        states = []
        prev = start

        i = 0
        for _ in range(steps):
            q = self.sample_valid()

            grow_status, grow_states = self.check_motion(prev, q)
            if grow_status == GrowStatus.TRAPPED:
                continue
            
            states.append(grow_states) # get last state
            prev = grow_states[-1]
            i += 1

        return states
 
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
        
        grow_status, geodesic_states, step_dists, distance_traveled = self.discrete_geodesic(start, end)

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
            scratch = self.interpolate(previous, end, self.delta / dist)
            r = self.constraint.project(scratch)

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

    def interpolate(self, start, end, alpha):
        return start + alpha * (end - start)

    def is_valid(self, q):
        return self.constraint.planning_scene.is_valid(q)
    
    def check_for_solution(self):
        try:
            path = nx.astar_path(self.graph, self.v_start, self.v_goal, heuristic=self.heuristic, weight='weight')
            length = sum(self.graph[u][v].get('weight', 1) for u, v in zip(path[:-1], path[1:]))
            import pdb; pdb.set_trace()
            return True, path[1:-1], length # remove dummy start and goal
        except nx.NetworkXNoPath:
            return False, None, 0

    def heuristic(self, v1, v2):
        return distance_q(self.graph.nodes[v1]['q'], self.graph.nodes[v2]['q'])
    

    def get_solution(self, path):
        v0 = path[0]
        total_path = [self.graph.nodes[v0]['q']]

        for v in path[1:]:
            q_start = self.graph.nodes[v0]['q']
            q_end = self.graph.nodes[v]['q']
            local_path = self.graph[v0][v]['path']
            if np.linalg.norm(local_path[0]-q_start) > np.linalg.norm(local_path[-1]-q_start):
                local_path.reverse()
            
            total_path += local_path[1:]
            v0 = v

        total_path = np.array(total_path)
        return total_path
    