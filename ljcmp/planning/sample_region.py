
import time
import copy
import numpy as np

from srmt.constraints.constraints import ConstraintBase, ConstraintIKBase
from ljcmp.models.latent_model import LatentModel, LatentValidityModel

from queue import PriorityQueue
from itertools import count

class RegionSampler(object):
    def __init__(self, constraint:ConstraintIKBase) -> None:
        self.constraint = constraint
        self.pose = None
        self.target_pose_update = False

    def set_target_pose(self, pose):
        self.pose = pose
        self.target_pose_update = True

    def sample_from_q0(self, q0):
        assert self.pose is not None, 'set pose first'

        if self.target_pose_update is False:
            return None

        self.target_pose_update = False
        
        r = self.constraint.solve_ik(q0, self.pose)

        if r is False:
            return None
        
        if (q0 < self.constraint.lb).any() or (q0 > self.constraint.ub).any() == False:
            if self.constraint.planning_scene.is_valid(q0) is True:
                return q0
            
        return None
    
    def sample(self, timeout=-1.0, q0=None):
        assert self.pose is not None, 'set pose first'

        start_time = time.time()

        q0_ik = self.sample_from_q0(q0)
        
        if q0_ik is not None:
            return q0_ik
        
        self.constraint.update_target(self.pose)
        while True:
            if timeout > 0:
                if time.time() - start_time > timeout:
                    return None
            q = np.random.uniform(self.constraint.lb, self.constraint.ub)
            r = self.constraint.solve_ik(q, self.pose)
            if r is False:
                continue
            
            if (q < self.constraint.lb).any() or (q > self.constraint.ub).any():
                continue
            if self.constraint.planning_scene.is_valid(q) is False:
                continue
            return q
        
class LatentRegionSampler(RegionSampler):
    def __init__(self, constraint_model:LatentModel, constraint:ConstraintIKBase, validity_model:LatentValidityModel, model_sample_count=512, new_config_threshold=1.5) -> None:
        super().__init__(constraint)
        self.priority_queue = PriorityQueue()
        self.unique = count()
        self.constraint_model = constraint_model
        self.validity_model = validity_model
        self.model_sample_count = model_sample_count
        self.new_config_threshold = new_config_threshold
        self.debug = False

    def set_target_pose(self, pose):
        return super().set_target_pose(pose)

    def sample(self, timeout=-1.0, q0=None):
        assert self.pose is not None, 'set pose first'

        self.constraint.update_target(self.pose)
        trial = 0
        start_time = time.time()

        q0_ik = self.sample_from_q0(q0)
        
        if q0_ik is not None:
            return q0_ik
        
        while True:
            if timeout > 0:
                if time.time() - start_time > timeout:
                    return None

            if self.priority_queue.empty():
                t1 = time.time()
                nq, nz = self.constraint_model.sample_with_estimated_validity_with_q(self.model_sample_count, self.validity_model)
                t2 = time.time()
                for q, z in zip(nq, nz):
                    f = self.constraint.function_ik(q)
                    self.priority_queue.put((np.linalg.norm(f), next(self.unique), q, z))
                t3 = time.time()
                if self.debug:
                    print('sample', t2-t1, 'put', t3-t2)
            f,_, q,z = self.priority_queue.get()

            new_f = self.constraint.function_ik(q)
            
            if np.linalg.norm(new_f) > np.linalg.norm(f) + self.new_config_threshold:
                self.priority_queue.queue.clear()
                continue
            
            trial += 1
            
            r = self.constraint.solve_ik(q, self.pose)
            if r is False:
                continue

            if self.debug:
                print(f, q, r, trial)
                self.constraint.planning_scene.display(q=q)
            
            if (q < self.constraint.lb).any() or (q > self.constraint.ub).any():
                continue
            if self.constraint.planning_scene.is_valid(q) is False:
                if self.debug:
                    self.constraint.planning_scene.print_current_collision_infos()
                continue
            return q