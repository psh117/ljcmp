

"""Planner class for default parameters and functions."""
class Planner(object):
    def __init__(self, start_region_fn=None, goal_region_fn=None) -> None:
        
        self.solved_time = -1.0
        self.p_sample = 0.02

        self.start_region_fn = start_region_fn
        self.goal_region_fn = goal_region_fn
        
        # properties
        self.max_distance = 0.1

        self.debug = True

    def print(self, *args):
        if self.debug:
            print(*args)
    
    def solve(self, max_time=10.0):
        raise NotImplementedError