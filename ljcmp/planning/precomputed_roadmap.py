import numpy as np

import networkx as nx

from ljcmp.planning.status_description import GrowStatus

from ljcmp.planning.constrained_bi_rrt import ConstrainedBiRRT
from ljcmp.planning.prm import ConstrainedPRM

import tqdm

class PrecomputedRoadmap(ConstrainedPRM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def save_graph(self, file_name):
        nx.write_gpickle(self.graph, file_name)

    def load_graph(self, file_name):
        self.graph = nx.read_gpickle(file_name)
        
    def compute(self, q_set, max_num_edges=10):
        
        print('Computing roadmap')
        for q in q_set:

            # data vaidation
            fn = np.linalg.norm(self.constraint.function(q))
            if fn > 1e-2:
                raise ValueError('q is not valid {}'.format(fn))
                
            # adding node
            v = self.add_node(q)
        
        print('Connecting roadmap')
        tq = tqdm.tqdm(self.graph.nodes)
        for v in tq:
            tq.set_description(f'Node {v}')

            n_v_edges = self.graph.edges(v)
            tq2 = tqdm.tqdm(self.graph.nodes, leave=False)

            for u in tq2:
                tq2.set_description(f'Node {v} - {u} ({len(n_v_edges)}/{max_num_edges})')
                if len(n_v_edges) >= max_num_edges:
                    break
                
                # skip self
                if u == v:
                    continue

                q_v = self.graph.nodes[v]['q']
                q_u = self.graph.nodes[u]['q']

                grow_status, geodesic_states, step_dists, distance_traveled = self.check_motion(q_v, q_u)

                if grow_status == GrowStatus.REACHED:
                    self.graph.add_edge(v, u, weight=distance_traveled)
    

class PrecomputedGraph(ConstrainedBiRRT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = nx.Graph()
        
    def save_graph(self, file_name):
        nx.write_gpickle(self.graph, file_name)

    def load_graph(self, file_name):
        self.graph = nx.read_gpickle(file_name)
    
    def set_graph(self, graph):
        self.graph = graph
        self.node_cnt = len(graph.nodes)

    def add_node(self, q, weight=1.0, type='int'):
        node_cnt = len(self.graph.nodes)
        self.graph.add_node(node_cnt, q=q, weight=weight, type=type, total_connection_attempts=1, successful_connection_attempts=0)

        return node_cnt

    def from_configs(self, q_set):
        for q in q_set:
            self.add_node(q)

    def compute(self, q_set, max_num_edges=10):
        
        print('Computing roadmap')
        for q in q_set:

            # data vaidation
            fn = np.linalg.norm(self.constraint.function(q))
            if fn > 1e-2:
                raise ValueError('q is not valid {}'.format(fn))
            
            # adding node
            v = self.add_node(q)
        
        print('Connecting roadmap')
        tq = tqdm.tqdm(self.graph.nodes)
        for v in tq:
            tq.set_description(f'Node {v}')

            n_v_edges = self.graph.edges(v)
            tq2 = tqdm.tqdm(self.graph.nodes, leave=False)

            for u in tq2:
                tq2.set_description(f'Node {v} - {u} ({len(n_v_edges)}/{max_num_edges})')
                if len(n_v_edges) >= max_num_edges:
                    break
                
                # skip self
                if u == v:
                    continue

                q_v = self.graph.nodes[v]['q']
                q_u = self.graph.nodes[u]['q']

                grow_status, geodesic_states, step_dists, distance_traveled = self.check_motion(q_v, q_u)

                if grow_status == GrowStatus.REACHED:
                    self.graph.add_edge(v, u, weight=distance_traveled)
    
    def random_sample(self):
        len_graph = len(self.graph.nodes)
        idx = np.random.randint(0, len_graph)
        q = self.graph.nodes[idx]['q']
        
        return q