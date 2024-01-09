import numpy as np
from anytree import Node

from ljcmp.models.latent_model import LatentModel
from srmt.constraints.constraints import ConstraintBase

from ljcmp.planning.distance_functions import distance_q, distance_z

class MotionTree():
    def __init__(self, name = '', multiple_roots = False, constraint : ConstraintBase = None) -> None:
        self.nodes = [None]
        self.name = name
        self.multiple_roots = multiple_roots
        self.grow_fail_count = 0
        self.max_dist = 0.32
        self.k = 3
        self.constraint = constraint
        if multiple_roots:
            self.nodes = [Node(name='root')]

    def __len__(self):
        if self.multiple_roots:
            return len(self.nodes) - 1
        
        return len(self.nodes)

    def set_root(self, init_x):
        self.nodes = [Node(name='root', x=init_x)]

    def add_root_node(self, x):
        return self.add_node(0, x)

    def get_root(self):
        return self.nodes[0]

    def add_node(self, parent_index, x):
        self.nodes.append(Node(name='n_{0}'.format(len(self.nodes)), x=x, parent=self.nodes[parent_index]))
        return len(self.nodes) - 1 # return index

    def get_nearest(self, x):
        """
        return: q, tree node index
        """
        if self.multiple_roots:
            dists = np.array([distance_q(x, node.x) for node in self.nodes[1:]])
            min_index = dists.argmin() + 1
        else: 
            dists = np.array([distance_q(x, node.x) for node in self.nodes])
            min_index = dists.argmin()

        return self.nodes[min_index].x, min_index

    def get_near_k(self, x, k):
        if self.multiple_roots:
            dists = np.array([(np.linalg.norm(x-node.x)) for node in self.nodes[1:]])    
            near_indices = np.argsort(dists)[:k] + 1
        else: 
            dists = np.array([(np.linalg.norm(x-node.x)) for node in self.nodes])
            near_indices = np.argsort(dists)[:k]

        return near_indices

    def get_near_radius(self, x, radius):
        if self.multiple_roots:
            dists = np.array([(np.linalg.norm(x-node.x)) for node in self.nodes[1:]])    
            near_indices = np.where(dists < radius)[0] + 1
        else: 
            dists = np.array([(np.linalg.norm(x-node.x)) for node in self.nodes])
            near_indices = np.where(dists < radius)[0]

        return near_indices

    def get_path(self, index):

        if self.multiple_roots:
            x_path = [node.x for node in self.nodes[index].path[1:]]
        else:
            x_path = [node.x for node in self.nodes[index].path]

        return x_path


class LatentMotionTree():
    def __init__(self, model : LatentModel, name = '', multiple_roots = False) -> None:
        self.nodes = []
        self.q_nodes = []
        self.require_q_to_z_nodes = []
        self.name = name
        self.multiple_roots = multiple_roots
        self.model = model
        self.grow_fail_count = 0

        if multiple_roots:
            self.nodes = [Node(name='root')]

    def __len__(self):
        if self.multiple_roots:
            return len(self.nodes) - 1
        
        return len(self.nodes)

    def set_root(self, init_z=None, init_q=None):
        assert(init_z is not None or init_q is not None)
        init_q = init_q.astype(np.double)
        self.nodes = [Node(name='root', z=init_z, q=init_q, checked=True, valid=True, projected=False if init_q is None else True, ref='z' if init_q is None else 'q')]
        if init_q is not None:
            self.q_nodes = [self.nodes[0]]
            
    def get_root(self):
        return self.nodes[0]

    def add_root_node(self, z, q):
        return self.add_node(parent_node=self.nodes[0], z=z, q=q, checked=True, valid=True, projected=True, ref='q')

    def add_node(self, parent_node, z=None, q=None, checked=False, valid=False, projected=False, ref='z'):
        """if q is not none, q should be projected.

        Args:
            parent_node (anytree.Node): parent node
            z (numpy.array, optional): latent space value. Defaults to None.
            q (numpy.array, optional): configuration space value. Defaults to None.
            checked (bool, optional): validity is checked or not. Defaults to False.
            valid (bool, optional): validity. Defaults to False.
            projected (bool, optional): projected or not. Defaults to False.
            ref (str, optional): the value reference 'z' or 'q'. Defaults to 'z'.

        Returns:
            anytree.Node: new node
        """
        assert(z is not None or q is not None)

        n = Node(name='n_{0}'.format(len(self.nodes)), 
                    z=z, q=q, checked=checked, 
                    valid=valid, 
                    projected=projected, 
                    ref=ref, 
                    parent=parent_node) 
        self.nodes.append(n)

        if checked and valid:
            self.q_nodes.append(n)
            
        return n 

    def delete_node(self, node):
        """
        only occurs for the z -> q case because q is already checked
        """

        "recursively delete all children"
        for c in node.children:
            self.delete_node(c)

        "remove from tree's nodes list"
        self.nodes.remove(node)

        "remove from parent's children list"
        if node.parent is not None:
            node.parent.children = tuple ([c for c in node.parent.children if c != node])

        node.checked = True
        node.valid = False

    def get_nearest(self, z):
        """
        return: z, tree node index
        """
        if self.multiple_roots:
            dists = np.array([distance_z(z,node.z) for node in self.nodes[1:]])    
            min_index = dists.argmin() + 1
        else: 
            dists = np.array([distance_z(z,node.z) for node in self.nodes])
            min_index = dists.argmin()

        return self.nodes[min_index].z, self.nodes[min_index]

    def get_nearest_q(self, q):
        """
        return: z, tree node index
        """
        dists = np.array([distance_q(q,node.q) for node in self.q_nodes])

        min_index = dists.argmin()

        return self.q_nodes[min_index].q, self.q_nodes[min_index]

    def get_path(self, target_node):
        nodes = target_node.path

        if self.multiple_roots:
            nodes = nodes[1:]

        z_path = [node.z for node in nodes]
        z_path = np.array(z_path)
        return z_path, nodes

    def get_path_q(self, target_node):
        nodes = target_node.path

        if self.multiple_roots:
            nodes = nodes[1:]

        q_path = [node.q for node in nodes]
        q_path = np.array(q_path)
        return q_path, nodes

    def get_path_ref(self, target_node):
        nodes = target_node.path

        if self.multiple_roots:
            nodes = nodes[1:]

        ref_path = [node.ref for node in nodes]
        ref_path = np.array(ref_path, dtype=str)
        return ref_path, nodes
