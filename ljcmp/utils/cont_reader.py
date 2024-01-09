import yaml
import numpy as np
import matplotlib.pyplot as plt
import scipy

class ContinuousGraspCandidates():
    def __init__(self, file_name='grasp.yaml'):
        self.file_name = file_name
        with open(file_name, 'r') as stream:
            self.yaml = yaml.safe_load(stream)

    def get_grasp(self, index, ratio):
        lb = np.array(self.yaml['grasp_points'][index]['lower_bound'])
        ub = np.array(self.yaml['grasp_points'][index]['upper_bound'])
        ori = np.array(self.yaml['grasp_points'][index]['orientation'])

        return ((ub-lb) * ratio + lb, ori)

    def get_transform(self, p, q):
        rot = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] = p

        return T
        
    def get_global_grasp(self, index, ratio, global_obj_pos, global_obj_quat):
        g_p, g_q = self.get_grasp(index,ratio)
        rot_g = scipy.spatial.transform.Rotation.from_quat(g_q).as_matrix()
        rot_c = scipy.spatial.transform.Rotation.from_quat(global_obj_quat).as_matrix()

        T_g = np.eye(4)
        T_g[:3,:3] = rot_g
        T_g[:3,3] = g_p

        T_c = np.eye(4)
        T_c[:3,:3] = rot_c
        T_c[:3,3] = global_obj_pos

        T_cg = np.dot(T_c, T_g)

        q_cg = scipy.spatial.transform.Rotation.from_matrix(T_cg[:3,:3]).as_quat()
        p_cg = T_cg[:3,3]
        
        return p_cg, q_cg
    
    def get_relative_transform(self, index_1, ratio_1, index_2, ratio_2):
        p_1, q_1 = self.get_grasp(index_1, ratio_1)
        p_2, q_2 = self.get_grasp(index_2, ratio_2)

        rot_1 = scipy.spatial.transform.Rotation.from_quat(q_1).as_matrix()
        rot_2 = scipy.spatial.transform.Rotation.from_quat(q_2).as_matrix()

        T_1 = np.eye(4)
        T_1[:3,:3] = rot_1
        T_1[:3,3] = p_1

        T_2 = np.eye(4)
        T_2[:3,:3] = rot_2
        T_2[:3,3] = p_2

        T_12 = np.dot(np.linalg.inv(T_1), T_2)

        q_12 = scipy.spatial.transform.Rotation.from_matrix(T_12[:3,:3]).as_quat()
        p_12 = T_12[:3,3]
        
        return p_12, q_12

    def display(self, display_indices=None):
        np.set_printoptions(precision=6, suppress=True, linewidth=200)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, gp in enumerate(self.yaml['grasp_points']):
            if display_indices is not None:
                if i not in display_indices:
                    continue
            
            lb = np.array(gp['lower_bound'])
            ub = np.array(gp['upper_bound'])
            ori = np.array(gp['orientation'])
            
            rot_m = scipy.spatial.transform.Rotation.from_quat(ori).as_matrix()
            
            z_axis_vec = rot_m[:, 2]
            y_axis_vec = rot_m[:, 1]
            
            # 20% 80% btw lb and ub
            lb_20 = (ub-lb) * 0.05 + lb
            ub_80 = (ub-lb) * 0.95 + lb

            z_len = -0.05
            y_len = 0.03

            lb_z_20 = lb_20 + z_axis_vec * z_len
            ub_z_80 = ub_80 + z_axis_vec * z_len

            lb_y_20 = lb_20 + y_axis_vec * y_len
            ub_y_80 = ub_80 + y_axis_vec * y_len

            lb_yz_20 = lb_y_20 + z_axis_vec * z_len
            ub_yz_80 = ub_y_80 + z_axis_vec * z_len

            ax.plot([lb_z_20[0], ub_z_80[0]], [lb_z_20[1], ub_z_80[1]], [lb_z_20[2], ub_z_80[2]], 'k')
            ax.plot(lb_z_20[0], lb_z_20[1], lb_z_20[2], 'ko')
            ax.plot(ub_z_80[0], ub_z_80[1], ub_z_80[2], 'ko')
            ax.plot([lb[0], ub[0]], [lb[1], ub[1]], [lb[2], ub[2]], 'r')
            ax.plot(lb[0], lb[1], lb[2], 'ro')
            ax.plot(ub[0], ub[1], ub[2], 'ro')
            ax.text(lb_yz_20[0], lb_yz_20[1], lb_yz_20[2], str(i), fontsize=20, fontdict={'weight': 'bold'})
            ax.text(ub_yz_80[0], ub_yz_80[1], ub_yz_80[2], str(i), fontsize=20, fontdict={'weight': 'bold'})
            ax.plot([lb_z_20[0], lb_20[0]], [lb_z_20[1], lb_20[1]], [lb_z_20[2], lb_20[2]], 'g')
            ax.plot([ub_z_80[0], ub_80[0]], [ub_z_80[1], ub_80[1]], [ub_z_80[2], ub_80[2]], 'g')

            ax.plot([lb_y_20[0], lb_20[0]], [lb_y_20[1], lb_20[1]], [lb_y_20[2], lb_20[2]], 'b')
            ax.plot([ub_y_80[0], ub_80[0]], [ub_y_80[1], ub_80[1]], [ub_y_80[2], ub_80[2]], 'b')

        plt.axis('equal')
        plt.show()