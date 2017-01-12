from __future__ import division

from datetime import datetime
import numpy as np
import os.path

os.environ['GLOG_minloglevel'] = '2'

from gps import __file__ as gps_filepath
from gps.agent.ros.agent_superball import AgentSUPERball
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY, RAMP_CONSTANT, RAMP_QUADRATIC
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_nn import DynamicsPriorNN
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_superball
from gps.proto.gps_pb2 import *


SENSOR_DIMS = {
    BAR_ENDPOINT_POSITIONS: 36,
    BAR_ENDPOINT_POSITIONS_ABS: 36,
    BAR_ENDPOINT_VELOCITIES: 36,
    MOTOR_POSITIONS: 12,
    MOTOR_VELOCITIES: 12,
    NODE_VECTORS: 6,
    NODE_VECTOR_VELOCITIES: 6,
    SUPERBALL_TRIS: 35,
    SUPERBALL_TRIS_VELOCITIES: 35,
    SUPERBALL_PTRIS: 11,
    SUPERBALL_PTRIS_VELOCITIES: 11,
    ACTION: 12,
}

def _get_path_elem(path, idx):
    if idx == 0:
        return os.path.split(path)[1]
    return _get_path_elem(os.path.split(path)[0], idx - 1)

EXP_NAME = _get_path_elem(os.path.abspath(__file__), 1)
BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
TYPE_DIR = _get_path_elem(os.path.abspath(__file__), 2)
EXP_DIR = BASE_DIR + '/../experiments/' + TYPE_DIR + '/' + EXP_NAME + '/'
FLANN_DIR = os.environ['FLANN_PATH']


common = {
    'experiment_name': EXP_NAME,
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 12,
    'iterations': 20,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentSUPERball,
    # 'state_estimator': True,
    'reset': [True, False] * 6,
    'bottom_faces': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'policy_test_horizon': 30 * 20,
    'dt': 0.1,
    'conditions': common['conditions'],
    'compute_vel': {
        BAR_ENDPOINT_VELOCITIES: BAR_ENDPOINT_POSITIONS_ABS,
        MOTOR_VELOCITIES: MOTOR_POSITIONS,
        NODE_VECTOR_VELOCITIES: NODE_VECTORS,
        SUPERBALL_TRIS_VELOCITIES: SUPERBALL_TRIS,
        SUPERBALL_PTRIS_VELOCITIES: SUPERBALL_PTRIS,
    },
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [BAR_ENDPOINT_POSITIONS, BAR_ENDPOINT_VELOCITIES, MOTOR_POSITIONS, MOTOR_VELOCITIES],
    'obs_include': [SUPERBALL_PTRIS, SUPERBALL_PTRIS_VELOCITIES, MOTOR_POSITIONS],
    'smooth_noise': True,
    'smooth_noise_renormalize': True,
    'smooth_noise_var': 2.0,
    'constraint': True,
    'constraint_file': FLANN_DIR + 'l0_filtered_2016_01_21_09_53_29.npy',
    'constraint_params': {
        'index_file': FLANN_DIR + 'index_filtered_2016_01_21_09_53_29_manhattan.flann',
        'distance_type': 'manhattan',
    },
    'ctrl_vel': False,
    'sensors_dropout_rate': 0.03,
    'sensors_noise_stdddev': 0.05,
}

algorithm = {
    'type': AlgorithmMDGPS,
    'conditions': common['conditions'],
    'iterations': common['iterations'],
    'lg_step_schedule': np.array([1e-4, 1e-3, 1e-2, 1e-2]),
    'policy_dual_rate': 0.2,
    'ent_reg_schedule': np.array([1e-3, 1e-3, 1e-2, 1e-1]),
    'fixed_lg_step': 3,
    # 'kl_step': 5.0,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    # 'init_pol_wt': 1e-3,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,
    'sample_on_policy': True,
}

algorithm['init_traj_distr'] = {
    'type': init_superball,
    # 'init_var': 25.0,
    'init_var': 1.0,
    'T': agent['T'],
    # 'init': [str(i) for i in xrange(common['conditions'])],
    'init': ['0', False, '1', False, '2', False, '3', False, '4', False, '5', False],
    # 'init_with_var': 0.1,
}

torque_cost = {
    'type': CostAction,
    'wu': np.ones(12) * 1e-2,
    'target': np.ones(12) * 0.95,
}

state_cost = {
    'type': CostState,
    'data_types': {
        BAR_ENDPOINT_VELOCITIES: {
            'average': (12, 3),
            'wp': [-1., 1., 0.],
            'target_state': [0., 0., 0.],
        },
    },
}

state_cost2 = {
    'type': CostState,
    'ramp_option': RAMP_QUADRATIC,
    'l1': 0.0,
    'l2': 1.0,
    'wp_final_multiplier': 1.0,
    'data_types': {
        MOTOR_POSITIONS: {
            'wp': np.ones(12),
            'target_state': np.ones(12) * 0.95,
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [state_cost, state_cost2, torque_cost],
    'weights': [25.0, 0.0, 0.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 40,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}
algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'weights_file_prefix': EXP_DIR + 'policy',
    'iterations': 5000,
    'network_arch_params': {
        'n_layers': 4,
        'dim_hidden': [128, 128, 128],
    },
    'use_gpu': True,
    'init_net': EXP_DIR + 'init_policy.caffemodel',
    'init_normalization': EXP_DIR + 'init_policy_constants.pkl',
}


algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 50,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}

config = {
    'iterations': common['iterations'],
    'num_samples': 15,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'exp_name': EXP_NAME,
    'verbose_trials': 0,
    'save_controller': True,
}

common['info'] = (
    'exp_name: '   + str(common['experiment_name'])              + '\n'
    'alg_type: '   + str(algorithm['type'].__name__)             + '\n'
    'alg_dyn:  '   + str(algorithm['dynamics']['type'].__name__) + '\n'
    'alg_cost: '   + str(algorithm['cost']['type'].__name__)     + '\n'
    'iterations: ' + str(config['iterations'])                   + '\n'
    'conditions: ' + str(algorithm['conditions'])                + '\n'
    'samples:    ' + str(config['num_samples'])                  + '\n'
)
