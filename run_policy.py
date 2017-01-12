"""Convenience script for running a policy on the real SUPERball."""

import copy
import numpy as np
import scipy.ndimage as sp_ndimage
import roslib; roslib.load_manifest('gps_agent_pkg')
import rospy
import time
import threading
import sys
import pickle
import os
import argparse

from scipy import io
from std_msgs.msg import String, Float32, UInt16, Float32MultiArray
from sensor_msgs.msg import Imu


FLANN_DIR = os.environ['FLANN_PATH']


SUPERBALL_BAR_ACCELERATIONS = 0
MOTOR_POSITIONS = 1
ACTION = 2


try:
    from superball_msg.msg import TimestampedFloat32
    Float32 = TimestampedFloat32
    USE_F32 = False
except ImportError:
    USE_F32 = True


SUPERBALL_SENSOR_DIMS = {
    MOTOR_POSITIONS: 12,
    SUPERBALL_BAR_ACCELERATIONS: 36,
    ACTION: 12,
}


SUPERBALL_IMU_TOPICS = ["/bbb2/0x71_imu_data",
                        "/bbb2/0x1_imu_data",
                        "/bbb4/0x71_imu_data",
                        "/bbb4/0x1_imu_data",
                        "/bbb6/0x71_imu_data",
                        "/bbb6/0x1_imu_data",
                        "/bbb8/0x71_imu_data",
                        "/bbb8/0x1_imu_data",
                        "/bbb10/0x71_imu_data",
                        "/bbb10/0x1_imu_data",
                        "/bbb12/0x71_imu_data",
                        "/bbb12/0x1_imu_data"]

HYPERPARAMS = {
    'dt': 0.1,
    'T': 50,
    'sensor_dims': SUPERBALL_SENSOR_DIMS,
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


def generate_noise(T, dU):
    """
    Generate a T x dU gaussian-distributed noise vector. This will
    approximately have mean 0 and variance 1, ignoring smoothing.

    Args:
        T: Number of time steps.
        dU: Dimensionality of actions.
    Hyperparams:
        smooth: Whether or not to perform smoothing of noise.
        var : If smooth=True, applies a Gaussian filter with this
            variance.
        renorm : If smooth=True, renormalizes data to have variance 1
            after smoothing.
    """
    smooth, var = True, 1
    renorm = True
    noise = np.random.randn(T, dU)
    if smooth:
        # Smooth noise. This violates the controller assumption, but
        # might produce smoother motions.
        for i in range(dU):
            noise[:, i] = sp_ndimage.filters.gaussian_filter(noise[:, i], var)
        if renorm:
            variance = np.var(noise, axis=0)
            noise = noise / np.sqrt(variance)
    return noise


class AgentSUPERball(object):
    """
    All communication between the algorithms and the SUPERball simulation is done through this class.
    """
    def __init__(self, use_acc_only):
        self._init_obs()
        config = copy.deepcopy(HYPERPARAMS)
        self._hyperparams = config

        self._use_acc_only = use_acc_only

        if self._hyperparams['constraint']:
            import superball_kinematic_tool as skt
            self._constraint = skt.KinematicMotorConstraints(
                self._hyperparams['constraint_file'], **self._hyperparams['constraint_params']
            )
        rospy.init_node('superball_agent', disable_signals=True, log_level=rospy.DEBUG)

        self._obs_update_lock = threading.Lock()
        self._motor_pos_sub = rospy.Subscriber(
            '/ranging_data_matlab', Float32MultiArray,
            callback=self._motor_pos_cb, queue_size=1
        )
        self._imu_sub = []
        for i in range(12):
            self._imu_sub.append(
                rospy.Subscriber(
                    SUPERBALL_IMU_TOPICS[i], Imu,
                    callback=self._imu_cb, queue_size=1
                )
            )

        self._ctrl_pub = rospy.Publisher('/superball/control', String, queue_size=1)
        self._timestep_pub = rospy.Publisher('/superball/timestep', UInt16, queue_size=1)
        self._init_motor_pubs()
        self._run_sim = False
        self._sim_thread = threading.Thread(target=self._continue_simulation)
        self._sim_thread.daemon = True
        self._sim_thread.start()
        self._action_rate = rospy.Rate(10)

    def _continue_simulation(self):
        rate = rospy.Rate(10)
        while True:
            self._timestep_pub.publish(UInt16(100))
            rate.sleep()


    def _motor_pos_cb(self, msg):
        with self._obs_update_lock:
            motor_pos = np.array(msg.data[-12:])
            indices = np.logical_not(np.isnan(motor_pos))
            self._sensor_readings[MOTOR_POSITIONS][indices] = 0.95 - np.abs(0.009 * np.maximum(0, motor_pos[indices] - 7.5))

    def _imu_cb(self, msg):
        with self._obs_update_lock:
            if not np.isnan(msg.linear_acceleration.y):
                i = [int(n) for n in msg.header.frame_id.split() if n.isdigit()]
                index = i[0] - 1
                if index % 2 == 0:
                    index += 1
                else:
                    index -=1
                self._sensor_readings[SUPERBALL_BAR_ACCELERATIONS][index] = msg.linear_acceleration.y

    def _init_motor_pubs(self):
        self._motor_pubs = []
        for i in range(12):
            if i % 2 == 0:
                bbb, board_id, sub_index = i + 2, 0x71, 0x2
            else:
                bbb, board_id, sub_index = i + 1, 0x1, 0x1
            self._motor_pubs.append(
                    rospy.Publisher('/bbb%d/0x%x_0x2040_0x%x' % (bbb, board_id, sub_index), Float32,
                        queue_size=1))

    def _init_obs(self):
        self._sensor_readings = {SUPERBALL_BAR_ACCELERATIONS: np.zeros(12), MOTOR_POSITIONS: np.zeros(12)}

    @property
    def obs(self):
        if self._use_acc_only:
            return self._sensor_readings[SUPERBALL_BAR_ACCELERATIONS]
        else:
            return np.concatenate([
                self._sensor_readings[SUPERBALL_BAR_ACCELERATIONS],
                self._sensor_readings[MOTOR_POSITIONS]
            ])

    def sample(self, policy, horizon):
        self._init_obs()
        self.relax()

        noise = generate_noise(horizon, 12)
        self._run_sim = True
        actions = []
        for t in range(horizon):
            obs_t = self.obs
            U_t = policy.act(np.zeros_like(obs_t), obs_t, t, noise[t,:])
            actions.append(U_t)
            self._set_motor_positions(U_t)
            self.advance_simulation()

        self._run_sim = False
        actions = np.vstack(actions)
        return actions

    def _set_motor_positions(self, pos):
        gain = 1 / 0.009
        msg = Float32()
        if not USE_F32:
            msg.header.stamp = rospy.rostime.get_rostime()
        if (pos - 0.95).all() and self._hyperparams['constraint']:
            pos = self._constraint.find_nearest_valid_values(pos)
        for i in range(12):
            msg.data = min(max(7.5, ((0.95 - pos[i]) * gain) + (7.5 / gain)), 40)
            self._motor_pubs[i].publish(msg)

    def relax(self):
        self._set_motor_positions(np.ones(12) * 0.95)
        self.advance_simulation(30)

    def reset(self, bottom_face=0):
        rospy.set_param('/bottom_face', bottom_face + 1)
        self._ctrl_pub.publish(String('reset'))

    def advance_simulation(self, step=1):
        for _ in xrange(step):
            self._action_rate.sleep()

    def replay_actions(self, actions):
        for t in range(actions.shape[0]):
            obs_t = self.obs
            U_t = actions[t, :]
            self._set_motor_positions(U_t)
            self.advance_simulation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run caffe policy.')
    parser.add_argument('policy', type=str, help='pickled policy')
    parser.add_argument('-t', '--timesteps', metavar='N', type=int, default=1000,
                        help='number of time steps')
    parser.add_argument('-r', '--replay', action='store_true', help='replay actions')
    parser.add_argument('-s', '--store_actions', type=str, help='store policy actions')
    parser.add_argument('-a', '--use_acc_only', action='store_true', help='use acceleration observation')
    args = parser.parse_args()

    with open(args.policy) as fin:
        policy = pickle.load(fin)

    agent = AgentSUPERball(args.use_acc_only)
    agent.reset()
    agent.relax()
    if args.replay:
        agent.replay_actions(policy)
    else:
        policy = policy.policy_opt.policy
        actions = agent.sample(policy, args.timesteps)
        if args.store_actions is not None:
            with open(args.store_actions, 'w') as fout:
                pickle.dump(actions, fout)
