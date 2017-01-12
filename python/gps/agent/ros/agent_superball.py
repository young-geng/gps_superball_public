import copy
import numpy as np
import roslib; roslib.load_manifest('gps_agent_pkg')
import rospy
import time
import threading
import sys
import pickle
import os

from scipy import io
from std_msgs.msg import String, Float32, UInt16

from gps_agent_pkg.msg import SUPERballState, SUPERballStateArray
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise
from gps.agent.config import agent_superball
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
from gps.proto.gps_pb2 import *
from gps.sample.sample import Sample
from gps.utility import fix_logging as logging


LOGGER = logging.getLogger(__name__)


try:
    from superball_msg.msg import TimestampedFloat32
    Float32 = TimestampedFloat32
    USE_F32 = False
except ImportError:
    LOGGER.debug('No TimestampedFloat32 found, using Float32 instead.')
    USE_F32 = True


SUPERBALL_DEFAULT_SAMPLE_PARAMETERS = {
    'reset': True,
    'relax': True,
    'bottom_face': None,
    'horizon': None,
    'start_motor_positions': None,
    'motor_position_control_gain': None,
    'debug': False,
}


SUPERBALL_COMPUTE_VELOCITY = {
    BAR_ENDPOINT_VELOCITIES: BAR_ENDPOINT_POSITIONS_ABS,
    MOTOR_VELOCITIES: MOTOR_POSITIONS,
    NODE_VECTOR_VELOCITIES: NODE_VECTORS,
    SUPERBALL_TRIS_VELOCITIES: SUPERBALL_TRIS,
    SUPERBALL_PTRIS_VELOCITIES: SUPERBALL_PTRIS,
}


SUPERBALL_COMPUTE_ACCELERATION =  {
    BAR_ENDPOINT_ACCELERATIONS: BAR_ENDPOINT_POSITIONS_ABS,
}


SUPERBALL_SENSOR_DIMS = {
    BAR_ENDPOINT_POSITIONS: 36,
    BAR_ENDPOINT_POSITIONS_ABS: 36,
    BAR_ENDPOINT_VELOCITIES: 36,
    PSEUDO_BAR_ENDPOINT_VELOCITIES: 36,
    MOTOR_POSITIONS: 12,
    MOTOR_VELOCITIES: 12,
    NODE_VECTORS: 6,
    NODE_VECTOR_VELOCITIES: 6,
    SUPERBALL_TRIS: 35,
    SUPERBALL_TRIS_VELOCITIES: 35,
    SUPERBALL_PTRIS: 11,
    SUPERBALL_PTRIS_VELOCITIES: 11,
    SUPERBALL_BAR_ACCELERATIONS: 12,
    BAR_ENDPOINT_ACCELERATIONS: 36,
    ACTION: 12,
}


class AgentSUPERball(Agent):
    """
    All communication between the algorithms and the SUPERball simulation is done through this class.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(agent_superball)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._sensor_types = set(self.x_data_types + self.obs_data_types)
        self.x0 = None  # FIXME: not sure if x0 is actually used somewhere.
        self._sensor_readings = {}
        self._prev_sensor_readings = {}
        self._prev2_sensor_readings = {}

        if self._hyperparams['constraint']:
            import superball_kinematic_tool as skt
            self._constraint = skt.KinematicMotorConstraints(
                self._hyperparams['constraint_file'], **self._hyperparams['constraint_params']
            )
        rospy.init_node('superball_agent', disable_signals=True, log_level=rospy.DEBUG)
        self._state_update = False
        self._state_update_cv = threading.Condition()

        if 'state_estimator' in self._hyperparams and self._hyperparams['state_estimator']:
            self._state_sub = rospy.Subscriber(
                '/superball/state', SUPERballStateArray,
                callback=self._handle_state_msg, queue_size=1
            )
        else:
            self._state_sub = rospy.Subscriber(
                '/superball/state_sim', SUPERballStateArray,
                callback=self._handle_state_msg, queue_size=1
            )
        self._ctrl_pub = rospy.Publisher('/superball/control', String, queue_size=1)
        self._timestep_pub = rospy.Publisher('/superball/timestep', UInt16, queue_size=1)
        self._init_motor_pubs()
        self._compute_rel_pos = True
        self.reset(0)

    def _handle_state_msg(self, msg):
        self._state_update_cv.acquire()

        # Dropout sensor observations
        if ('sensors_dropout_rate' in self._hyperparams
                and self._hyperparams['sensors_dropout_rate'] > 0
                and np.random.rand() < self._hyperparams['sensors_dropout_rate']):
            self._state_update = True
            self._state_update_cv.notify()
            self._state_update_cv.release()
            return

        # Update records for previous sensors
        self._prev2_sensor_readings = copy.deepcopy(self._prev_sensor_readings)
        self._prev_sensor_readings = copy.deepcopy(self._sensor_readings)
        self._sensor_readings = {}
        # Absolute bar endpoint positions.

        bar_endpt_pos = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos.extend([[pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z]])
        self._sensor_readings[BAR_ENDPOINT_POSITIONS_ABS] = np.array(bar_endpt_pos).flatten()

        # Relative bar endpoint positions.
        bar_endpt_pos = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos.extend([[pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z]])
        bar_endpt_pos = np.array(bar_endpt_pos)
        bar_endpt_pos = bar_endpt_pos - np.mean(bar_endpt_pos, axis=0, keepdims=True)
        self._sensor_readings[BAR_ENDPOINT_POSITIONS] = bar_endpt_pos.flatten()

        # Motor positions.
        pos = [[msg.states[i].motor_pos1.data, msg.states[i].motor_pos2.data] for i in range(6)]
        self._sensor_readings[MOTOR_POSITIONS] = np.array(pos).flatten()

        # Node vectors
        bar_endpt_pos = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos.extend([[pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z]])
        node_vec = np.array(bar_endpt_pos)
        node_vec[1:12:2] = node_vec[1:12:2] - node_vec[0:11:2]
        norm = np.linalg.norm(node_vec[1:12:2], axis=1)
        node_vec[1:12:2] = node_vec[1:12:2]/np.array([norm, norm, norm]).T
        self._sensor_readings[NODE_VECTORS] = np.arccos(node_vec[1:12:2,2])

        # Superball translation and rotation invariant states
        bar_endpt_pos = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos += [[pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z]]
        bar_endpt_pos = np.array(bar_endpt_pos)
        # Compute relative position for translational invariance
        bar_endpt_pos = bar_endpt_pos - np.mean(bar_endpt_pos, axis=0, keepdims=True)

        horizontal_angle = np.arctan2(bar_endpt_pos[:, 1], bar_endpt_pos[:, 0])

        # Correct angle for rotational invariance around Z axis
        horizontal_angle = horizontal_angle - horizontal_angle[0]

        vertical_angle = np.arctan(
            bar_endpt_pos[:, 2] / np.linalg.norm(bar_endpt_pos[:, :2], axis=1)
        )

        distance = np.linalg.norm(bar_endpt_pos, axis=1)

        self._sensor_readings[SUPERBALL_TRIS] = np.hstack(
            [horizontal_angle[1:], vertical_angle, distance]
        ).flatten()

        # Superball proprioceptive translation and rotation invariant states
        bar_endpt_pos_1 = []
        bar_endpt_pos_2 = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos_1 += [[pos1.x, pos1.y, pos1.z]]
            bar_endpt_pos_2 += [[pos2.x, pos2.y, pos2.z]]
        bar_endpt_pos_1 = np.array(bar_endpt_pos_1)
        bar_endpt_pos_2 = np.array(bar_endpt_pos_2)
        bar_vector = bar_endpt_pos_2 - bar_endpt_pos_1

        horizontal_angle = np.arctan2(bar_vector[:, 1], bar_vector[:, 0])
        horizontal_angle = horizontal_angle - horizontal_angle[0]

        vertical_angle = np.arctan(
            bar_vector[:, 2] / np.linalg.norm(bar_vector[:, :2], axis=1)
        )
        self._sensor_readings[SUPERBALL_PTRIS] = np.hstack(
            [horizontal_angle[1:], vertical_angle]
        ).flatten()


        # Compute velocity
        for sensor in SUPERBALL_COMPUTE_VELOCITY:
            pos_sensor = SUPERBALL_COMPUTE_VELOCITY[sensor]
            if pos_sensor not in self._prev_sensor_readings:
                # First time step
                self._sensor_readings[sensor] = np.zeros(SUPERBALL_SENSOR_DIMS[sensor])
            else:
                vels = (self._sensor_readings[pos_sensor] - self._prev_sensor_readings[pos_sensor]) / self._hyperparams['dt']
                self._sensor_readings[sensor] = vels

        # Compute acceleration
        for sensor in SUPERBALL_COMPUTE_ACCELERATION:
            pos_sensor = SUPERBALL_COMPUTE_ACCELERATION[sensor]
            if pos_sensor not in self._prev2_sensor_readings:
                self._sensor_readings[sensor] = np.zeros(SUPERBALL_SENSOR_DIMS[sensor])
            else:
                v1 = (self._prev_sensor_readings[pos_sensor] - self._prev2_sensor_readings[pos_sensor]) / self._hyperparams['dt']
                v2 = (self._sensor_readings[pos_sensor] - self._prev_sensor_readings[pos_sensor]) / self._hyperparams['dt']
                accel = (v1 - v2) / self._hyperparams['dt']
                self._sensor_readings[sensor] = accel


        # Pseudo bar end point velocity
        if BAR_ENDPOINT_POSITIONS not in self._prev_sensor_readings:
            self._sensor_readings[PSEUDO_BAR_ENDPOINT_VELOCITIES] = np.zeros(SUPERBALL_SENSOR_DIMS[BAR_ENDPOINT_POSITIONS])
        else:
            self._sensor_readings[PSEUDO_BAR_ENDPOINT_VELOCITIES] = (self._sensor_readings[BAR_ENDPOINT_POSITIONS_ABS] - self._prev_sensor_readings[BAR_ENDPOINT_POSITIONS]) / self._hyperparams['dt']

        # Bar acceleration
        bar_endpt_pos = []
        for i in range(6):
            pos1, pos2 = msg.states[i].pos1, msg.states[i].pos2
            bar_endpt_pos.extend([[pos1.x, pos1.y, pos1.z], [pos2.x, pos2.y, pos2.z]])
        bar_endpt_pos = np.array(bar_endpt_pos)
        node_vec = np.zeros_like(bar_endpt_pos)
        node_vec[1:12:2] = bar_endpt_pos[1:12:2] - bar_endpt_pos[0:11:2]
        node_vec[0:11:2] = bar_endpt_pos[0:11:2] - bar_endpt_pos[1:12:2]
        norm = np.linalg.norm(node_vec, axis=1)
        node_vec = node_vec / np.array([norm, norm, norm]).T

        node_accel = self._sensor_readings[BAR_ENDPOINT_ACCELERATIONS].copy().reshape(12, 3)
        node_accel = np.sum(node_accel * node_vec, axis=1)
        projected_gravity = np.dot(node_vec, np.array([0, 0, -9.81]))
        # print "node accel: ", node_accel
        # print "gravity: ", projected_gravity
        bar_accel = node_accel + projected_gravity
        # bar_accel = projected_gravity
        self._sensor_readings[SUPERBALL_BAR_ACCELERATIONS] = bar_accel



        # Add noise to sensor
        if 'sensors_noise' in self._hyperparams:
            if type(self._hyperparams['sensors_noise']) != dict:
                for sensor_type in self._sensor_readings:
                    stddev = self._hyperparams['sensors_noise']
                    noise = np.random.randn(*self._sensor_readings[sensor_type].shape) * stddev
                    # truncate_limit = 100
                    # noise[noise < -truncate_limit * stddev] = -truncate_limit * stddev
                    # noise[noise > truncate_limit * stddev] = truncate_limit * stddev
                    self._sensor_readings[sensor_type] += noise
            else:
                for sensor_type in self._sensor_readings:
                    if sensor_type in self._hyperparams['sensors_noise']:
                        stddev = self._hyperparams['sensors_noise'][sensor_type]
                        noise = np.random.randn(*self._sensor_readings[sensor_type].shape) * stddev
                        # truncate_limit = 100
                        # noise[noise < -truncate_limit * stddev] = -truncate_limit * stddev
                        # noise[noise > truncate_limit * stddev] = truncate_limit * stddev
                        self._sensor_readings[sensor_type] += noise

        # self._sensor_readings[SUPERBALL_BAR_ACCELERATIONS][7] = 0
        # self._sensor_readings[MOTOR_POSITIONS][6] = 0.95



        self._state_update = True
        self._state_update_cv.notify()
        self._state_update_cv.release()

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

    def sample(self, policy, condition, verbose=True, save=True, noisy=False,
               screenshot_prefix=None, superball_parameters=None):
        if superball_parameters is None:
            superball_parameters = {}
        sample_params = copy.deepcopy(SUPERBALL_DEFAULT_SAMPLE_PARAMETERS)
        sample_params.update(superball_parameters)
        rospy.set_param('/verbose_trial', int(verbose))
        if screenshot_prefix:
            import pyscreenshot

        if sample_params['horizon'] is not None:
            # We don't save the sample if the horizon is customly defined
            horizon = sample_params['horizon']
            save = False
        else:
            horizon = self.T

        gain = sample_params['motor_position_control_gain']

        # Reset or relax
        if sample_params['reset']:
            self.reset(
                0, sample_params['bottom_face'],
                sample_params['start_motor_positions']
            )
        elif sample_params['relax']:
            self.relax()
        new_sample = self._init_sample(horizon)
        U = np.zeros([horizon, self.dU])
        # noise = generate_noise(self.T, self.dU, smooth=self._hyperparams['smooth_noise'],
        #         var=self._hyperparams['smooth_noise_var'],
        #         renorm=self._hyperparams['smooth_noise_renormalize'])
        noise = generate_noise(horizon, self.dU, self._hyperparams)  # FIXME: looks right?
        for t in range(horizon):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U_t = policy.act(X_t, obs_t, t, noise[t,:])

            if sample_params['debug'] and t >= horizon - 1:
                sys.stdout.write('[')
                for elem in X_t[X_t.shape[0] - 24:X_t.shape[0] - 12]:
                    sys.stdout.write('{}, '.format(elem))
                sys.stdout.write('],\n')

            # print obs_t


            U[t,:] = U_t
            if (t+1) < horizon:
                if self._hyperparams['ctrl_vel']:
                    self._set_motor_velocities(U_t)
                else:
                    self._set_motor_positions(U_t)
                self._advance_simulation()
                if screenshot_prefix:
                    img = screenshot_prefix + '_' + str(t).zfill(3) + '.png'
                    pyscreenshot.grab(bbox=(65, 50, 705, 530)).save(img)
                self._set_sample(new_sample, t)
        new_sample.set(ACTION, U)
        # with open('actions.pkl', 'w') as fout:
        #     pickle.dump(U, fout)
        # os._exit(0)
        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def _init_sample(self, T=None):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self, T)
        self._advance_simulation()
        for sensor in self._sensor_types:
            sample.set(sensor, self._sensor_readings[sensor], t=0)
        return sample

    def _set_motor_positions(self, pos):
        gain = 1 / 0.009
        msg = Float32()
        if not USE_F32:
            msg.header.stamp = rospy.rostime.get_rostime()
        if (pos - 0.95).all() and self._hyperparams['constraint']:
            pos = self._constraint.find_nearest_valid_values(pos)
        for i in range(12):
            # msg.data = (0.95 - pos[i]) / 0.009
            # if i == 6:
            #     continue
            msg.data = min(max(7.5, ((0.95 - pos[i]) * gain)), 40)
            self._motor_pubs[i].publish(msg)

    def _set_motor_velocities(self, vel):
        pos = self._sensor_readings[MOTOR_POSITIONS].copy()
        pos += vel * self._hyperparams['dt']
        msg = Float32()
        if not USE_F32:
            msg.header.stamp = rospy.rostime.get_rostime()
        if self._hyperparams['constraint']:
            pos = self._constraint.find_nearest_valid_values(pos)
        for i in range(12):
            msg.data = min(max(0, ((0.95 - pos[i]) / 0.009)), 45)
            self._motor_pubs[i].publish(msg)

    def _advance_simulation(self):
        self._timestep_pub.publish(UInt16(100))  # 100 ms, i.e., 10 Hz.

        self._state_update_cv.acquire()
        while not self._state_update:
            #time.sleep(0.001)
            self._state_update_cv.wait()
        self._state_update = False

        self._state_update_cv.release()

    def _set_sample(self, sample, t):
        for sensor in self._sensor_types:
            sample.set(sensor, self._sensor_readings[sensor], t=t+1)

    def reset(self, condition=0, bottom_face=None, start_motor_positions=None):
        if bottom_face is None:
            bottom_face = 0
        rospy.set_param('/bottom_face', bottom_face + 1)
        self._ctrl_pub.publish(String('reset'))
        if start_motor_positions is None:
            self._set_motor_positions(np.ones(12) * 0.95)
        else:
            self._set_motor_positions(np.array(start_motor_positions))
        time.sleep(0.5)
        for _ in range(30):
            self._advance_simulation()

    def relax(self):
        # print 'Relaxing'
        self._set_motor_positions(np.ones(12) * 0.95)
        time.sleep(0.5)
        for _ in range(30):
            self._advance_simulation()
        # print 'Relaxed!'

    def dist_traveled(self, sample, T):
        pos = sample.get(BAR_ENDPOINT_POSITIONS)
        dist = 0
        prev = pos[0,:].reshape((12, 3)).mean(axis=0)
        for t in range(1,T):
            curr = pos[t,:].reshape((12, 3)).mean(axis=0)
            dist += np.linalg.norm(curr-prev)
            prev = curr
        return dist
