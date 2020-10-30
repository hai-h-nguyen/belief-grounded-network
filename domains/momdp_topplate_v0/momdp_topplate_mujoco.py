#
# Created by Xinchao Song on June 1, 2020.
#

import copy
import operator
from pathlib import Path
from collections import namedtuple
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (ERROR: you need to install mujoco_py, "
        "and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

# Theta discrete values
THETA_DOWN = 0
THETA_NEUTRAL = 1
THETA_UP = 2

# Action values
ACT_DOWN = 0
ACT_UP = 1
ACT_GRASP = 2

# State namedtuple
State = namedtuple('State', 'z_g, theta, z_t')


class MomdpTopPlateV0(gym.Env):
    """
    Description:

    The MuJoCo simulation environment of the plates environment.

    Observation:
         Type: Discrete
         Num    Observation         Values
         0      Gripah Position     0, 1, 2, 3, 4, 5, 6, 7
         1      Gripah Angle        0 (THETA_DOWN), 1 (THETA_NEUTRAL), 2 (THETA_UP)
         2      Action              0 (ACT_DOWN), 1 (ACT_UP), 2 (ACT_GRASP)

    Actions:
         Type: Discrete(3)
         Num    Action
         0      moving down
         1      Moving up
         2      Grasp

    Reward:
         Reward of 1 for successfully grasp the top plate

     Starting State:
         The starting state of the gripah is assigned to z_g = 0 and theta = THETA_NEUTRAL

     Episode Termination:
         When the gripah tries to do a grasp.
     """

    def __init__(self, rendering=False, seed=None):
        """
        The initialization of the simulation environment.
        """

        # mujoco - py
        xml_path = Path(__file__).resolve().parent / 'mujoco_model' / 'plates_stacked_model.xml'
        self.model = mujoco_py.load_model_from_path(str(xml_path))
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Initializes only when self.render() is called.
        self.rendering = rendering

        # MuJoCo
        self.num_plate_default = 20  # Must be consistent with plates_stacked_model.xml.
        self.plate_max_idx = 5  # Can be any number between [1, self.num_plate_default].
        self.slide_z_id = self.model.joint_name2id('slide:gripah-base-z')
        self.hinge_wide_finger_id = self.model.joint_name2id('hinge:wide-finger')
        self.hinge_narrow_finger_id = self.model.joint_name2id('hinge:narrow-finger')
        self.plates_body_ids = self._get_plates_body_ids()
        self.velocity_z_id = self.model.actuator_name2id('velocity:z')
        self.velocity_narrow_finger_id = self.model.actuator_name2id('velocity:narrow-finger')
        self.position_narrow_finger_id = self.model.actuator_name2id('position:narrow-finger')
        self.gripah_velocity = 10  # This is the maximum speed. Setting any higher speed needs recalibration the model.
        self.qpos_nfinger = 0

        # Z range
        self.z_down_limit = 0
        self.z_up_limit = self.plate_max_idx + 2
        self.num_z_slots = self.z_up_limit + 1
        self.size_scale = 10  # 1 unit size in the code means 10 unit in the MuJoCo

        # Theta
        self.FINGER_THETA_DOWN = THETA_DOWN
        self.FINGER_THETA_NEUTRAL = THETA_NEUTRAL
        self.FINGER_THETA_UP = THETA_UP
        self.max_theta = THETA_UP
        self.angle_str = {THETA_DOWN: 't_down', THETA_NEUTRAL: 't_neutral', THETA_UP: 't_up'}
        self.low_stiffness = 250

        # Action
        self.action_space = spaces.Discrete(3)
        self.norm_action = self.action_space.n - 1
        self.action_str = {ACT_DOWN: 'MoveDown', ACT_UP: 'MoveUP', ACT_GRASP: 'Grasp'}

        # States and Obs
        self.num_plate_selected = 0
        self.discount = 1.0
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.z_g_down_limit = self.z_down_limit
        self.z_g_up_limit = self.z_up_limit
        self.z_g = self.z_g_down_limit
        self.z_t = None
        self.theta = self.FINGER_THETA_NEUTRAL

        # Belief
        self.start_belief = self._generate_start_belief()
        self.belief = np.zeros((self.num_z_slots, 1))

        self.debug = False

        # numpy random
        self.np_random = None
        self.seed(seed)

    def reset(self):
        """
        Resets the current MuJoCo simulation environment with given initial x coordinates of the four plates.

        :return: the observations of the environment
        """

        # Resets the mujoco env
        self.sim.reset()

        # Determines the number of plates to use.
        self.num_plate_selected = self.np_random.randint(1, self.plate_max_idx + 1)

        # Setups the unselected plates on the table.
        for plate_z, body_id in self.plates_body_ids.items():
            if plate_z > self.num_plate_selected:
                self.model.body_pos[body_id][1] = -20
            else:
                self.model.body_pos[body_id][1] = 0

        # Assigns the start state of the narrow finger to mujoco-py
        self._control_narrow_finger(theta_target=0.9, teleport=True)

        # Resets all state parameters.
        self.z_g = self.z_g_down_limit
        self.theta = self.FINGER_THETA_NEUTRAL
        self.z_t = self.num_plate_selected

        # Resets belief.
        self.belief = copy.deepcopy(self.start_belief)

        if self.debug:
            print()
            print("Reset", self.z_g, self.theta, self.z_t)
            # print(self.belief)      

        return np.array([self.z_g / self.z_up_limit, self.theta / self.max_theta, -1.0 / self.norm_action])

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action:
        0 - move down
        1 - move up
        2 - grasp

        :return: observations, reward, done, info
        """

        reward = 0
        done = False

        # Executes the determined action.
        if action == 0:
            self._move_gripah_down()

        elif action == 1:
            self._move_gripah_up()

        elif action == 2:
            done = True

            if self._grasp():
                if self.debug:
                    print("Success")
                reward += 1
            else:
                if self.debug:
                    print("Fail")

        else:
            raise ValueError("Unknown action index received.")

        if not done:
            self._update_belief()

        obs = np.array(
            [self.z_g / self.z_up_limit, self.theta / self.max_theta, float(action / self.norm_action)])
        info = {'curr_state': self.get_state(), 'belief': self.get_belief()}
        info['reward_cat'] = reward

        if self.debug:
            print(self.action_str[action], self.z_g, self.angle_str[self.theta], self.z_t)
            # print(self.belief)
        return obs, reward, done, info

    def get_state(self):
        """
        Gets the current state.

        :return: the current state
        """

        state_normalized = State(z_g=self.z_g / self.z_up_limit,
                                 theta=self.theta / self.max_theta,
                                 z_t=self.z_t / self.plate_max_idx)

        return np.array(state_normalized)

    def get_belief(self):
        """
        Gets the current belief.

        :return: the current belief
        """

        return np.concatenate(
            (np.array([self.z_g / self.z_up_limit, self.theta / self.max_theta]), self.belief.flatten()), axis=0)

    def render(self, mode='human'):
        """
        Renders the environment.
        """

        if mode != 'human':
            raise NotImplementedError("Only the human mode is supported.")

        if self.rendering:
            if self.viewer is None:
                self.viewer = mujoco_py.MjViewer(self.sim)
                self.viewer.cam.distance = 150
                self.viewer.cam.azimuth = 225
                self.viewer.cam.elevation = -45

            self.viewer.render()

    def close(self):
        """
        Closes the simulation environment.
        """

        # mujoco-py will close the env automatically.
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :param seed: the seed for the random number generator(s)
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def _generate_start_belief(self):
        """
        Generates the default start belief.

        :return: the default start belief
        """

        belief = np.zeros((self.num_z_slots, 1))
        all_valid_z_t = list(range(self.z_g_down_limit + 1, self.z_up_limit))
        start_belief_value = 1 / len(all_valid_z_t)

        for z_t in all_valid_z_t:
            belief[z_t] = start_belief_value

        return belief

    def _update_belief(self):
        """
        Updates the belief.
        """

        old_belief = np.array(self.belief)

        for z_t in range(self.num_z_slots):
            state = State(z_g=self.z_g, theta=self.theta, z_t=z_t)

            if self._is_valid_state(state):
                self.belief[z_t] = old_belief[z_t]
            else:
                self.belief[z_t] = 0

        assert (np.sum(self.belief) != 0.0)
        self.belief = self.belief / np.sum(self.belief)

    def _is_valid_state(self, state):
        """
        Determines if the given state is valid for calculating the belief.

        :param state: State(z_g, theta, z_t)
        :return: True if the given state is valid, False otherwise
        """

        if state.z_g > state.z_t or state.z_g == 0:
            return state.theta == self.FINGER_THETA_NEUTRAL
        else:
            return state.theta != self.FINGER_THETA_NEUTRAL

    def _move_gripah_down(self):
        """
        Moves the gripah down with soft fingers.
        """

        if self.z_g <= self.z_g_down_limit:
            return

        self.z_g -= 1
        self._move_gripah_along_z(z_g=self.z_g)

    def _move_gripah_up(self):
        """
        Moves the gripah up with soft fingers.
        """

        if self.z_g >= self.z_g_up_limit:
            return

        self.z_g += 1
        self._move_gripah_along_z(z_g=self.z_g)

    def _grasp(self):
        """
        Tries to do a grasp.

        :return: True if the grasp succeed, False otherwise
        """

        if self.z_g != self.z_t:
            return False

        self.theta = self._get_theta()

        return True

    def _move_gripah_along_z(self, z_g):
        """
        Moves the gripah with the given z_g and stiffness.

        :param z_g:       the target z_g for the gripah
        """

        if z_g == 0:
            velocity = self.gripah_velocity * 0.5
        else:
            velocity = self.gripah_velocity        

        self._set_wide_finger_stiffness(self.low_stiffness)
        self._control_slider_z(z_target=z_g * self.size_scale, velocity=velocity, teleport=False)
        self.theta = self._get_theta()

    def _control_slider_z(self, z_target, velocity, teleport=False):
        """
        Controls the joint z of the gripah to move to the given target state.

        :param z_target: the target state that joint z of the gripah should move to.
        :param velocity: the velocity that the gripah moves with
        :param teleport: teleport mode. The gripah will be teleported to the desired state without running
                         simulation. Note when running the actuator in teleport mode, the gripah is not able
                         to interact with other objects
        """

        if teleport:
            self.sim.data.qpos[self.slide_z_id] = z_target
            self.sim.data.ctrl[self.velocity_z_id] = 0
            self.sim.step()

            return

        if self._get_raw_z_g() < z_target:
            while self._get_raw_z_g() <= z_target:
                self.sim.data.ctrl[self.velocity_z_id] = velocity
                self.sim.step()
                self.render()

        elif self._get_raw_z_g() > z_target:
            while self._get_raw_z_g() >= z_target:
                self.sim.data.ctrl[self.velocity_z_id] = -velocity
                self.sim.step()
                self.render()

    def _control_narrow_finger(self, theta_target, teleport=False):
        """
        Controls the narrow finger to rotate to the given target state.

        :param theta_target: the target state that the narrow finger should rotate to.
        :param teleport:     teleport mode. The gripah will be teleported to the desired state without running
                             simulation. Note when running the actuator in teleport mode, the gripah is not able
                             to interact with other objects
        """

        self.qpos_nfinger = -theta_target

        if teleport:
            self.sim.data.qpos[self.hinge_narrow_finger_id] = self.qpos_nfinger
            self.sim.data.ctrl[self.position_narrow_finger_id] = self.qpos_nfinger
            self.sim.step()

            return

        self.sim.data.ctrl[self.position_narrow_finger_id] = self.qpos_nfinger
        while True:
            last_state = self._get_gripah_raw_state()
            self.sim.step()
            self.render()
            now_state = self._get_gripah_raw_state()

            for diff in map(operator.sub, last_state, now_state):
                if abs(round(diff, 3)) > 0.001:
                    break
            else:
                break

    def _get_theta(self):
        """
        Gets the current angle of the angle of the wide finger.

        :return: the current angle of the angle of the wide finger
        """

        theta = self._get_wide_finger_angle()

        if theta < -0.008:
            theta_discretized = self.FINGER_THETA_DOWN
        elif theta > 0.008:
            theta_discretized = self.FINGER_THETA_UP
        else:
            theta_discretized = self.FINGER_THETA_NEUTRAL

        return theta_discretized

    def _get_wide_finger_angle(self):
        """
        Gets the current angle of the wide finger. Since the raw value is
        negative but a positive number is expected in this environment, the
        additive inverse of the result from the MuJoCo will be returned.

        :return: the current angle of the wide finger
        """

        return -self.sim.data.qpos[self.hinge_wide_finger_id]

    def _get_wide_finger_stiffness(self):
        """
        Gets the current stiffness of the wide finger.

        :return: the current stiffness of the wide finger
        """

        return self.model.jnt_stiffness[self.hinge_wide_finger_id]

    def _set_wide_finger_stiffness(self, stiffness):
        """
        Sets the current stiffness of the wide finger.

        :param stiffness the stiffness to be set
        """

        self.model.jnt_stiffness[self.hinge_wide_finger_id] = stiffness

    def _get_narrow_finger_angle(self):
        """
        Gets the current angle of the narrow finger. Since the raw value is
        negative but a positive number is expected in this environment, the
        additive inverse of the result from the MuJoCo will be returned.

        :return: the current angle of the narrow finger
        """

        return -self.sim.data.qpos[self.hinge_narrow_finger_id]

    def _get_narrow_finger_stiffness(self):
        """
        Gets the current stiffness of the narrow finger.

        :return: the current stiffness of the narrow finger
        """

        return self.model.model.jnt_stiffness[self.hinge_narrow_finger_id]

    def _set_narrow_finger_stiffness(self, stiffness):
        """
        Sets the current stiffness of the narrow finger.

        :param stiffness the stiffness to be set
        """

        self.model.jnt_stiffness[self.hinge_narrow_finger_id] = stiffness

    def _get_raw_z_g(self):
        """
        Gets the raw value of z_g in MuJoCo.

        :return: the raw value of z_g
        """

        return self.sim.data.qpos[self.slide_z_id]

    def _get_gripah_raw_state(self):
        """
        Gets the current state of the gripah (x, y, z, and the angle of the narrow finger).

        :return: the current raw state of the gripah
        """

        x = self.sim.data.sensordata[0]
        y = self.sim.data.sensordata[1]
        z = self.sim.data.sensordata[2]
        w = self._get_narrow_finger_angle()

        return x, y, z, w

    def _get_plates_body_ids(self):
        """
        Gets the body indices of plates from MuJoCo.

        :return: a list containing the body indices of plates
        """
        plates_body_ids = {}

        for plate_z in range(1, self.num_plate_default + 1):
            plates_body_ids[plate_z] = self.model.body_name2id("plate%d" % plate_z)

        return plates_body_ids
