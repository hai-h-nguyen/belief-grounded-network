# Created by Xinchao Song on November 5, 2019.

import random
import time
from collections import namedtuple

import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np

from gym.envs.momdp_bumps_1d_v0.rdda_ur5_control_client import RddaUr5ControlClient

# Theta discrete values
THETA_LEFT = 0
THETA_NEUTRAL = 1
THETA_RIGHT = 2

# Action values
ACT_LEFT_SOFT = 0
ACT_LEFT_HARD = 1
ACT_RIGHT_SOFT = 2
ACT_RIGHT_HARD = 3

# State namedtuple
State = namedtuple('State', 'y_g, theta, y_bump1, y_bump2')


class MomdpBumps2dRobotEnv(gym.Env):
    """
    Description:

    The robot environment of a general two bump environment. Based on the PlateScriptSimulator.

    Observation:
         Type: Discrete
         Num    Observation         Values
         0      Gripah Position X   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
         1      Gripah Angle        0 (THETA_DOWN), 1 (THETA_NEUTRAL), 2 (THETA_UP)
         2      Action              0 (ACT_LEFT_SOFT), 1 (ACT_LEFT_HARD), 2 (ACT_RIGHT_SOFT), 3 (ACT_RIGHT_HARD)

    Actions:
         Type: Discrete(3)
         Num    Action
         0      Moving left with soft finger
         1      Moving left with hard finger
         2      Moving right with soft finger
         3      Moving right with hard finger

    Reward:
         Reward of 1 for successfully push the right bump to the right

     Starting State:
         The starting state of the gripah is assigned to y_g = random and theta = THETA_NEUTRAL

     Episode Termination:
         when the gripah tried to push a bump, or
         when the gripah moves out the given range [0, 10].
     """

    metadata = {
        'render.modes': ['human', ]
    }

    def __init__(self, seed=None):
        """
        The initialization of the class.
        """

        self.robot_controller = RddaUr5ControlClient()

        # UR5e
        self.robot_start_pos_x = 0.6
        self.robot_start_pos_y = 0
        self.robot_start_pos_z = 0.2
        self.robot_velocity_high = 0.1
        self.robot_velocity_low = 0.02

        # RDDA
        self.fixed_stiffness = 1
        self.low_stiffness = 0.2
        self.high_stiffness = 4
        self.theta_baseline = 0

        # X range
        self.y_origin = -0.32
        self.y_left_limit = 0
        self.y_right_limit = 14
        self.y_hardware_right_limit = 10
        self.num_y_slots = self.y_right_limit + 1
        self.size_scale = 0.08  # 1 unit size in the code means 0.08 meter in reality

        # Bumps
        self.y_bump1 = None
        self.y_bump2 = None
        self.min_bump_distance = 2
        self.max_bump_distance = int(self.y_right_limit / 2)
        self.y_bump1_limit_min = 2
        self.y_bump2_limit_min = self.y_bump1_limit_min + self.min_bump_distance
        self.y_bump2_limit_max = self.y_right_limit - 2
        self.y_bump1_limit_max = self.y_bump2_limit_max - self.min_bump_distance

        # Theta
        self.FINGER_POINT_RIGHT = THETA_RIGHT
        self.FINGER_POINT_DOWN = THETA_NEUTRAL
        self.FINGER_POINT_LEFT = THETA_LEFT
        self.max_theta = THETA_RIGHT
        self.angle_str = {THETA_LEFT: 't_left', THETA_NEUTRAL: 't_neutral', THETA_RIGHT: 't_right'}

        # Action
        self.action_space = spaces.Discrete(4)
        self.norm_action = self.action_space.n - 1
        self.action_str = {ACT_LEFT_SOFT: 'Left&Soft', ACT_LEFT_HARD: 'Left&Hard', ACT_RIGHT_SOFT: 'Right&Soft',
                           ACT_RIGHT_HARD: 'Right&Hard'}

        # States and Obs
        self.discount = 1.0
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.y_g_left_limit = self.y_left_limit
        self.y_g_right_limit = self.y_right_limit
        self.y_g = None
        self.theta = self.FINGER_POINT_DOWN
        self.start_state = None

        # Belief
        self.belief = np.zeros((self.num_y_slots, self.num_y_slots))

        # numpy random
        self.np_random = None
        self.seed(seed)
        self.action = None

    def reset(self):
        """
        Resets the robot environment with given initial y coordinates of the two bumps.

        :return: the observations of the environment
        """

        # Moves the robot back to the home pose.
        self.robot_controller.init_rdda_stiffness()
        self.robot_controller.home_ur5()
        self.robot_controller.set_rdda_stiffness(self.fixed_stiffness, self.fixed_stiffness)

        # Determines the start state of y_g, theta, y_bump1, and y_bump2.
        while True:
            is_random_str = input("Place the gripah and bumps randomly? [y/n] ")

            if is_random_str == 'y':
                is_random = True
                break
            elif is_random_str == 'n':
                is_random = False
                break

        if is_random:
            self.y_bump1 = self.np_random.randint(self.y_bump1_limit_min, self.y_bump1_limit_max + 1)

            while True:
                self.y_bump2 = self.np_random.randint(self.y_bump2_limit_min, self.y_bump2_limit_max + 1)
                if self.min_bump_distance <= self.y_bump2 - self.y_bump1 <= self.max_bump_distance:
                    break

            while True:
                self.y_g = self.np_random.randint(self.y_g_left_limit, self.y_g_right_limit + 1)
                if self.y_g != self.y_bump1 and self.y_g != self.y_bump2:
                    break

        else:
            self.y_g = int(input("Enter the start position of the gripah: "))
            self.y_bump1 = int(input("Enter the start position of bump #1: "))
            self.y_bump2 = int(input("Enter the start position of bump #2: "))

        # self.y_g = 4
        # self.y_bump1 = 3
        # self.y_bump2 = 6

        self.start_state = State(y_g=self.y_g, theta=self.theta, y_bump1=self.y_bump1, y_bump2=self.y_bump2)

        print("The gripper will be automatically placed at %d" % self.y_g)
        print("The bump #1 should be manually placed at %d" % self.y_bump1)
        print("The bump #2 should be manually placed at %d" % self.y_bump2)

        # Waits for the human operator to manually reset the bumps.
        input("After placing the bumps correctly, press Enter to continue...")

        # Resets the RDDA gripper.
        origins = self.robot_controller.read_rdda_origins()
        lower_bounds = self.robot_controller.read_rdda_lower_bounds()
        self.robot_controller.set_rdda_positions(origins[0], lower_bounds[1] + 0.1)

        # Moves to the start position.
        self.robot_start_pos_y = self.y_origin + self.y_g * self.size_scale
        time.sleep(0.3)
        self.robot_controller.move_ur5(self.robot_start_pos_x, self.robot_start_pos_y, self.robot_start_pos_z,
                                       self.robot_velocity_high)
        time.sleep(0.3)
        self.theta_baseline = self.robot_controller.read_rdda_positions()[0]

        return np.array((self.y_g / self.y_g_right_limit, self.theta / self.max_theta, -1.0 / self.norm_action))

    def step(self, action):
        """
        Steps the robot with the given action and returns the observations.

        :param action:
        0 - move left + finger soft
        1 - move left + finger hard
        2 - move right + finger soft
        3 - move right + finger hard

        :return: the observations of the environment
        """

        pre_y_g = self.y_g
        reward = 0
        done = False
        is_bump1_pushed = False
        is_bump2_pushed = False

        # Executes the determined action.
        if action == 0:
            self._move_gripah_left_soft()

        elif action == 1:
            self._move_gripah_left_hard()

        elif action == 2:
            self._move_gripah_right_soft()

        elif action == 3:
            self._move_gripah_right_hard()

        else:
            raise ValueError("Unknown action index received.")

        if action == 1 or action == 3:
            if pre_y_g == self.y_bump1 or self.y_g == self.y_bump1:
                is_bump1_pushed = self.is_bump_pushed()

            if pre_y_g == self.y_bump2 or self.y_g == self.y_bump2:
                is_bump2_pushed = self.is_bump_pushed()

        # Episode Termination:
        #   when the gripah tried to push a bump, or
        #   when the gripah moves out the given range [0, 10].
        if is_bump1_pushed or is_bump2_pushed:
            done = True

        if is_bump2_pushed:
            reward = 1

        obs = np.array((self.y_g / self.y_g_right_limit, self.theta / self.max_theta, float(action / self.norm_action)))
        info = {'curr_state': self.get_state(), 'belief': self.get_belief()}

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Renders the environment. Does thing for this environment.
        """

        pass

    def close(self):
        """
        Resets the robot back to the home position with the initial stiffness.

        :return: None
        """
        self.robot_controller.init_rdda_stiffness()
        self.robot_controller.home_ur5()

    def get_state(self):
        """
        Gets the current state.

        :return: the current state
        """

        state_normalized = State(y_g=self.y_g / self.y_g_right_limit,
                                 theta=self.theta / self.max_theta,
                                 y_bump1=self.y_bump1 / self.y_right_limit,
                                 y_bump2=self.y_bump2 / self.y_right_limit)

        return np.array(state_normalized)

    def get_belief(self):
        """
        Gets the current belief.

        :return: the current belief
        """

        return np.concatenate(
            (np.array((self.y_g / self.y_g_right_limit, self.theta / self.max_theta)), self.belief.flatten()), axis=0)

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :seed the seed for the random number generator(s)
        """

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def _move_gripah_left_soft(self):
        """
        Moves the gripah left with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g <= self.y_g_left_limit:
            return False

        self.y_g -= 1
        self._move_gripah_along_y(y_g=self.y_g, stiffness=self.low_stiffness)

        return True

    def _move_gripah_left_hard(self):
        """
        Moves the gripah left with a hard finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g <= self.y_g_left_limit:
            return False

        self.y_g -= 1
        self._move_gripah_along_y(y_g=self.y_g, stiffness=self.high_stiffness)

        return True

    def _move_gripah_right_soft(self):
        """
        Moves the gripah right with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g >= self.y_g_right_limit:
            return False

        self.y_g += 1
        self._move_gripah_along_y(y_g=self.y_g, stiffness=self.low_stiffness)

        return True

    def _move_gripah_right_hard(self):
        """
        Moves the gripah right with a hard finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.y_g >= self.y_g_right_limit:
            return False

        self.y_g += 1
        self._move_gripah_along_y(y_g=self.y_g, stiffness=self.high_stiffness)

        return True

    def _move_gripah_along_y(self, y_g, stiffness):
        """
        Moves the gripah with the given y_g and stiffness.

        :param y_g:       the target y_g for the gripah
        :param stiffness: the stiffness of the wide finger
        """

        self._set_wide_finger_stiffness(stiffness)

        if y_g <= self.y_hardware_right_limit:
            self.robot_controller.move_ur5_linear(self.y_origin + y_g * self.size_scale)
            time.sleep(0.3)

        self.theta = self._get_theta()

    def _get_theta(self):
        """
        Gets the current angle of the finger

        :return: the index of the joint
        """

        theta = self._get_wide_finger_angle()

        if theta < -0.11:
            theta_discretized = self.FINGER_POINT_LEFT
        elif theta > 0.12:
            theta_discretized = self.FINGER_POINT_RIGHT
        else:
            theta_discretized = self.FINGER_POINT_DOWN

        return theta_discretized

    def _get_wide_finger_angle(self):
        """
        Gets the current angle of the wide finger (finger #1).

        :return: the current angle of the wide finger
        """
        angles = self.robot_controller.read_rdda_positions()

        return float(angles[0] - self.theta_baseline)

    def _set_wide_finger_stiffness(self, stiffness):
        """
        Sets the stiffness of the joint of the wide finger (finger #1).

        :param stiffness: the stiffness to be set
        :return: None
        """
        self.robot_controller.set_rdda_stiffness(stiffness, self.fixed_stiffness)

    @staticmethod
    def is_bump_pushed():
        """
        Determines if the bumps pushed by the robot according to the human input.
        """

        while True:
            is_pushed_str = input("Is either bump pushed? [y/n] ")

            if is_pushed_str == 'y':
                return True
            elif is_pushed_str == 'n':
                return False
