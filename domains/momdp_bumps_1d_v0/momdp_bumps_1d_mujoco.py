#
# Created by Xinchao Song on June 1, 2020.
#

import operator
from pathlib import Path
from collections import namedtuple

import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from copy import deepcopy

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (ERROR: you need to install mujoco_py, "
        "and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

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
State = namedtuple('State', 'x_g, theta, x_bump1, x_bump2')



class MomdpBumps1dV0(gym.Env):
    """
    Description:

    The simulation environment of a general two bump environment. Based on the PlateScriptSimulator.

    Observation:
         Type: Discrete
         Num    Observation         Values
         0      Gripah Position X   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
         1      Gripah Angle        0 (THETA_LEFT), 1 (THETA_NEUTRAL), 2 (THETA_RIGHT)
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
         The starting state of the gripah is assigned to x_g = random and theta = THETA_NEUTRAL

     Episode Termination:
         when the gripah tried to push a bump, or
         when the gripah moves out the given range [0, 10].
     """

    metadata = {
        'render.modes': ['human', ]
    }

    def __init__(self, rendering=False, seed=None):
        """
        The initialization of the simulation environment.
        """

        # mujoco-py
        xml_path = Path(__file__).resolve().parent / 'mujoco_model' / 'bumps_1d_model.xml'
        self.model = mujoco_py.load_model_from_path(str(xml_path))
        self.sim = mujoco_py.MjSim(self.model)
        self.viewer = None  # Initializes only when self.render() is called.
        self.rendering = rendering

        # MuJoCo joints ids
        self.slide_x_id = self.model.joint_name2id('slide:gripah-base-x')
        self.hinge_wide_finger_id = self.model.joint_name2id('hinge:wide-finger')
        self.hinge_narrow_finger_id = self.model.joint_name2id('hinge:narrow-finger')
        self.slide_bump1_id = self.model.joint_name2id('slide:bump1')
        self.slide_bump2_id = self.model.joint_name2id('slide:bump2')

        # MuJoCo actuators ids
        self.velocity_x_id = self.model.actuator_name2id('velocity:x')
        self.velocity_narrow_finger_id = self.model.actuator_name2id('velocity:narrow-finger')
        self.position_narrow_finger_id = self.model.actuator_name2id('position:narrow-finger')

        # X range
        gripah_body_id = self.model.body_name2id("gripah-base")
        self.x_origin = self.model.body_pos[gripah_body_id][0]
        self.x_left_limit = 0
        self.x_right_limit = 14
        self.num_x_slots = self.x_right_limit + 1
        self.size_scale = 8  # 1 unit size in the code means 8 unit in the MuJoCo

        # X range marks in MuJoCo
        world_center_id = self.model.site_name2id("world-center")
        world_right_end_id = self.model.site_name2id("world-right-end")
        self.model.site_pos[world_center_id][0] = self.x_right_limit * self.size_scale / 2
        self.model.site_pos[world_right_end_id][0] = self.x_right_limit * self.size_scale

        # Actuator inner states
        self.default_velocity = 10
        self.qpos_nfinger = 0

        # Bumps
        self.ori_x_bump1 = None
        self.ori_x_bump2 = None
        self.x_bump1 = None
        self.x_bump2 = None
        self.min_bump_distance = 2
        self.x_bump1_limit_min = 2
        self.x_bump2_limit_min = self.x_bump1_limit_min + self.min_bump_distance
        self.x_bump2_limit_max = self.x_right_limit - 2
        self.x_bump1_limit_max = self.x_bump2_limit_max - self.min_bump_distance
        self.max_bump_distance = int(self.x_right_limit / 2)

        # Theta
        self.FINGER_POINT_RIGHT = THETA_RIGHT
        self.FINGER_POINT_DOWN = THETA_NEUTRAL
        self.FINGER_POINT_LEFT = THETA_LEFT
        self.max_theta = THETA_RIGHT
        self.angle_str = {THETA_LEFT: 't_left', THETA_NEUTRAL: 't_neutral', THETA_RIGHT: 't_right'}
        self.low_stiffness = 50
        self.high_stiffness = 10000

        # Action
        self.action_space = spaces.Discrete(4)
        self.norm_action = self.action_space.n - 1
        self.action_str = {ACT_LEFT_SOFT: 'Left&Soft', ACT_LEFT_HARD: 'Left&Hard', ACT_RIGHT_SOFT: 'Right&Soft',
                           ACT_RIGHT_HARD: 'Right&Hard'}

        # States and Obs
        self.discount = 1.0
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.x_g_left_limit = self.x_left_limit
        self.x_g_right_limit = self.x_right_limit
        self.x_g = None
        self.theta = self.FINGER_POINT_DOWN
        self.start_state = None
        self.all_valid_states = self._generate_valid_states()

        # Belief
        self.belief = np.zeros((self.num_x_slots, self.num_x_slots))

        # Debug
        self.transitions = []
        self.debug_belief = True


        self.traj_cnt = 0

        # numpy random
        self.seed(seed)

    def reset(self):
        """
        Resets the current MuJoCo simulation environment with given initial x coordinates of the two bumps.

        :return: the observations of the environment
        """

        # Resets the mujoco env
        self.sim.reset()

        # Determines the start state of x_g, theta, x_bump1, and x_bump2.
        self.x_bump1 = self.np_random.randint(self.x_bump1_limit_min, self.x_bump1_limit_max + 1)

        while True:
            self.x_bump2 = self.np_random.randint(self.x_bump2_limit_min, self.x_bump2_limit_max + 1)
            if self.min_bump_distance <= self.x_bump2 - self.x_bump1 <= self.max_bump_distance:
                break

        while True:
            self.x_g = self.np_random.randint(self.x_g_left_limit, self.x_g_right_limit + 1)
            if self.x_g != self.x_bump1 and self.x_g != self.x_bump2:
                break


        # Records the start state.
        self.ori_x_bump1 = self.x_bump1
        self.ori_x_bump2 = self.x_bump2

        # Assigns the start state to mujoco-py
        self.sim.data.qpos[self.slide_bump1_id] = self.x_bump1 * self.size_scale
        self.sim.data.qpos[self.slide_bump2_id] = self.x_bump2 * self.size_scale
        self.sim.data.qpos[self.slide_x_id] = self.x_g * self.size_scale
        self.actuate(pos_nfinger=0.9, teleport=True)
        self.theta = self._get_theta()
        self.start_state = State(x_g=self.x_g, theta=self.theta, x_bump1=self.x_bump1, x_bump2=self.x_bump2)
        assert (self._is_valid_state(self.start_state))


        # Belief
        self.belief = np.zeros((self.num_x_slots, self.num_x_slots))
        valid_start_states = []

        # Gets all valid start states.
        for _state in self.all_valid_states:
            if _state.x_g == self.x_g and _state.theta == self.theta and self._is_valid_state(_state):
                valid_start_states.append(_state)

        # Calculates the belief for all start states.
        start_state_belief = 1 / len(valid_start_states)

        for _state in valid_start_states:
            self.belief[_state.x_bump1, _state.x_bump2] = start_state_belief

        self.transitions = []
        self.transitions.append([self.start_state])

        # Save belief to file
        if (self.debug_belief):
            self.belief_file = open(str(self.traj_cnt) + ".txt","w")
            self.belief_file.write(';'.join(['%.5f' % num for num in self.belief.flatten()]) + '\n')

        return np.array((self.x_g / self.x_g_right_limit, self.theta / self.max_theta, -1.0 / self.norm_action))

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action:
        0 - move left + finger soft
        1 - move left + finger hard
        2 - move right + finger soft
        3 - move right + finger hard

        :return: the observations of the environment
        """

        reward = 0
        done = False
        pre_x_g = self.x_g
        pre_theta = self.theta

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

        # Episode Termination:
        #   when the gripah tried to push a bump, or
        #   when the gripah moves out the given range [0, 10].
        if self.x_bump1 != self.ori_x_bump1 or self.x_bump2 != self.ori_x_bump2:
            done = True
            self.traj_cnt += 1

        if not done:
            self._update_belief(action, pre_x_g, pre_theta)

        if self.x_bump2 > self.ori_x_bump2:
            reward = 1

        obs = np.array((self.x_g / self.x_g_right_limit, self.theta / self.max_theta, float(action / self.norm_action)))
        info = {'curr_state': self.get_state(), 'belief': self.get_belief()}

        self.transitions.append([self.x_g, self.theta, self.x_bump1, self.x_bump2, action])

        if (self.debug_belief):

            self.belief_file.write(str(action) + '\n')

            current_state = [self.x_g, self.theta, self.x_bump1, self.x_bump2]
            self.belief_file.write(';'.join(['%.d' % state for state in current_state]) + '\n')
            self.belief_file.write(';'.join(['%.6f' % num for num in self.belief.flatten()]) + '\n')


        return obs, reward, done, info

    def get_state(self):
        """
        Gets the current state.

        :return: the current state
        """

        state_normalized = State(x_g=self.x_g / self.x_g_right_limit,
                                 theta=self.theta / self.max_theta,
                                 x_bump1=self.x_bump1 / self.x_right_limit,
                                 x_bump2=self.x_bump2 / self.x_right_limit)

        return np.array(state_normalized)

    def test_transition(self):
        actions = [0, 2, 1, 2, 2, 0, 1, 0, 2, 0, 2, 1, 2, 1, 2, 1, 0]

        self.reset()

        for a in actions:
            self.step(a)

    def get_belief(self):
        """
        Gets the current belief.

        :return: the current belief
        """

        return np.concatenate(
            (np.array((self.x_g / self.x_g_right_limit, self.theta / self.max_theta)), self.belief.flatten()), axis=0)

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
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -15

            self.viewer.render()

    def close(self):
        """
        Closes the simulation environment.
        """

        # mujoco-py will close the env automatically.
        pass

    def seed(self, seed):
        """
        Sets the seed for this environment's random number generator(s).

        :seed the seed for the random number generator(s)
        """

        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def actuate(self, pos_x=None, pos_nfinger=None, teleport=False):
        """
        Actuates all actuators with the given parameters.

        :param pos_x:       the coordinate vector of x to be set
        :param pos_nfinger: the coordinate vector of the narrow finger to be set
        :param teleport:    teleport mode. The gripah will be teleported to the desired state without running
                            simulation. Note when running the actuator in teleport mode, the gripah is not able
                            to interact with other objects
        """

        if pos_x is not None:
            self._control_slider_x(x_target=self.x_origin + pos_x, teleport=teleport)

        if pos_nfinger is not None:
            self._control_narrow_finger(theta_target=pos_nfinger, teleport=teleport)

    def _generate_valid_states(self):
        """
        Generates a list of all valid states.

        :return: a list of all valid states
        """

        valid_states = []

        for _x_g in range(self.num_x_slots):
            for _theta in range(len(self.angle_str)):
                for _x_bump1 in range(self.num_x_slots):
                    for _x_bump2 in range(self.num_x_slots):
                        state = State(x_g=_x_g, theta=_theta, x_bump1=_x_bump1, x_bump2=_x_bump2)

                        if self._is_valid_state(state):
                            valid_states.append(state)

        return valid_states

    def _is_valid_state(self, state):
        """
        Determines if the given state is valid for calculating the belief.

        :param state: (x_g, theta, x_bump1, x_bump2)
        :return: True if the current state is valid, False otherwise
        """

        # Conditions for checking
        cond1 = self.min_bump_distance <= state.x_bump2 - state.x_bump1 <= self.max_bump_distance
        cond2 = self.x_bump1_limit_min <= state.x_bump1 <= self.x_bump1_limit_max
        cond3 = self.x_bump2_limit_min <= state.x_bump2 <= self.x_bump2_limit_max
        cond4 = self.x_g_left_limit <= state.x_g <= self.x_g_right_limit
        cond5 = (state.x_g != state.x_bump1 and state.x_g != state.x_bump2) or (state.theta != self.FINGER_POINT_DOWN)

        return cond1 and cond2 and cond3 and cond4 and cond5

    def _update_belief(self, action, prev_x_g, prev_theta):
        """
        Updates the belief.

        :param action: the action move the gripah to the current state
        :param prior:  (x_g, theta)
        """ 

        # print(self.x_g, self.theta, self.x_bump1, self.x_bump2, action)

        old_belief = np.array(self.belief)

        # Loops through the belief state corresponding to this row
        self.belief = np.zeros((self.num_x_slots, self.num_x_slots))

        for _x_bump1 in range(self.num_x_slots):
            for _x_bump2 in range(self.num_x_slots):
                state = State(x_g=self.x_g, theta=self.theta, x_bump1=_x_bump1, x_bump2=_x_bump2)

                if (self._is_valid_state(state)):

                    pre_state = State(x_g=prev_x_g, theta=prev_theta, x_bump1=_x_bump1, x_bump2=_x_bump2)

                    if self._is_valid_state(pre_state):
                        next_state = self._step_emulation(pre_state, action)

                        if next_state == state:
                            self.belief[state.x_bump1, state.x_bump2] = old_belief[_x_bump1, _x_bump2]

        # This should be never to happen.
        if sum(sum(self.belief)) == 0.0:
            print(self.transitions)

        assert (sum(sum(self.belief)) != 0.0)

        self.belief = self.belief / sum(sum(self.belief))

    def test_belief_update(self):
        prev_x_g = 9
        prev_theta = 1
        action = 0
        self.x_g = 8
        self.theta = 2

        self.belief = np.load('old_belief.npy')
        print(sum(sum(self.belief)))

        for _x_bump1 in range(self.num_x_slots):
            for _x_bump2 in range(self.num_x_slots):
                if (self.belief[_x_bump1, _x_bump2] != 0.0):
                    print(_x_bump1, _x_bump2)

        self._update_belief(action, prev_x_g, prev_theta)


    def _find_all_previous_states(self, state, action, pre_x_g, pre_theta):
        """
        Finds all possible previous states of the current given states.

        :param state:     (x_g, theta, x_bump1, x_bump2)
        :param action:    the action move the gripah to the current state
        :param pre_x_g:   prior x_g
        :param pre_theta: prior theta
        """

        previous_states = []

        # Only search 1 step around
        x_bump1_min = state.x_bump1
        x_bump1_max = state.x_bump1 + 1
        x_bump2_min = state.x_bump2
        x_bump2_max = state.x_bump2 + 1

        for _x_bump1 in range(x_bump1_min, x_bump1_max + 1):
            for _x_bump2 in range(x_bump2_min, x_bump2_max + 1):
                test_state = State(x_g=pre_x_g, theta=pre_theta, x_bump1=_x_bump1, x_bump2=_x_bump2)

                if self._is_valid_state(test_state):
                    next_state = self._step_emulation(test_state, action)

                    if (next_state == state) and (test_state not in previous_states):
                        previous_states.append(test_state)

        return previous_states

    def _step_emulation(self, state, action):
        """
        Emulates the step for calculating the belief

        :param state:  (x_g, theta, x_bump1, x_bump2)
        :param action: the action move the gripah to the current state
        """

        assert (self._is_valid_state(state))

        x_g = state.x_g
        theta = state.theta
        x_bump1 = state.x_bump1
        x_bump2 = state.x_bump2

        # The gripah goes left with a soft finger.
        if action == ACT_LEFT_SOFT:
            if x_g == x_bump1 + 1 or x_g == x_bump2 + 1:
                theta = self.FINGER_POINT_RIGHT
            else:
                theta = self.FINGER_POINT_DOWN

            x_g = max(x_g - 1, self.x_g_left_limit)

        # The gripah goes left with a hard finger.
        elif action == ACT_LEFT_HARD:
            is_push = False

            if (x_g == x_bump1 + 1 and theta == self.FINGER_POINT_DOWN) \
                    or (x_g == x_bump1 and theta == self.FINGER_POINT_RIGHT):
                is_push = True
                x_bump1 -= 1

            if (x_g == x_bump2 + 1 and theta == self.FINGER_POINT_DOWN) \
                    or (x_g == x_bump2 and theta == self.FINGER_POINT_RIGHT):
                is_push = True
                x_bump2 -= 1

            if is_push:
                # The resulting theta is different between pushing left and right because of the shape of
                # the RDDA finger. The left side of the finger is more curving than the right.
                theta = self.FINGER_POINT_RIGHT
            else:
                theta = self.FINGER_POINT_DOWN

            x_g = max(x_g - 1, self.x_g_left_limit)

        # The gripah goes right with a soft finger.
        elif action == ACT_RIGHT_SOFT:
            # one of bumps is here then the angle is negative
            if x_g == x_bump1 - 1 or x_g == x_bump2 - 1:
                theta = self.FINGER_POINT_LEFT
            else:
                theta = self.FINGER_POINT_DOWN

            x_g = min(x_g + 1, self.x_g_right_limit)

        # The gripah goes right with a hard finger.
        elif action == ACT_RIGHT_HARD:
            is_push = False

            if (x_g == x_bump1 - 1 and theta == self.FINGER_POINT_DOWN) \
                    or (x_g == x_bump1 and theta == self.FINGER_POINT_LEFT):
                is_push = True
                x_bump1 += 1

            if (x_g == x_bump2 - 1 and theta == self.FINGER_POINT_DOWN) \
                    or (x_g == x_bump2 and theta == self.FINGER_POINT_LEFT):
                is_push = True
                x_bump2 += 1

            if not is_push:
                theta = self.FINGER_POINT_DOWN

            x_g = min(x_g + 1, self.x_g_right_limit)

        else:
            return state

        return State(x_g=x_g, theta=theta, x_bump1=x_bump1, x_bump2=x_bump2)

    def _move_gripah_left_soft(self):
        """
        Moves the gripah left with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g <= self.x_g_left_limit:
            return

        self.x_g -= 1
        self._move_gripah_along_x(self.x_g, self.low_stiffness)

    def _move_gripah_left_hard(self):
        """
        Moves the gripah left with a hard finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g <= self.x_g_left_limit:
            return

        is_pushed = False

        if (self.x_g == self.x_bump1 + 1 and self.theta == self.FINGER_POINT_DOWN) \
                or (self.x_g == self.x_bump1 and self.theta == self.FINGER_POINT_RIGHT):
            is_pushed = True
            self.x_bump1 -= 1

        if (self.x_g == self.x_bump2 + 1 and self.theta == self.FINGER_POINT_DOWN) \
                or (self.x_g == self.x_bump2 and self.theta == self.FINGER_POINT_RIGHT):
            is_pushed = True
            self.x_bump2 -= 1

        self.x_g -= 1

        if is_pushed:
            # The resulting theta is different between pushing left and right because of the shape of the RDDA finger.
            # The left side of the finger is more curving than the right.
            self.theta = self.FINGER_POINT_RIGHT
        else:
            self._move_gripah_along_x(self.x_g, self.high_stiffness)

    def _move_gripah_right_soft(self):
        """
        Moves the gripah right with a soft finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g >= self.x_g_right_limit:
            return

        self.x_g += 1
        self._move_gripah_along_x(self.x_g, self.low_stiffness)

    def _move_gripah_right_hard(self):
        """
        Moves the gripah right with a hard finger in the allowed range.

        :return: True if the move succeeds, False otherwise
        """

        if self.x_g >= self.x_g_right_limit:
            return

        is_pushed = False

        if (self.x_g == self.x_bump1 - 1 and self.theta == self.FINGER_POINT_DOWN) \
                or (self.x_g == self.x_bump1 and self.theta == self.FINGER_POINT_LEFT):
            is_pushed = True
            self.x_bump1 += 1

        if (self.x_g == self.x_bump2 - 1 and self.theta == self.FINGER_POINT_DOWN) \
                or (self.x_g == self.x_bump2 and self.theta == self.FINGER_POINT_LEFT):
            is_pushed = True
            self.x_bump2 += 1

        self.x_g += 1

        if is_pushed:
            # The resulting theta is different between pushing left and right because of the shape of the RDDA finger.
            # The left side of the finger is more curving than the right.
            self.theta = self._get_theta()
        else:
            self._move_gripah_along_x(self.x_g, self.high_stiffness)

    def _move_gripah_along_x(self, x_g, stiffness):
        """
        Moves the gripah with the given x_g and stiffness.
        :param x_g:       the target x_g for the gripah
        :param stiffness: the stiffness of the wide finger
        """

        self._set_wide_finger_stiffness(stiffness)
        self.actuate(pos_x=x_g * self.size_scale)
        self.theta = self._get_theta()

    def _control_slider_x(self, x_target, teleport=False):
        """
        Controls the joint x of the gripah to move to the given target state.

        :param x_target: the target state that joint x of the gripah should move to.
        :param teleport:     teleport mode. The gripah will be teleported to the desired state without running
                             simulation. Note when running the actuator in teleport mode, the gripah is not able
                             to interact with other objects
        """

        if teleport:
            self.sim.data.qpos[self.slide_x_id] = x_target
            self.sim.data.ctrl[self.velocity_x_id] = 0
            self.sim.step()

            return

        if self._get_raw_x_g() < x_target:
            while self._get_raw_x_g() <= x_target:
                self.sim.data.ctrl[self.velocity_x_id] = self.default_velocity
                self.sim.step()
                self.render()

        elif self._get_raw_x_g() > x_target:
            while self._get_raw_x_g() >= x_target:
                self.sim.data.ctrl[self.velocity_x_id] = -self.default_velocity
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
        Gets the current angle of the angle of the wide finger

        :return: the current angle of the angle of the wide finger
        """

        theta = self._get_wide_finger_angle()
        if theta < -0.11:
            theta_discretized = self.FINGER_POINT_LEFT
        elif theta > 0.15:
            theta_discretized = self.FINGER_POINT_RIGHT
        else:
            theta_discretized = self.FINGER_POINT_DOWN

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

    def _get_raw_x_g(self):
        """
        Gets the raw value of x_g in MuJoCo.

        :return: the raw value of x_g
        """

        return self.sim.data.sensordata[0]

    def _get_gripah_raw_state(self):
        """
        Gets the current state of the gripah (x, y, z, and the angle of the narrow finger).

        :return: the current state of the gripah
        """

        x = self.sim.data.sensordata[0]
        y = self.sim.data.sensordata[1]
        z = self.sim.data.sensordata[2]
        w = self._get_narrow_finger_angle()

        return x, y, z, w
