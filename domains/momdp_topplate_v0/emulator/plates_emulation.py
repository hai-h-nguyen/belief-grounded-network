#
# Created by Xinchao Song and Hai Nguyen on June 1, 2020.
#
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

FAR_POSITIVE = 4
POSITIVE = 3
NEUTRAL = 2
NEGATIVE = 1
FAR_NEGATIVE = 0

MOVE_DOWN = 0
MOVE_UP = 1
GRASP = 2


class PlatesEmulationEnv(gym.Env):
    """
    Description:

    The emulation environment of a general two momdp_plates_v0 environment.

    Observation:
         Type: Discrete
         Num    Observation         Values
         0      Gripah Position     Down_Limit = 0, 1, 2, 3, 4, 5, 6, ... , Up_Limit
         1      Gripah Angle        0 (far negative), 1 (negative), 2 (neutral), 3 (positive), 4 (far positive)

    Actions:
         Type: Discrete(3)
         Num    Action
         0      moving down with soft finger
         1      Moving up with soft finger
         2      Grasping a plate and place it on the table

    Reward:
         Reward of 1 for successfully place one plate on an empty position

     Starting State:
         The starting state of the gripah is assigned to x_g = 0 and theta = neutral

     Episode Termination:
         When the gripah tried to grasp a plate.
     """

    metadata = {
        'render.modes': ['human', ]
    }

    def __init__(self, seed=None):
        """
        The initialization of the emulation environment.
        """

        # Plates
        self.plate_max_idx = 10
        self.num_plate_selected = None
        self.target_plate_idx = None
        self.selected_plates_idx = []

        # x_g
        self.z_g_limit_down = 0
        self.z_g_limit_up = self.plate_max_idx + 3  # The gripper is allowed to travel a little bit more

        # Theta
        self.theta_limit_negative = 0
        self.theta_limit_positive = 4
        self.FINGER_FAR_POSITIVE = FAR_POSITIVE
        self.FINGER_POSITIVE = POSITIVE
        self.FINGER_NEUTRAL = NEUTRAL
        self.FINGER_NEGATIVE = NEGATIVE
        self.FINGER_FAR_NEGATIVE = FAR_NEGATIVE

        # We want to fix the belief size to avoid the 
        # tests infer the top plate from the size of belief
        # The top plate can be anywhere from [1] to [self.z_g_level_limit_up - 1]
        self.belief_size = self.z_g_limit_up - 1

        # We calculate the success rate
        self.discount = 1.0

        # Seed
        self.np_random = None
        self.seed(seed)

        # gym
        self.action_space = spaces.Discrete(3)
        # State + Action -> dim=3
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)

        # A crude way to scale actions to [0, 1]
        self.norm_action = self.action_space.n - 1

        # State parameters initialization
        self.z_t = None
        self.z_g = None
        self.theta = None
        self.belief = None

    def reset(self):
        """
        Resets variables and states
        :return: the observations of the environment
        """

        # Determines the number of momdp_plates_v0 to use.
        # self.num_plate_selected can be set from outside through self.set_num_plate(num).
        if self.num_plate_selected is None:
            self.num_plate_selected = self.np_random.randint(1, self.plate_max_idx + 1)

        # Setups momdp_plates_v0 on the stack
        plate_indices = list(range(1, self.plate_max_idx + 1))
        self.selected_plates_idx = plate_indices[:self.num_plate_selected]

        # Determines the index of the target plate
        # self.target_plate_idx can be set from outside through self.set_target_plate_index(index).
        if self.target_plate_idx is None:
            self.target_plate_idx = self.np_random.choice(self.selected_plates_idx)

        # Resets all state parameters
        self.z_t = self.target_plate_idx
        self.z_g = self.z_g_limit_down
        self.theta = self.FINGER_NEUTRAL
        self.belief = np.ones((self.belief_size, 1)) / self.belief_size

        # print("Reset", self.x_g, self.theta, self.z_t, self.belief)

        return np.array([self.z_g / self.z_g_limit_up, self.theta / self.theta_limit_positive, -1.0 / self.norm_action])

    def step(self, action):
        """
        Steps the emulation with the given action and returns the observations.

        :param action:
        0 - move down + finger soft
        1 - move up + finger soft
        2 - grasp a plate and place it on the table

        :return: the observations of the environment
        """

        done = False
        reward = 0

        # Executes the determined action.
        if action == 0:
            self._move_gripah_down()
            self._update_belief()  # Update only with movement commands

        elif action == 1:
            self._move_gripah_up()
            self._update_belief()  # Update only with movement commands

        elif action == 2:
            done = True

            if self._grasp():
                reward += 1

        else:
            raise ValueError("Unknown action index received.")

        obs = np.array(
            [self.z_g / self.z_g_limit_up, self.theta / self.theta_limit_positive, float(action / self.norm_action)])
        info = {'curr_state': self.get_state(), 'belief': self.get_belief()}

        return obs, reward, done, info

    def _update_belief(self):
        """
        Updates the belief.
        """

        old_belief = np.array(self.belief)

        for z_t in range(self.belief_size):
            state = [self.z_g, self.theta, z_t]

            if self._isValidState(state):
                self.belief[z_t] = old_belief[z_t]
            else:
                self.belief[z_t] = 0.0

        self.belief = self.belief / np.sum(self.belief)

    def _isValidState(self, state):
        """
        Determines if the given state is valid for calculating the belief.
        :param state: [self.x_g, self.theta, z_t]
        """
        z_g = state[0]
        theta = state[1]
        z_t = state[2]

        if z_t == 0:
            return False

        if (theta == self.FINGER_NEGATIVE) \
                or (theta == self.FINGER_POSITIVE) \
                or (theta == self.FINGER_NEUTRAL):
            return z_g != z_t

        if (theta == self.FINGER_FAR_NEGATIVE) \
                or (theta == self.FINGER_FAR_POSITIVE):
            return z_g == z_t

    def generate_pomdps(self):
        z_g_range = range(self.z_g_limit_down, self.z_g_limit_up + 1)
        z_t_range = range(1, self.plate_max_idx + 1)
        theta_range = range(self.theta_limit_negative, self.theta_limit_positive + 1)
        texts = []
        states = []

        texts.append('discount: %f\n' % self.discount)
        texts.append('values: reward\n')
        texts.append('states: ')

        for z_g in z_g_range:
            if z_g in z_t_range:
                for theta in theta_range:
                    for z_t in z_t_range:
                        s = 's_%d_%d_%d ' % (z_g, theta, z_t)
                        states.append(s)
                        texts.append(s)
                    texts.append('\n        ')
            else:
                for z_t in z_t_range:
                    s = 's_%d_%d_%d ' % (z_g, self.FINGER_NEUTRAL, z_t)
                    states.append(s)
                    texts.append(s)
                texts.append('\n        ')

        texts[-1] = '\nactions: MOVE_LEFT MOVE_RIGHT ACT_GRASP\n'

        texts.append('observations: ')

        for z_g in z_g_range:
            if z_g in z_t_range:
                for theta in theta_range:
                    texts.append('o_%d_%d ' % (z_g, theta))
            else:
                texts.append('o_%d_%d ' % (z_g, self.FINGER_NEUTRAL))

            texts.append('\n              ')

        print(''.join(texts))

        # TRANSITIONS
        print('T: %d : %s : %s' % (1, states[0], self._step_pomdps(states[0], 1)))

    def _step_pomdps(self, state_index, action):
        z_g, theta, z_t = map(int, state_index[2:].split('_'))

        temp_env = PlatesEmulationEnv()
        temp_env.selected_plates_idx = list(range(1, temp_env.plate_max_idx + 1))
        temp_env.z_g = z_g
        temp_env.theta = theta
        temp_env.z_t = z_t

        if action == 0:
            temp_env._move_gripah_down()
        elif action == 1:
            temp_env._move_gripah_up()
        elif action == 2:
            temp_env._grasp()
        else:
            raise ValueError("Unknown action index received.")

        return 's_%d_%d_%d' % (temp_env.z_g, temp_env.theta, temp_env.z_t)

    def get_state(self):
        """
        Gets the current normalized state.
        :return: the current state
        """
        return np.array(
            [self.z_g / self.z_g_limit_up, self.theta / self.theta_limit_positive, self.z_t / self.plate_max_idx])

    def get_belief(self):
        """
        Gets the current belief.
        :return: the current belief
        """
        return np.concatenate(
            (np.array([self.z_g / self.z_g_limit_up, self.theta / self.theta_limit_positive]), self.belief.flatten()),
            axis=0)

    def set_num_plate(self, num):
        """
        Sets the number of momdp_plates_v0 selected.
        :num: the number of momdp_plates_v0 selected; if set to None,
              the number will be random
        """
        self.num_plate_selected = num

    def set_target_plate_index(self, index):
        """
        Sets the index of target momdp_plates_v0.
        :num: the number of momdp_plates_v0 selected; if set to None,
              the number will be random
        """
        self.target_plate_idx = index

    def render(self, mode='human'):
        """
        Renders the environment.
        """

        pass

    def close(self):
        """
        Closes the emulation environment.
        """

        pass

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator(s).

        :param seed the seed for the random number generator(s)
        :return: a list of seed
        """
        self.np_random, seed_ = seeding.np_random(seed)

        return [seed_]

    def _move_gripah_up(self):
        """
        Moves the gripah up with a soft finger.
        """

        self.z_g = min(self.z_g_limit_up, self.z_g + 1)

        # Moving up and there is a plate at the same position
        if self.z_g in self.selected_plates_idx:
            if self.z_g == self.z_t:
                self.theta = self.FINGER_FAR_NEGATIVE
            else:
                self.theta = self.FINGER_NEGATIVE

        # No plate underneath
        else:
            self.theta = self.FINGER_NEUTRAL

    def _move_gripah_down(self):
        """
        Moves the gripah down with a soft finger.
        """

        self.z_g = max(self.z_g_limit_down, self.z_g - 1)

        # Moving down, plate underneath
        if self.z_g in self.selected_plates_idx:
            if self.z_g == self.z_t:
                self.theta = self.FINGER_FAR_POSITIVE
            else:
                self.theta = self.FINGER_POSITIVE

        # No plate underneath
        else:
            self.theta = self.FINGER_NEUTRAL

        return True

    def _grasp(self):
        """
        Grasps a plate and place it on the table.

        :return: True if succeed, False otherwise
        """

        if self.z_g != self.z_t:
            return False

        self.theta = self.FINGER_NEUTRAL

        return True

    def test_belief(self):
        # Go up then go down
        action_series = [
            [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_UP, MOVE_UP, MOVE_UP],
            [MOVE_UP, MOVE_UP, MOVE_DOWN, MOVE_DOWN, MOVE_UP, MOVE_UP, MOVE_UP],
            [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN],
            [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN]]
        z_g_series = [[1, 2, 3, 4, 3, 2, 1, 2, 3, 4],
                      [1, 2, 1, 0, 1, 2, 3],
                      [1, 2, 3, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5, 4, 3, 2]]
        theta_expected_series = [
            [NEGATIVE, FAR_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, FAR_POSITIVE, POSITIVE, FAR_NEGATIVE, NEGATIVE,
             NEUTRAL],
            [FAR_NEGATIVE, NEUTRAL, FAR_POSITIVE, NEUTRAL, FAR_NEGATIVE, NEUTRAL, NEUTRAL],
            [NEGATIVE, FAR_NEGATIVE, NEUTRAL, NEUTRAL, NEUTRAL, FAR_POSITIVE, POSITIVE],
            [NEGATIVE, NEGATIVE, FAR_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, FAR_POSITIVE, POSITIVE]]
        num_plate = [3, 1, 2, 4]
        target_indices = [2, 1, 2, 3]

        for test_num in range(len(action_series)):
            actions = action_series[test_num]
            z_gs = z_g_series[test_num]
            thetas_expected = theta_expected_series[test_num]

            self.set_num_plate(num_plate[test_num])
            self.set_target_plate_index(target_indices[test_num])
            self.reset()

            for i, a in enumerate(actions):
                obs, reward, done, info = self.step(a)

                print("#", test_num, ",", i)
                print("x_g:", self.z_g)
                print("theta:", self.theta)
                print("z_t:", self.z_t)
                print("belief:")
                print(self.belief.flatten())
                print()

                assert (self.z_t == target_indices[test_num])
                assert (self.z_g == z_gs[i])
                assert (self.theta == thetas_expected[i])
                assert (done == 0)

    def test_transitions(self):
        # Case 1: Go up all the way
        for num_plate in range(1, 5):
            self.set_num_plate(num_plate)
            self.set_target_plate_index(num_plate)
            self.reset()

            assert (self.num_plate_selected == num_plate)
            assert (self.z_t == num_plate)

            actions = [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP]
            dones = [0, 0, 0, 0, 0, 0, 0, 0]
            z_gs = [1, 2, 3, 4, 5, 6, 7, 8]
            thetas_expected = [NEGATIVE] * len(actions)
            thetas_expected[num_plate - 1] = FAR_NEGATIVE
            for i in range(num_plate, len(thetas_expected)):
                thetas_expected[i] = NEUTRAL

            for i in range(len(actions)):
                obs, reward, done, info = self.step(actions[i])
                assert (self.z_t == num_plate)
                assert (self.z_g == z_gs[i])
                assert (self.theta == thetas_expected[i])
                assert (done == dones[i])

        # Case 2: Go up all the way and then go down all the way
        for num_plate in range(1, 5):
            self.set_num_plate(num_plate)
            self.set_target_plate_index(num_plate)
            self.reset()

            assert (self.num_plate_selected == num_plate)
            assert (self.z_t == num_plate)

            # Go up all the way
            actions_up = [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP]
            dones = [0, 0, 0, 0, 0, 0, 0]
            z_gs = [1, 2, 3, 4, 5, 6, 7]
            thetas_expected = [NEGATIVE] * len(actions_up)
            thetas_expected[num_plate - 1] = FAR_NEGATIVE
            for i in range(num_plate, len(thetas_expected)):
                thetas_expected[i] = NEUTRAL

            for i in range(len(actions_up)):
                obs, reward, done, info = self.step(actions_up[i])
                assert (self.z_t == num_plate)
                assert (self.z_g == z_gs[i])
                assert (self.theta == thetas_expected[i])
                assert (done == dones[i])

            # Go down all the way
            actions_down = [MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN]
            dones = [0, 0, 0, 0, 0, 0, 0, 0]
            z_gs = [6, 5, 4, 3, 2, 1, 0, 0]
            thetas_expected = [POSITIVE] * len(actions_down)
            thetas_expected[len(actions_down) - 2 - num_plate] = FAR_POSITIVE
            for i in range(len(actions_down) - 2 - num_plate):
                thetas_expected[i] = NEUTRAL
            for i in range(len(actions_down) - 2, len(actions_down)):
                thetas_expected[i] = NEUTRAL

            for i in range(len(actions_down)):
                obs, reward, done, info = self.step(actions_down[i])
                assert (self.z_t == num_plate)
                assert (self.z_g == z_gs[i])
                assert (self.theta == thetas_expected[i])
                assert (done == dones[i])

            # Case 3: Go up then go down
            num_plate = 3
            self.set_num_plate(num_plate)
            self.set_target_plate_index(num_plate)
            self.reset()

            actions = [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, MOVE_DOWN, MOVE_DOWN, MOVE_DOWN, MOVE_UP, MOVE_UP, MOVE_UP]
            dones = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            z_gs = [1, 2, 3, 4, 3, 2, 1, 2, 3, 4]
            thetas_expected = [NEGATIVE, NEGATIVE, FAR_NEGATIVE, NEUTRAL, FAR_POSITIVE, POSITIVE, POSITIVE, NEGATIVE,
                               FAR_NEGATIVE, NEUTRAL]

            for i in range(len(actions)):
                obs, reward, done, info = self.step(actions[i])
                assert (self.z_t == num_plate)
                assert (self.z_g == z_gs[i])
                assert (self.theta == thetas_expected[i])
                assert (done == dones[i])

            # Case 4: Go up then grasp
            num_plate = 3
            self.set_num_plate(num_plate)
            self.set_target_plate_index(num_plate)
            self.reset()

            actions = [MOVE_UP, MOVE_UP, MOVE_UP, GRASP]
            z_gs = [1, 2, 3, 3]
            dones = [0, 0, 0, 1]
            rewards = [0, 0, 0, 1]

            for i in range(len(actions)):
                obs, reward, done, info = self.step(actions[i])

                assert (self.z_g == z_gs[i])
                assert (reward == rewards[i])
                assert (done == dones[i])
                assert (self.z_t == num_plate)

            # Case 5: Go up then grasp at a wrong place
            num_plate = 3
            self.set_num_plate(num_plate)
            self.set_target_plate_index(num_plate)
            self.reset()

            actions = [MOVE_UP, MOVE_UP, MOVE_UP, MOVE_UP, GRASP]
            z_gs = [1, 2, 3, 4, 4]
            dones = [0, 0, 0, 0, 1]

            for i in range(len(actions)):
                obs, reward, done, info = self.step(actions[i])

                assert (self.z_g == z_gs[i])
                assert (done == dones[i])
                assert (self.z_t == num_plate)
