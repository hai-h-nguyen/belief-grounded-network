import numpy as np
import gym
from gym import spaces
import copy

# Actions: L/S: Left/Right, S/H: Soft/Hard
a_LS = 0
a_LH = 1
a_RS = 2
a_RH = 3

# Angle positions: 1 is at the middle, 0: negative, 2: positive
POS_ANGLE = 2
NEG_ANGLE = 0
ZERO_ANGLE = 1

MAX_EP_LEN = 100

class BumpsEmulationEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.cart_pos = 0
        self.l_bump_pos = 0
        self.r_bump_pos = 0
        self.angle = ZERO_ANGLE  

        self.ep_len = 0 # episode length
        self.or_l_bump_pos = 0 # original pos of left bump
        self.or_r_bump_pos = 0 # original pos of right bump

        self.l_limit = 0
        self.r_limit = 41
        self.min_bump_distance = 2
        self.max_bump_distance = 20

        self.action_size = 4
        self.discount = 1.0

        # Count the number of states
        total_state = 0

        self.min_bump_L = 2
        self.min_bump_R = self.min_bump_L + self.min_bump_distance
        self.max_bump_R = self.r_limit - 1 - 2
        self.max_bump_L = self.max_bump_R - self.min_bump_distance

        # List all possible states (even invalid ones)
        self.valid_states = []

        for angle in range(0, 3):
            for cart_pos in range(self.l_limit, self.r_limit):
                for lb_pos in range(self.l_limit, self.r_limit):
                    for rb_pos in range(self.l_limit, self.r_limit):
                        state = [angle, cart_pos, lb_pos, rb_pos]

                        if (self.isValidState(state)):
                            self.valid_states.append(state)

        self.state_size = len(self.valid_states)

        # Add initial belief to all states (even invalid ones)
        self.full_belief = 1/(self.r_limit**2) * np.ones((self.r_limit, self.r_limit))

        # +2: belief size over the bumps position + cart pos + angle
        self.belief_size = self.r_limit ** 2

        # Size 3 = Observation (2) + action (1)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3,), dtype=np.float32)
        self.state_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_size)
        self.norm_action = self.action_size - 1.0

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)

    def get_obs(self):
        return np.array([self.angle/2.0, self.cart_pos/40.0])

    def get_state(self):
        return np.array([self.angle/2.0, self.cart_pos/40.0, self.l_bump_pos/40.0, self.r_bump_pos/40.0])

    def get_belief(self):
        return_vect = np.concatenate((np.array([self.angle/2.0, self.cart_pos/40.0]), self.full_belief.flatten()), axis=0)
        return return_vect

    def reset(self):
        # Get one random state from the state list
        validStartState = False

        while (not validStartState):
            random_index = np.random.randint(self.state_size)
            (_, self.cart_pos, self.l_bump_pos, self.r_bump_pos) = self.valid_states[random_index]
            self.angle = ZERO_ANGLE

            validStartState = self.isValidState([self.angle, self.cart_pos, self.l_bump_pos, self.r_bump_pos])

        # Save original bump position and cart side
        self.or_l_bump_pos = self.l_bump_pos
        self.or_r_bump_pos = self.r_bump_pos

        self.ep_len = 0

        self.full_belief = 1/(self.r_limit**2) * np.zeros((self.r_limit, self.r_limit))

        valid_state = 0

        for lbp in range(0, self.r_limit):
            for rbp in range(0, self.r_limit):
                state = [self.angle, self.cart_pos, lbp, rbp]
                if (self.isValidState(state)):
                    valid_state += 1

        for lbp in range(0, self.r_limit):
            for rbp in range(0, self.r_limit):
                state = [self.angle, self.cart_pos, lbp, rbp]
                if (self.isValidState(state)):
                    self.full_belief[lbp, rbp] = 1/valid_state

        return_vect = np.concatenate((np.array([(self.angle - 1.0)/2.0, self.cart_pos/40.0]), self.full_belief.flatten()), axis=0)       

        return np.array([self.angle/2.0, self.cart_pos/40.0, -1.0/self.norm_action])

    def _step(self, state, action):

        assert(self.isValidState(state))

        angle      = state[0]
        cart_pos   = state[1]
        l_bump_pos = state[2]
        r_bump_pos = state[3]

        # cart go left and finger soft
        if action == a_LS:
            # one of bumps is on the left
            if ((cart_pos == r_bump_pos + 1) or (cart_pos == l_bump_pos + 1)):
                angle = POS_ANGLE
            else:
                angle = ZERO_ANGLE

            # cart is moved left
            cart_pos = max(cart_pos - 1, self.l_limit)

        # cart go left and finger stiff
        elif action == a_LH:
            # right bump is here then it is moved left if angle is positive
            if ((cart_pos == r_bump_pos) and (angle == POS_ANGLE)):
                # r_bump_pos = max(r_bump_pos - 1, self.x_bump2_limit_min)
                r_bump_pos -= 1

            # left bump is here then it is moved left if angle is positive
            if ((cart_pos == l_bump_pos) and (angle == POS_ANGLE)):
                # l_bump_pos = max(l_bump_pos - 1, self.bump0_left_limit)
                l_bump_pos -= 1

            # left bump is on the left then it is moved left if angle is zero
            if ((cart_pos - 1 == l_bump_pos) and (angle == ZERO_ANGLE)):
                # l_bump_pos = max(l_bump_pos - 1, self.bump0_left_limit)
                l_bump_pos -= 1   

            # right bump is on the left then it is moved left if angle is zero
            if ((cart_pos - 1 == r_bump_pos) and (angle == ZERO_ANGLE)):
                # r_bump_pos = max(r_bump_pos - 1, self.x_bump2_limit_min)
                r_bump_pos -= 1               

            # cart is moved left
            cart_pos = max(cart_pos - 1, self.l_limit)

        # cart go right and finger soft
        elif action == a_RS:
            # one of bumps is here then the angle is negative
            if ((cart_pos == r_bump_pos - 1) or (cart_pos == l_bump_pos - 1)):
                angle = NEG_ANGLE
            else:
                angle = ZERO_ANGLE

            # cart is moved right
            cart_pos = min(cart_pos + 1, self.r_limit - 1)

        # cart go right and finger stiff
        # RH action
        elif action == a_RH:

            # right bump is here then it is moved right if angle is negative
            if (cart_pos == r_bump_pos and angle == NEG_ANGLE):
                # r_bump_pos = min(r_bump_pos + 1, self.x_bump2_limit_max)
                r_bump_pos += 1

            # left bump is here then it is moved right if angle is negative
            if (cart_pos == l_bump_pos and angle == NEG_ANGLE):
                # l_bump_pos = min(l_bump_pos + 1, self.x_bump1_limit_max)
                l_bump_pos += 1        

            # left bump is on the right then it is moved right if angle is zero
            if (cart_pos + 1 == l_bump_pos and angle == ZERO_ANGLE):
                # l_bump_pos = min(l_bump_pos + 1, self.x_bump1_limit_max)
                l_bump_pos += 1

            # right bump is on the right then it is moved right if angle is zero
            if (cart_pos + 1 == r_bump_pos and angle == ZERO_ANGLE):
                # r_bump_pos = min(r_bump_pos + 1, self.x_bump2_limit_max)
                r_bump_pos += 1

            # cart is moved right
            cart_pos = min(cart_pos + 1, self.r_limit - 1)

        else:
            raise ValueError("Unknown action", action)

        return [angle, cart_pos, l_bump_pos, r_bump_pos]

    def find_previous_state(self, state, action, prior, done):
        assert(action < self.action_size)

        angle      = prior[0]
        cart_pos   = prior[1]
        l_bump_pos = state[2]
        r_bump_pos = state[3] 

        possible_states = []

        # Only search 1 step around
        l_bump_pos_min = l_bump_pos - 1
        l_bump_pos_max = l_bump_pos + 2
        r_bump_pos_min = r_bump_pos - 1
        r_bump_pos_max = r_bump_pos + 2

        if ((action == a_LS) or (action == a_RS) or (not done)):
            l_bump_pos_min = l_bump_pos
            l_bump_pos_max = l_bump_pos + 1
            r_bump_pos_min = r_bump_pos
            r_bump_pos_max = r_bump_pos + 1

        for _left_bump in range(l_bump_pos_min, l_bump_pos_max):
            for _right_bump in range(r_bump_pos_min, r_bump_pos_max):
                test_state = [angle, cart_pos, _left_bump, _right_bump]
                if (not self.isValidState(test_state)):
                    continue
                else:
                    next_state = self._step(test_state, action)

                    if ((next_state == state) and ([_left_bump, _right_bump] not in possible_states)):
                        possible_states.append([_left_bump, _right_bump])

        return possible_states

    def step(self, action):

        if action == 4:
            return np.array([self.angle/2.0, self.cart_pos/40.0, float(action/self.norm_action)]), 0, False, {}

        saved_angle = self.angle
        saved_cart_pos = self.cart_pos
        saved_lpos = self.l_bump_pos
        saved_rpos = self.r_bump_pos

        prev_state = [saved_angle, saved_cart_pos, saved_lpos, saved_rpos]

        if (not self.isValidState(prev_state)):
            raise ValueError("Invalid state", prev_state)

        (self.angle, self.cart_pos, self.l_bump_pos, self.r_bump_pos)\
        = self._step(prev_state, action)

        self.ep_len += 1

        #-----------------------------------------------------------------
        ## Check for termination of the episode
        done = False

        cond1 = False
        cond2 = False
        cond3 = False

        # Left bump is moved, terminate
        if abs(self.l_bump_pos - self.or_l_bump_pos) > 0:
            cond1 = True

        # Right bump is moved, terminate
        if abs(self.r_bump_pos - self.or_r_bump_pos) > 0:
            cond2 = True

        # Episode is too long
        cond3 = (self.ep_len >= MAX_EP_LEN)

        # Either case will lead to termination
        done = cond1 or cond2 or cond3

        #-----------------------------------------------------------------
        ## Calculate reward
        reward = 0

        # Positive reward only when the agent moves the right bump at least 1 units to the right
        if self.r_bump_pos - self.or_r_bump_pos >= 1:
            reward = 1
        #-----------------------------------------------------------------
        ## Update belief
        old_belief = np.array(self.full_belief)

        # Loop through the belief state corresponding to this row
        for lbp in range(0, self.r_limit):
            for rbp in range(0, self.r_limit):
                state = [self.angle, self.cart_pos, lbp, rbp]
                if (not self.isValidState(state) and (not done)):
                    self.full_belief[lbp, rbp] = 0.0
                    continue
                else:
                    # Find the previous state
                    potential_pre_states = self.find_previous_state(state, action, [saved_angle, saved_cart_pos], done)

                    total_pre_state = len(potential_pre_states)
                    T_component = []
                    if (total_pre_state >= 1):
                        for i in range(total_pre_state):
                            (pre_lbp, pre_rbp) = potential_pre_states[i]
                            T_component.append(old_belief[pre_lbp, pre_rbp])
                        self.full_belief[lbp, rbp] = sum(T_component)
                    else:
                        self.full_belief[lbp, rbp] = 0.0

        # Normalize belief
        total_sum = sum(sum(self.full_belief))

        if (total_sum == 0):
            print(prev_state, action, [self.angle, self.cart_pos, self.l_bump_pos, self.r_bump_pos])

        assert(total_sum > 0)
        self.full_belief = self.full_belief / total_sum

        info = {}
        info['curr_state'] = self.get_state()
        info['pre_state'] = np.array(prev_state)
        info['belief'] = self.get_belief()

        return_vect = np.concatenate((np.array([(self.angle - 1.0)/2.0, self.cart_pos/40.0]), self.full_belief.flatten()), axis=0)

        return np.array([self.angle/2.0, self.cart_pos/40.0, float(action/self.norm_action)]), reward, done, info

    def isValidState(self, state):
        assert(isinstance(state, list))
        assert(len(state) == 4)

        angle = state[0]
        cart_pos = state[1]
        lb_pos = state[2]
        rb_pos = state[3]

        # Conditions for checking
        cond1 = (rb_pos - lb_pos >= self.min_bump_distance)
        cond2 = (rb_pos - lb_pos <= self.max_bump_distance)
        cond3 = (lb_pos >= self.min_bump_L)
        cond4 = (rb_pos <= self.max_bump_R)
        cond6 = (cart_pos >= 0) and (cart_pos <= self.r_limit - 1)

        cond5 = True
        if (cart_pos == lb_pos or cart_pos == rb_pos):
            cond5 = not(angle == ZERO_ANGLE)

        return (cond1 and cond2 and cond3 and cond4 and cond5 and cond6)


        
        




