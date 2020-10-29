#
# Created by Xinchao Song on June 15, 2020.
#

from warnings import warn

from gym.envs.plates.emulator.plates_env_states import states_transition_table


class PlateScriptSimulator:
    """
    The simulation environment of a four momdp_plates_v0 environment based on a states table.
    """

    def __init__(self):
        """
        The initialization of this script simulator.
        """
        self.num_plate_total = 4
        self.num_plate_left = 4
        self.top_plate = 0
        self.init_state = (0, 0)
        self.current_state = self.init_state

        # (num_plate_left, last state(x_g, angle), action): new state
        self.states_transition_table = states_transition_table

    def step(self, action):
        """
        Steps the simulation with the given action and returns the observations.

        :param action:
        0 - move up + finger soft
        1 - move down + finger soft
        2 - grasp a plate and place it on the table

        :return: the observations of the environment
        """

        if action == 0 or action == 1:
            if (self.num_plate_left, self.current_state, action) in self.states_transition_table:
                new_state = self.states_transition_table[(self.num_plate_left, self.current_state, action)]

            else:
                closest_state = None

                for err in range(5, 0, -1):
                    trial_state = (self.current_state[0], round(self.current_state[1] - 0.01 * err, 2))
                    trial_transition = (self.num_plate_left, trial_state, action)

                    if trial_transition in self.states_transition_table:
                        closest_state = trial_transition
                        continue

                    trial_state = (self.current_state[0], round(self.current_state[1] + 0.01 * err, 2))
                    trial_transition = (self.num_plate_left, trial_state, action)

                    if trial_transition in self.states_transition_table:
                        closest_state = trial_transition

                if closest_state is None:
                    new_state = (self.current_state[0] - 1, self.current_state[1])
                    warn('Script simulator warning: unknown state: (%d, (%d, %f), %d)' % (
                        self.num_plate_left, self.current_state[0], self.current_state[1], action))
                else:
                    new_state = self.states_transition_table[closest_state]

        elif action == 2:
            self.num_plate_left -= 1
            self.top_plate += 1
            new_state = (self.current_state[0], 0.0)
        else:
            raise ValueError("Unknown action received.")

        self.current_state = new_state

        return self.current_state

    def reset(self, num_plate):
        """
        Resets the current simulator with all given initial configurations.

        :num_plate: the number of momdp_plates_v0 on the stack
        :return: the observations of the environment
        """

        self.num_plate_left = num_plate
        self.current_state = self.init_state

        return self.current_state

    def render(self, mode='human'):
        """
        Renders the simulator. A dummy method.
        """
        pass

    def close(self):
        """
        Closes the current simulator. A dummy method.
        """
        pass
