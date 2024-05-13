import pykep as pk

class EarthToMarsEnvironment:
    def __init__(self):
        # Initialize environment parameters
        self.current_time = 0
        self.current_state = self.get_initial_state()

    def get_initial_state(self):
        # Compute initial state of spaceship using PyKEP
        initial_state = ...
        return initial_state

    def step(self, action):
        # Simulate one time step in the environment
        # Update spacecraft state based on action
        # Compute reward based on new state
        # Return observation, reward, and done flag
        ...

    def reset(self):
        # Reset environment to initial state
        self.current_time = 0
        self.current_state = self.get_initial_state()
        return self.current_state
