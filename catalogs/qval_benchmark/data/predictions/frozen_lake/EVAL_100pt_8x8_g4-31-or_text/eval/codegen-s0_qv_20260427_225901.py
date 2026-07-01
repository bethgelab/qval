import math

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the Frozen Lake environment.
    The Q-value is approximated as the expected discounted reward, 
    which we estimate using the agent's proximity to the goal.
    """

    def get_coords(grid_str, char):
        """Finds the (row, col) coordinates of the first occurrence of char in the grid."""
        lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
        for r, line in enumerate(lines):
            # Normalize line by removing spaces to handle different ASCII formats
            normalized_line = line.replace(' ', '')
            c = normalized_line.find(char)
            if c != -1:
                return (r, c)
        return None

    def get_all_coords(grid_str, char):
        """Finds all (row, col) coordinates of char in the grid."""
        coords = []
        lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
        for r, line in enumerate(lines):
            normalized_line = line.replace(' ', '')
            c = normalized_line.find(char)
            while c != -1:
                coords.append((r, c))
                c = normalized_line.find(char, c + 1)
        return coords

    # 1. Extract essential positions from the state
    g_pos = get_coords(state, 'G')
    h_positions = get_all_coords(state, 'H')
    agent_state_pos = get_coords(state, '@')
    
    # 2. Extract the agent's position in the next state
    agent_next_pos = get_coords(next_state, '@')
    
    # 3. Determine the outcome based on agent's position in next_state
    if agent_next_pos is None:
        # The agent is no longer visible in the grid, implying it either 
        # reached the goal 'G' or fell into a hole 'H'.
        if agent_state_pos is None:
            return 0.0
            
        # Calculate the target coordinates based on the action taken
        r, c = agent_state_pos
        if action == 'up':
            rt, ct = r - 1, c
        elif action == 'down':
            rt, ct = r + 1, c
        elif action == 'left':
            rt, ct = r, c - 1
        elif action == 'right':
            rt, ct = r, c + 1
        else:
            rt, ct = r, c
            
        # If the target move was exactly on the goal, the value is 1.0
        if g_pos and (rt, ct) == g_pos:
            return 1.0
        # Otherwise, assume it fell into a hole or the episode ended for other reasons
        return 0.0

    # 4. If agent is still in the grid, evaluate the resulting state
    # If the agent is positioned exactly on a hole
    if agent_next_pos in h_positions:
        return 0.0
    
    # If the agent is positioned exactly on the goal
    if g_pos and agent_next_pos == g_pos:
        return 1.0
    
    # Heuristic: Value the state based on Manhattan distance to the goal
    if g_pos:
        dist = abs(agent_next_pos[0] - g_pos[0]) + abs(agent_next_pos[1] - g_pos[1])
        # Use a discount factor (gamma = 0.9) to approximate the expected return
        return 0.9 ** dist
    
    return 0.0