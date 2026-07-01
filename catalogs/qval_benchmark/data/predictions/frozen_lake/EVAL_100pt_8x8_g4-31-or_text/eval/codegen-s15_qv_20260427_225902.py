def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The Q-value is approximated based on the Manhattan distance from the agent to the goal 
    in the resulting next state, considering obstacles (holes).
    """

    def get_coords(grid_str):
        coords = {'H': []}
        rows = grid_str.splitlines()
        for r, row in enumerate(rows):
            col = 0
            for char in row:
                if char == ' ':
                    continue
                if char == 'H':
                    coords['H'].append((r, col))
                else:
                    coords[char] = (r, col)
                col += 1
        return coords

    # Extract features from the state before the action
    state_coords = get_coords(state)
    # Extract features from the resulting next state
    next_coords = get_coords(next_state)

    # Agent's position in the next state
    a_pos = next_coords.get('@')
    # Goal's position in the initial state (to avoid confusion if agent is on it in next_state)
    g_pos = state_coords.get('G')
    # Locations of holes in the initial state
    h_positions = state_coords.get('H', [])

    # If agent is not found in the grid, assume a terminal failure or invalid state
    if a_pos is None:
        return 0.0

    # If agent reached the goal in the next state
    if g_pos is not None and a_pos == g_pos:
        return 1.0

    # If agent fell into a hole in the next state
    if a_pos in h_positions:
        return 0.0

    # If the goal is not found, we cannot estimate a path
    if g_pos is None:
        return 0.0

    # Use Manhattan distance as a heuristic for the expected discounted reward.
    # Reward is 1.0 at the goal, and the Q-value decays as distance increases.
    dist = abs(a_pos[0] - g_pos[0]) + abs(a_pos[1] - g_pos[1])
    
    # Using a decay factor of 0.9 per step. 
    # Max distance in 8x8 is 14, 0.9^14 approx 0.22.
    return 0.9 ** dist