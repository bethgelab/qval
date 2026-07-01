import math

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for a Frozen Lake environment.
    The Q-value is approximated based on the proximity of the next state to the goal,
    taking into account whether the agent has reached the goal or fallen into a hole.
    """
    
    def get_coords(grid_str, char):
        lines = grid_str.strip().split('\n')
        for r, line in enumerate(lines):
            for c, val in enumerate(line):
                if val == char:
                    return r, c
        return None

    # Identify key positions in the states
    sr, sc = get_coords(state, '@')
    ar, ac = get_coords(next_state, '@')
    gr, gc = get_coords(next_state, 'G')
    
    # If goal is not found, we can't estimate distance
    if gr is None or gc is None:
        return 0.0

    # Determine the intended next position based on the action
    nr, nc = sr, sc
    if action == 'up':
        nr -= 1
    elif action == 'down':
        nr += 1
    elif action == 'left':
        nc -= 1
    elif action == 'right':
        nc += 1
    
    # Grid boundary check (8x8)
    if not (0 <= nr < 8 and 0 <= nc < 8):
        nr, nc = sr, sc

    # Case 1: Agent is no longer present in next_state
    # This usually happens if the agent falls into a hole or reaches the goal.
    if ar is None or ac is None:
        # If the intended move lands on the goal, the return is 1.0
        if nr == gr and nc == gc:
            return 1.0
        # Otherwise, it likely fell into a hole or timed out
        return 0.0

    # Case 2: Agent is present in next_state
    # If the agent reached the goal, Q-value is 1.0
    if ar == gr and ac == gc:
        return 1.0
    
    # Check if the agent moved into a hole.
    # We check the symbol at the agent's new position in the original state grid.
    state_lines = state.strip().split('\n')
    if 0 <= ar < len(state_lines) and 0 <= ac < len(state_lines[ar]):
        if state_lines[ar][ac] == 'H':
            return 0.0

    # Case 3: Agent is on the frozen lake.
    # Estimate Q-value based on the discounted distance to the goal.
    # Using a discount factor of 0.9 to encourage efficiency.
    dist = abs(ar - gr) + abs(ac - gc)
    return math.pow(0.9, dist)