def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The estimate is based on the proximity to the goal and the immediate reward outcome.
    """
    def get_positions(grid_str: str, char: str):
        positions = []
        for r, line in enumerate(grid_str.split('\n')):
            if not line:
                continue
            idx = line.find(char)
            while idx != -1:
                positions.append((r, idx))
                idx = line.find(char, idx + 1)
        return positions

    # Identify the agent's current position in the next state
    agent_positions = get_positions(next_state, '@')
    if not agent_positions:
        return 0.0
    agent_pos = agent_positions[0]

    # Identify goal and hole positions from the static map (the state string)
    # The state string contains the terrain without the agent's movement applied,
    # so 'G' and 'H' are guaranteed to be at their fixed locations.
    goal_positions = get_positions(state, 'G')
    hole_positions = get_positions(state, 'H')

    # If the agent has reached the goal
    if goal_positions and agent_pos in goal_positions:
        return 1.0

    # If the agent has fallen into a hole
    if hole_positions and agent_pos in hole_positions:
        return 0.0

    # If the agent is in a safe state, estimate value based on Manhattan distance to the goal.
    # Q(s, a) is approximately gamma^dist when the reward is sparse and only at the goal.
    if goal_positions:
        # Find the Manhattan distance to the nearest goal
        min_dist = min(
            abs(agent_pos[0] - g[0]) + abs(agent_pos[1] - g[1]) 
            for g in goal_positions
        )
        # Use a discount factor (e.g., 0.95) to reward efficiency and proximity
        return 0.95 ** min_dist

    return 0.0