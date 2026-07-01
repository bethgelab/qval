import math

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The Q-value represents the expected discounted cumulative reward.
    """
    
    def parse_grid(s: str):
        # Split the grid into lines and remove any empty strings
        lines = [line for line in s.split('\n') if line.strip()]
        # Convert each line into a list of characters for easy coordinate access
        return [list(line) for line in lines]

    # Parse both current and next state grids
    grid_s = parse_grid(state)
    grid_n = parse_grid(next_state)

    # 1. Identify the locations of the Goal ('G') and Holes ('H')
    # We look at both grids because the goal or hole might be 
    # replaced by the agent icon '@' in the next state.
    goal_pos = None
    holes = set()

    # Look in the original state
    for r, row in enumerate(grid_s):
        for c, char in enumerate(row):
            if char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))
    
    # Look in the next state (in case G or H is still visible)
    for r, row in enumerate(grid_n):
        for c, char in enumerate(row):
            if char == 'G':
                goal_pos = (r, c)
            elif char == 'H':
                holes.add((r, c))

    # 2. Identify the agent's current position in the next state
    agent_pos = None
    for r, row in enumerate(grid_n):
        for c, char in enumerate(row):
            # The agent is primarily marked by '@'
            if char == '@':
                agent_pos = (r, c)
                break
        if agent_pos:
            break
            
    # If '@' is not found, the agent might be on a cell that is now 'G' or 'H'
    if agent_pos is None:
        for r, row in enumerate(grid_n):
            for c, char in enumerate(row):
                if char == 'G' or char == 'H':
                    agent_pos = (r, c)
                    break
            if agent_pos:
                break

    # If we cannot find the agent's position, return a default low value
    if agent_pos is None:
        return 0.0

    # 3. Evaluate the outcome based on the agent's position
    # If the agent reached the goal, the reward is 1.0
    if goal_pos and agent_pos == goal_pos:
        return 1.0
    
    # If the agent fell into a hole, the reward is 0.0
    if holes and agent_pos in holes:
        return 0.0

    # 4. Heuristic for non-terminal states: distance-based estimation
    # Q(s, a) is roughly gamma^dist, where dist is the distance to the goal.
    # This reflects the expected discounted reward assuming an optimal policy.
    if goal_pos:
        # Use Manhattan distance as an efficient proxy for path length
        dist = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
        # Gamma of 0.98 provides a smooth decay for an 8x8 grid
        return math.pow(0.98, dist)

    return 0.0