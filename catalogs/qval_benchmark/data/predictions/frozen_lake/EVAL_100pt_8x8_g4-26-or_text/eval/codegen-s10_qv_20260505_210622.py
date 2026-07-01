import math

def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    The Q-value is approximated based on the distance to the goal and the risk of falling into holes.
    """
    def get_grid(s):
        # Split by newline and filter out genuinely empty lines.
        # We avoid using .strip() on the entire string to preserve leading/trailing spaces in rows.
        lines = s.split('\n')
        grid = []
        for line in lines:
            if line:  # Only add non-empty lines
                grid.append(line)
        return grid

    grid_s = get_grid(state)
    grid_ns = get_grid(next_state)

    if not grid_s or not grid_ns:
        return 0.0

    # Find the agent's new position '@' in the next state.
    r_at, c_at = -1, -1
    for r, row in enumerate(grid_ns):
        idx = row.find('@')
        if idx != -1:
            r_at, c_at = r, idx
            break
    
    # If the agent position cannot be found, we return a 0 value.
    if r_at == -1:
        return 0.0

    # Find the goal 'G' position. We check both state and next_state because
    # the '@' character might have replaced 'G' in the current state.
    r_g, c_g = -1, -1
    for r, row in enumerate(grid_s):
        idx = row.find('G')
        if idx != -1:
            r_g, c_g = r, idx
            break
    
    if r_g == -1:
        for r, row in enumerate(grid_ns):
            idx = row.find('G')
            if idx != -1:
                r_g, c_g = r, idx
                break

    # Identify all hole positions 'H' in the original state.
    holes = set()
    for r, row in enumerate(grid_s):
        for c, char in enumerate(row):
            if char == 'H':
                holes.add((r, c))

    # Check if the action led the agent directly into a hole.
    if (r_at, c_at) in holes:
        return 0.0
    
    # Check if the action led the agent directly to the goal.
    if r_g != -1 and r_at == r_g and c_at == c_g:
        return 1.0
    
    # If the goal cannot be identified, we cannot estimate a path to it.
    if r_g == -1:
        return 0.0

    # Calculate the Manhattan distance to the goal.
    # Since the environment is an 8x8 grid, the maximum Manhattan distance is 14.
    dist = abs(r_at - r_g) + abs(c_at - c_g)
    
    # Use a decay factor based on distance to reward shorter paths.
    # A decay of 0.8 means the value decreases significantly as the distance grows,
    # which helps the agent prioritize efficient paths toward the goal.
    return float(math.pow(0.8, dist))