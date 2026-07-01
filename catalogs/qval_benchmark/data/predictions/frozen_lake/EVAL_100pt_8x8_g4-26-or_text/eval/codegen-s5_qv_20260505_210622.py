def signal_function(state: str, action: str, next_state: str) -> float:
    def get_grid(s: str):
        grid = []
        for line in s.split('\n'):
            row = [c for c in line if c not in (' ', '\t', '\r', '\n')]
            if row:
                grid.append(row)
        return grid

    grid_s = get_grid(state)
    grid_ns = get_grid(next_state)

    if not grid_s or not grid_ns:
        return 0.0

    # Find agent position in next_state
    r_ns, c_ns = -1, -1
    for r, row in enumerate(grid_ns):
        for c, val in enumerate(row):
            if val == '@':
                r_ns, c_ns = r, c
                break
        if r_ns != -1:
            break

    # Find goal position (prefer looking in grid_s to avoid 'G' being replaced by '@')
    r_g, c_g = -1, -1
    for r, row in enumerate(grid_s):
        for c, val in enumerate(row):
            if val == 'G':
                r_g, c_g = r, c
                break
        if r_g != -1:
            break

    # If goal not found in grid_s, look in grid_ns
    if r_g == -1:
        for r, row in enumerate(grid_ns):
            for c, val in enumerate(row):
                if val == 'G':
                    r_g, c_g = r, c
                    break
            if r_g != -1:
                break

    # Find hole positions in grid_s
    holes = set()
    for r, row in enumerate(grid_s):
        for c, val in enumerate(row):
            if val == 'H':
                holes.add((r, c))

    # If we can't locate critical points, return 0.0
    if r_ns == -1 or r_g == -1:
        return 0.0

    # Immediate outcome checks
    if r_ns == r_g and c_ns == c_g:
        return 1.0
    
    # Check if the new position is a hole
    # The hole might have been covered by '@' in grid_ns, 
    # so we verify against the hole positions found in grid_s.
    if (r_ns, c_ns) in holes:
        return 0.0

    # Manhattan distance to goal as a heuristic for Q-value
    dist = abs(r_ns - r_g) + abs(c_ns - c_g)
    
    # A linear decay based on distance. Max distance in an 8x8 grid is 14.
    # We scale the decay so that reaching the goal is preferred.
    q_value = max(0.0, 1.0 - (dist / 15.0))
    
    return float(q_value)