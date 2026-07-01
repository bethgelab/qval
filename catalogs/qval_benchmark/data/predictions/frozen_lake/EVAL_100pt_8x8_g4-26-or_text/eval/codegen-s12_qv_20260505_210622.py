def signal_function(state: str, action: str, next_state: str) -> float:
    import math

    def parse_grid(grid_str):
        grid = []
        for line in grid_str.strip().split('\n'):
            row = line.replace(' ', '').strip()
            if row:
                grid.append(list(row))
        return grid

    def find_positions(grid):
        ra, ca, rg, cg, holes = -1, -1, -1, -1, []
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                char = grid[r][c]
                if char == '@':
                    ra, ca = r, c
                elif char == 'G':
                    rg, cg = r, c
                elif char == 'H':
                    holes.append((r, c))
        return ra, ca, rg, cg, holes

    grid_s = parse_grid(state)
    grid_ns = parse_grid(next_state)

    ra_s, ca_s, rg_s, cg_s, holes_s = find_positions(grid_s)
    ra_ns, ca_ns, rg_ns, cg_ns, holes_ns = find_positions(grid_ns)

    # 1. Check if agent reached the Goal in next_state
    # If 'G' is replaced by '@', rg_ns will be -1.
    is_goal = False
    if rg_ns != -1:
        if ra_ns == rg_ns and ca_ns == cg_ns:
            is_goal = True
    elif rg_s != -1:
        if ra_ns == rg_s and ca_ns == cg_s:
            is_goal = True

    if is_goal:
        return 1.0

    # 2. Check if agent fell into a Hole in next_state
    is_hole = False
    if ra_ns != -1:
        if (ra_ns, ca_ns) in holes_ns:
            is_hole = True
        elif (ra_ns, ca_ns) in holes_s:
            is_hole = True

    if is_hole:
        return 0.0

    # 3. Estimate V(s') for a safe cell
    # Find the goal position from either next_state or state
    rg, cg = -1, -1
    if rg_ns != -1:
        rg, cg = rg_ns, cg_ns
    elif rg_s != -1:
        rg, cg = rg_s, cg_s

    if rg == -1 or ra_ns == -1:
        return 0.0

    dist_g = abs(ra_ns - rg) + abs(ca_ns - cg)
    
    # Use a distance-based decay for the reward (Max steps = 30)
    # As distance to goal increases, the expected return decreases.
    val = max(0.0, 1.0 - (dist_g / 30.0))

    # 4. Risk Penalty: Reduce value if the agent is adjacent to a hole
    all_holes = holes_ns + holes_s
    if all_holes:
        min_dist_h = min(abs(ra_ns - hr) + abs(ca_ns - hc) for hr, hc in all_holes)
        if min_dist_h == 1:
            val *= 0.5
        elif min_dist_h == 2:
            val *= 0.8

    return float(val)