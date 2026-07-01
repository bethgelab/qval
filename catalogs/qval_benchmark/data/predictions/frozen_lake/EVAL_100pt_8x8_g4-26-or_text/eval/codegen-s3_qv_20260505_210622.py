def signal_function(state: str, action: str, next_state: str) -> float:
    import math

    def get_grid(s):
        return [list(line.strip()) for line in s.strip().split('\n') if line.strip()]

    grid_s = get_grid(state)
    grid_ns = get_grid(next_state)
    if not grid_s or not grid_ns:
        return 0.0

    r_old, c_old = -1, -1
    r_new, c_new = -1, -1
    gr, gc = -1, -1

    # Locate the agent in both states and the goal in the current state
    for r in range(len(grid_s)):
        for c in range(len(grid_s[0])):
            char = grid_s[r][c]
            if char == '@':
                r_old, c_old = r, c
            elif char == 'G':
                gr, gc = r, c
        
        if r < len(grid_ns):
            for c in range(len(grid_ns[0])):
                if grid_ns[r][c] == '@':
                    r_new, c_new = r, c

    # If the goal was not explicitly visible (e.g., replaced by '@' in state), 
    # attempt to find it in the next_state.
    if gr == -1:
        for r in range(len(grid_ns)):
            for c in range(len(grid_ns[0])):
                if grid_ns[r][c] == 'G':
                    gr, gc = r, c
    
    # If essential features are missing, we cannot reliably estimate.
    if r_new == -1 or gr == -1:
        return 0.0

    # Determine the effective position and the nature of the tile.
    # In Frozen Lake, the character '@' in the next_state replaces the tile.
    # We use the previous state's grid to identify if the move landed on a hole or goal.
    if r_new == r_old and c_new == c_old:
        # The agent hit a wall and stayed in place. 
        # Since it was not in a hole or goal in 'state', it remains on a safe tile.
        if r_old == gr and c_old == gc:
            return 1.0
        r_eff, c_eff = r_old, c_old
    else:
        # The agent moved to a new coordinate. 
        # Check what that coordinate contained in the original grid.
        if 0 <= r_new < len(grid_s) and 0 <= c_new < len(grid_s[0]):
            tile_at_new_pos = grid_s[r_new][c_new]
            if tile_at_new_pos == 'G':
                return 1.0
            if tile_at_new_pos == 'H':
                return 0.0
            # If it was '.', 'S', or even '@' (though @ shouldn't be at a new pos), it's safe.
            r_eff, c_eff = r_new, c_new
        else:
            # Fallback (should not be triggered by valid env dynamics)
            r_eff, c_eff = r_old, c_old

    # If the effective position is the goal, return max reward.
    if r_eff == gr and c_eff == gc:
        return 1.0

    # Return an approximation of the Q-value using a potential-based distance decay.
    # A decay factor of 0.95 appropriately models the value of approaching the goal.
    dist = abs(r_eff - gr) + abs(c_eff - gc)
    return math.pow(0.95, dist)