def signal_function(state: str, action: str, next_state: str) -> float:
    def parse_grid(grid_str):
        lines = [line for line in grid_str.strip().split('\n') if line.strip()]
        if not lines:
            return []
        # Handle 8x8 string without newlines or grid with newlines
        if len(lines) == 1 and len(lines[0]) == 64:
            grid = []
            for i in range(0, 64, 8):
                grid.append(list(lines[0][i:i+8]))
        else:
            grid = [list(line) for line in lines]
        return grid

    grid_s = parse_grid(state)
    grid_ns = parse_grid(next_state)

    if not grid_s or not grid_ns:
        return 0.0

    # Find G and H locations from the original state grid
    # The state grid contains the markers for Goal and Holes before the action
    pos_G = None
    pos_Hs = []
    for r in range(len(grid_s)):
        for c in range(len(grid_s[r])):
            char = grid_s[r][c]
            if char == 'G':
                pos_G = (r, c)
            elif char == 'H':
                pos_Hs.append((r, c))

    # Find the agent's new position in the next state
    pos_at = None
    for r in range(len(grid_ns)):
        for c in range(len(grid_ns[r])):
            if grid_ns[r][c] == '@':
                pos_at = (r, c)
                break
        if pos_at:
            break

    if not pos_at:
        return 0.0

    # If the action leads directly to the goal or a hole
    if pos_G and pos_at == pos_G:
        return 1.0
    if pos_Hs and pos_at in pos_Hs:
        return 0.0

    # If the goal is not identifiable, assume low value
    if not pos_G:
        return 0.0

    # Calculate Manhattan distance to goal and nearest hole
    d_G = abs(pos_at[0] - pos_G[0]) + abs(pos_at[1] - pos_G[1])
    
    d_H = 10.0  # Default large distance if no holes exist
    if pos_Hs:
        d_H = min(abs(pos_at[0] - h[0]) + abs(pos_at[1] - h[1]) for h in pos_Hs)

    # Q-value estimation:
    # 1. Base value is roughly the inverse of distance to the goal.
    # 2. A safety factor reduces the value if the agent is adjacent to a hole.
    # As d_H increases, the safety factor approaches 1.0.
    # As d_G increases, the Q-value decreases.
    
    q_base = 1.0 / (1.0 + d_G)
    safety_factor = 1.0 - (1.0 / (d_H + 1.0))
    
    # Ensure safety_factor doesn't drop too low if d_H is at least 1
    # If d_H = 1, factor is 0.5; if d_H = 2, factor is 0.66; etc.
    
    return float(q_base * safety_factor)