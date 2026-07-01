def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value Q(s, a) for the Frozen Lake environment.
    The Q-value is approximated based on the Manhattan distance to the goal 
    and whether the action leads to a terminal state (goal or hole).
    """
    def parse_grid(grid_str: str):
        lines = [line.strip() for line in grid_str.strip().split('\n') if line.strip()]
        grid = []
        for line in lines:
            # Handle grids that may be space-separated or condensed
            row = line.split()
            if len(row) < 8:
                # Remove all spaces and treat each character as a cell
                row = list(line.replace(' ', ''))
            grid.append(row)
        return grid

    def find_char(grid, char):
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c] == char:
                    return (r, c)
        return None

    # Parse current state and next state
    grid_s = parse_grid(state)
    grid_ns = parse_grid(next_state)
    
    # Identify key positions
    # Goal position is constant for a given map; get it from the current state
    goal_pos = find_char(grid_s, 'G')
    # Agent position in the next state determines the outcome of the action
    agent_pos_next = find_char(grid_ns, '@')
    
    if not goal_pos or not agent_pos_next:
        return 0.0
    
    # 1. Check if the agent reached the goal
    if agent_pos_next == goal_pos:
        return 1.0
        
    # 2. Check if the agent fell into a hole
    # We check the cell in the original state grid because the '@' symbol 
    # in next_state covers the 'H' symbol.
    r, c = agent_pos_next
    if r < len(grid_s) and c < len(grid_s[r]) and grid_s[r][c] == 'H':
        return 0.0
        
    # 3. Distance-based value estimation
    # Q(s, a) = R + gamma * V(s'). Since R=0 for non-terminal states, 
    # we estimate the value of the next state based on proximity to the goal.
    # A common proxy for V(s) in grid worlds is gamma^distance.
    dist = abs(r - goal_pos[0]) + abs(c - goal_pos[1])
    
    # Using a discount factor gamma = 0.9 to value closer states more highly
    return 0.9 ** dist