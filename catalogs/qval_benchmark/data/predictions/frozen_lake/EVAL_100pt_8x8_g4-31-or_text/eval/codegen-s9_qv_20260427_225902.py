def signal_function(state: str, action: str, next_state: str) -> float:
    """
    Estimates the Q-value for a given state, action, and next state in a Frozen Lake environment.
    Q(s, a) is approximated based on the distance from the agent's resulting position 
    to the goal, discounted by that distance, and considering terminal states.
    """
    
    def get_pos(grid_str, char):
        """Helper to find the (row, col) coordinates of a character in the ASCII grid."""
        lines = grid_str.strip().split('\n')
        for r, line in enumerate(lines):
            c = line.find(char)
            if c != -1:
                return r, c
        return None

    # Find positions of agent in next_state and goal in the current state
    agent_pos_next = get_pos(next_state, '@')
    goal_pos_state = get_pos(state, 'G')
    
    # If the agent position cannot be found in the next state, determine result
    if agent_pos_next is None:
        # If the goal disappeared from the grid in next_state, the agent likely reached it
        goal_pos_next = get_pos(next_state, 'G')
        if goal_pos_state is not None and goal_pos_next is None:
            return 1.0
        return 0.0

    # If no goal is found in the environment, it's impossible to win
    if goal_pos_state is None:
        return 0.0
        
    # Agent reached the goal (terminal reward)
    if agent_pos_next == goal_pos_state:
        return 1.0
        
    # Agent fell into a hole (terminal failure)
    # We check the state grid at the agent's next position to see if it was a hole 'H'
    lines_state = state.strip().split('\n')
    r_next, c_next = agent_pos_next
    if r_next < len(lines_state) and c_next < len(lines_state[0]):
        if lines_state[r_next][c_next] == 'H':
            return 0.0
    
    # For non-terminal states, the Q-value is approximated by the discounted distance to the goal.
    # Q(s, a) ≈ gamma^dist, where dist is the Manhattan distance.
    # A discount factor (gamma) of 0.9 is used to prioritize shorter paths.
    dist = abs(r_next - goal_pos_state[0]) + abs(c_next - goal_pos_state[1])
    
    # Maximum possible distance on 8x8 grid is 14. 
    # Step limit is 30, so we are well within the limit if a path exists.
    return 0.9 ** dist