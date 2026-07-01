import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Helper to find position of a character in the grid
    def find_char(grid_str, char):
        lines = grid_str.split('\n')
        # Filter out empty lines if any (e.g. trailing newline)
        lines = [l for l in lines if l]
        for r, line in enumerate(lines):
            for c, cell in enumerate(line):
                if cell == char:
                    return (r, c)
        return None

    # Parse state to find initial agent and goal positions
    # Goal position is static, so finding it in state is sufficient
    goal_pos = find_char(state, 'G')
    agent_pos = find_char(state, '@')
    
    # Parse next_state to find new agent position
    next_agent_pos = find_char(next_state, '@')
    
    # If agent is still visible in next_state
    if next_agent_pos is not None:
        # Check if agent is on a hole
        # We check the character at the agent's position in next_state
        # Note: next_state might have '@' at the position, but we need to know if that cell was a hole
        # However, if '@' is present, it usually overwrites the cell content.
        # But if it fell in a hole, '@' might be removed. 
        # Let's check if the position corresponds to a hole in the static map (state)
        # Actually, if next_agent_pos is found, the episode is likely not over (unless @ remains on G/H)
        
        # Check if agent reached goal (pos matches goal_pos)
        if goal_pos and next_agent_pos == goal_pos:
            return 1.0
            
        # Check if agent is on a hole (pos matches hole_pos in state)
        # We need to find hole positions in state to know if current pos is a hole
        # But we can just check if next_state has 'H' at next_agent_pos? 
        # If '@' is there, 'H' is likely overwritten.
        # So we check state's hole map.
        # To do this efficiently without full scan, we can assume static map holes.
        # Let's find 'H' in state.
        # Note: There could be multiple holes.
        # We check if next_agent_pos is one of the hole positions.
        # Since we can't easily store all holes without iterating, let's iterate state once.
        # But we need to do this efficiently.
        # Let's just check if next_agent_pos is a hole by looking at state.
        # We need a function to check if a pos is a hole.
        # We can find all holes in state.
        holes = []
        if state:
            lines = state.split('\n')
            lines = [l for l in lines if l]
            for r, line in enumerate(lines):
                for c, cell in enumerate(line):
                    if cell == 'H':
                        holes.append((r, c))
        
        if next_agent_pos in holes:
            return 0.0
            
        # Calculate Manhattan distance to goal
        if goal_pos:
            dist = abs(next_agent_pos[0] - goal_pos[0]) + abs(next_agent_pos[1] - goal_pos[1])
            # Exponential decay for Q-value estimation
            # 0.9 discount per step approx
            return max(0.0, 0.9 ** dist)
        else:
            return 0.0

    else:
        # Agent is not visible in next_state (episode likely ended)
        # Check if agent was at Goal or Hole
        if agent_pos is None:
            return 0.0
            
        # Check character at agent_pos in next_state
        # We need to reconstruct the line in next_state
        lines = next_state.split('\n')
        lines = [l for l in lines if l]
        if 0 <= agent_pos[0] < len(lines) and 0 <= agent_pos[1] < len(lines[agent_pos[0]]):
            cell_char = lines[agent_pos[0]][agent_pos[1]]
            if cell_char == 'G':
                return 1.0
            elif cell_char == 'H':
                return 0.0
            else:
                # Agent disappeared but not at G or H? (Should not happen based on rules)
                return 0.0
        else:
            return 0.0