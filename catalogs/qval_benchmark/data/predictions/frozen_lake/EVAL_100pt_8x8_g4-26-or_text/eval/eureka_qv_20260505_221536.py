import collections
import math

def signal_function(state: str, action: str, next_state: str):
    """
    Estimates the Q-value Q(s, a) for a Frozen Lake grid.
    Uses a geometric decay for the distance-to-goal and a multiplicative 
    safety multiplier based on the distance to the nearest hole.
    """
    s_lines = [line.strip() for line in state.strip().split('\n') if line.strip()]
    ns_lines = [line.strip() for line in next_state.strip().split('\n') if line.strip()]
    
    if not ns_lines:
        return 0.0

    rows = len(ns_lines)
    cols = len(ns_lines[0]) if rows > 0 else 0
    
    goal_pos = None
    holes = set()
    agent_new = None

    # Parse the next state and initial state to identify the agent, goal, and holes.
    # We prioritize next_state for the agent's new position.
    for r in range(rows):
        # Parse next_state
        if r < len(ns_lines):
            line = ns_lines[r]
            for c, char in enumerate(line):
                if char == 'G':
                    goal_pos = (r, c)
                elif char == 'H':
                    holes.add((r, c))
                elif char == '@':
                    agent_new = (r, c)
        
        # Parse state to ensure goal and holes are fully identified
        if r < len(s_lines):
            line = s_lines[r]
            for c, char in enumerate(line):
                if c < cols:
                    if char == 'G' and goal_pos is None:
                        goal_pos = (r, c)
                    elif char == 'H' and (r, c) not in holes:
                        holes.add((r, c))
                    
    if agent_new is None or goal_pos is None:
        return 0.0
    
    # Terminal conditions: agent reaches goal or falls into a hole.
    if agent_new in holes:
        return 0.0
    if agent_new == goal_pos:
        return 1.0
        
    # BFS 1: Find the shortest safe path distance to the goal (avoiding holes).
    queue_g = collections.deque([(agent_new, 0)])
    visited_g = {agent_new}
    dist_to_goal = -1
    
    while queue_g:
        (curr_r, curr_c), d = queue_g.popleft()
        if (curr_r, curr_c) == goal_pos:
            dist_to_goal = d
            break
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in holes and (nr, nc) not in visited_g:
                visited_g.add((nr, nc))
                queue_g.append(((nr, nc), d + 1))
                    
    # If the goal is unreachable via a safe path, the Q-value is 0.
    if dist_to_goal == -1:
        return 0.0
        
    # BFS 2: Find the distance to the nearest hole.
    queue_h = collections.deque([(agent_new, 0)])
    visited_h = {agent_new}
    dist_to_nearest_hole = float('inf')
    
    while queue_h:
        (curr_r, curr_c), d = queue_h.popleft()
        if (curr_r, curr_c) in holes:
            dist_to_nearest_hole = d
            break
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited_h:
                visited_h.add((nr, nc))
                queue_h.append(((nr, nc), d + 1))

    # 1. Distance Reward: Uses a geometric decay (0.98^d) to model discounted returns.
    dist_reward = math.pow(0.98, dist_to_goal)
    
    # 2. Safety Multiplier: Multiplicative approach to reduce Q-value when near hazards.
    # A distance of 1 from a hole results in a 0.7 multiplier.
    if dist_to_nearest_hole == float('inf'):
        safety_multiplier = 1.0
    else:
        safety_multiplier = 1.0 - 0.3 * math.exp(-(dist_to_nearest_hole - 1))
    
    total_q = dist_reward * safety_multiplier
    
    return total_q, {
        "dist_reward": dist_reward,
        "safety_multiplier": safety_multiplier,
        "dist_to_goal": float(dist_to_goal),
        "dist_to_hole": float(dist_to_nearest_hole) if dist_to_nearest_hole != float('inf') else 0.0
    }