import re
import math

def signal_function(state: str, action: str, next_state: str) -> float:
    # Base Q-value estimation for TerminalBench shell tasks.
    # Heuristic: Favor states that look like they are progressing towards a goal
    # (e.g., containing 'Success', 'Done', 'root', 'sudo', or specific file creations)
    # and penalize error states ('Error', 'Failed', 'Permission denied').
    
    # Define positive and negative keywords based on typical terminal task outcomes
    success_indicators = ['success', 'done', 'root@', 'created', 'wrote', 'exported', 'generated', 'passed', 'true', '1', 'ok', 'exit status 0']
    error_indicators = ['error', 'failed', 'denied', 'permission', 'no such file', 'syntax error', 'usage:', 'false', '0', 'exit status 1', 'fail', 'broken']
    
    def score_text(text):
        score = 0.0
        text_lower = text.lower()
        
        # Check for success indicators
        for indicator in success_indicators:
            if indicator in text_lower:
                score += 1.0
        
        # Check for error indicators
        for indicator in error_indicators:
            if indicator in text_lower:
                score -= 1.5
        
        # Heuristic: If the state is very short and doesn't contain errors, it might be a start state
        # If it's very long, it might be a verbose output or a successful log.
        # We assume intermediate states with some content are better than empty ones.
        if len(text) > 0 and len(text) < 50:
             score += 0.1
        
        return score

    state_score = score_text(state)
    next_state_score = score_text(next_state)
    
    # The action itself doesn't change the fundamental state quality much in this context,
    # but we can assume taking an action that leads to a better state is good.
    # Q(s,a) approx R(s,a) + gamma * V(s')
    # Here we approximate V(s) roughly by the score of the state text.
    # Immediate reward R(s,a) is hard to know, but we can infer it from the delta.
    
    # If the next state is significantly better than the current state, the action was good.
    # If the next state is worse, the action was bad.
    # We add a small base value if the state looks promising (e.g., contains 'root' or 'sudo' implies privilege which is often needed).
    
    base_value = 0.0
    if 'root' in state.lower() or 'sudo' in state.lower():
        base_value = 0.5
    
    # Estimate immediate reward based on improvement
    improvement = next_state_score - state_score
    
    # Discount factor gamma
    gamma = 0.9
    
    # Estimated Q-value
    # Q(s,a) ~ (improvement) + gamma * V(next_state) + base_context
    # We scale the improvement to be within a reasonable range [-1, 1]
    q_value = (improvement * 0.8) + (gamma * next_state_score) + base_value
    
    # Clamp the value to a reasonable range for stability
    return max(-2.0, min(2.0, q_value))