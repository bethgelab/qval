import re
import json
import collections
import math
import string

def signal_function(state: str, action: str, next_state: str) -> float:
    # Heuristic Q-value estimation based on state analysis
    
    # Extract key features from state representations
    state_lower = state.lower()
    next_state_lower = next_state.lower()
    action_lower = action.lower()
    
    # Define success indicators that suggest goal completion
    success_keywords = ['saved', 'added', 'created', 'sent', 'completed', 'done', 'success', 'ok', 'check', 'verify', 'achieved', 'goal', 'target']
    progress_keywords = ['progress', 'step', 'remaining', 'left', 'current', 'active', 'selected', 'highlighted', 'focused']
    error_keywords = ['error', 'failed', 'invalid', 'missing', 'cannot', 'unable', 'not allowed', 'blocked', 'disabled']
    navigation_keywords = ['back', 'previous', 'next', 'home', 'main', 'dashboard', 'menu', 'sidebar', 'navigation', 'link']
    form_keywords = ['input', 'text', 'button', 'submit', 'fill', 'select', 'dropdown', 'checkbox', 'radio', 'date', 'time']
    
    # Count occurrences of key features
    success_count = sum(1 for kw in success_keywords if kw in state_lower)
    success_count_next = sum(1 for kw in success_keywords if kw in next_state_lower)
    
    progress_count = sum(1 for kw in progress_keywords if kw in state_lower)
    progress_count_next = sum(1 for kw in progress_keywords if kw in next_state_lower)
    
    error_count = sum(1 for kw in error_keywords if kw in state_lower)
    error_count_next = sum(1 for kw in error_keywords if kw in next_state_lower)
    
    # Check if action appears to be a valid interaction
    action_type = 'click' if 'click' in action_lower else 'fill' if 'fill' in action_lower else 'press' if 'press' in action_lower else 'scroll' if 'scroll' in action_lower else 'noop'
    
    # Check for bid numbers in state (indicates interactive elements)
    bid_pattern = r'\bid\d+\b'
    bids_in_state = len(re.findall(bid_pattern, state))
    bids_in_next = len(re.findall(bid_pattern, next_state))
    
    # Calculate feature-based scores
    # Success improvement: higher if next state has more success indicators
    success_improvement = (success_count_next - success_count) / max(1, success_count + 1)
    
    # Error reduction: higher if next state has fewer errors
    error_reduction = max(0, error_count - error_count_next) / max(1, error_count + 1)
    
    # Progress change: higher if progress indicators increase
    progress_change = (progress_count_next - progress_count) / max(1, progress_count + 1)
    
    # Bid reduction heuristic: fewer interactive elements might indicate completion
    bid_change = max(0, bids_in_state - bids_in_next) / max(1, bids_in_state + 1)
    
    # Action validity heuristic: check if action mentions specific bid that exists
    action_bid = re.search(r"click\(['\"]?\s*bid(\d+)", action) or re.search(r"fill\(['\"]?\s*bid(\d+)", action) or re.search(r"press\(['\"]?\s*bid(\d+)", action)
    if action_bid:
        action_bid_num = int(action_bid.group(1))
        # Check if this bid number exists in state
        bid_exists = f"bid{action_bid_num}" in state_lower
        action_valid = 1.0 if bid_exists else 0.0
    else:
        action_valid = 0.8  # Default validity for other actions
    
    # Calculate combined Q-value estimate
    # Weighted combination of heuristics
    q_success = 0.3 * success_improvement
    q_error = 0.2 * error_reduction
    q_progress = 0.2 * progress_change
    q_bids = 0.15 * bid_change
    q_action = 0.15 * action_valid
    
    # Normalize and combine
    base_q = q_success + q_error + q_progress + q_bids + q_action
    
    # Apply decay for potential issues
    if error_count_next > 0:
        base_q *= 0.5
    if error_count > 0 and error_count_next == 0:
        base_q *= 1.2
    
    # Clamp to valid range
    q_value = max(0.0, min(1.0, base_q))
    
    # Boost for clear success indicators in next state
    if success_count_next > 2:
        q_value = min(1.0, q_value + 0.1)
    
    # Slight penalty for noop actions unless near goal
    if action_type == 'noop':
        q_value *= 0.8
    
    return round(q_value, 4)