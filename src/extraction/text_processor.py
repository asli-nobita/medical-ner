"""Text processing utilities for medical document extraction."""


def group_tokens_into_lines(tokens, line_tolerance=10): 
    """
    Groups tokens into lines based on y-coordinate proximity.
    
    Args:
        tokens (list): List of token dictionaries with position data
        line_tolerance (int): Maximum vertical distance to consider tokens on same line
        
    Returns:
        list: List of lines, each containing tokens sorted by x-coordinate
    """ 
    if not tokens: 
        return [] 
    # sort tokens by top coordinate 
    sorted_tokens = sorted(tokens, key=lambda t: t['top']) 
    lines = [] 
    current_line = [sorted_tokens[0]] 
    current_y = sorted_tokens[0]['top'] 
    
    for token in sorted_tokens[1:]: 
        if abs(token['top'] - current_y) <= line_tolerance: 
            current_line.append(token) 
        else: 
            # start new line 
            # sort current line by left coordinate 
            current_line.sort(key=lambda t:t['left']) 
            lines.append(current_line) 
            current_line = [token] 
            current_y = token['top'] 
    if current_line: 
        current_line.sort(key=lambda t:t['left']) 
        lines.append(current_line) 
    
    return lines 


def extract_line_text(line_tokens): 
    """
    Extracts combined text from a line of tokens.
    
    Args:
        line_tokens (list): List of tokens in a line
        
    Returns:
        str: Combined text with appropriate spacing
    """ 
    if not line_tokens: 
        return [] 
    text_parts = [] 
    prev_token = None 
    
    for token in line_tokens: 
        if prev_token: 
            gap = token['left'] - (prev_token['left'] + prev_token['width']) 
            if gap > 10: 
                text_parts.append(' ') 
        text_parts.append(token['text']) 
        prev_token = token 
        
    return ''.join(text_parts)