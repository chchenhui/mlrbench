"""
String Manipulation Library
A collection of functions for common string operations.
"""

def reverse_string(s):
    """
    Reverses a string.
    
    Args:
        s (str): The input string to reverse
        
    Returns:
        str: The reversed string
    """
    return s[::-1]

def is_palindrome(s):
    """
    Checks if a string is a palindrome (reads the same forward and backward).
    
    Args:
        s (str): The input string to check
        
    Returns:
        bool: True if the string is a palindrome, False otherwise
    """
    # Remove spaces and convert to lowercase for case-insensitive check
    s = s.lower()
    return s == s[::-1]

def count_substring(s, sub):
    """
    Counts occurrences of a substring in a string.
    
    Args:
        s (str): The main string to search in
        sub (str): The substring to count
        
    Returns:
        int: The number of occurrences of sub in s
    """
    count = 0
    start = 0
    
    while True:
        start = s.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count

def capitalize_words(s):
    """
    Capitalizes the first letter of each word in a string.
    
    Args:
        s (str): The input string
        
    Returns:
        str: String with each word capitalized
    """
    return ' '.join(word.capitalize() for word in s.split())