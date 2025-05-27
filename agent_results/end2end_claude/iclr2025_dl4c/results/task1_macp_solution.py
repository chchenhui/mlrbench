"""
String Manipulation Library

A collection of utility functions for common string manipulation operations.
Implements a clean, modular design with comprehensive documentation and error handling.
"""

class StringManipulator:
    """A class providing various string manipulation operations."""
    
    @staticmethod
    def reverse_string(s):
        """
        Reverses a string.
        
        Args:
            s (str): The input string to reverse
            
        Returns:
            str: The reversed string
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string")
        return s[::-1]
    
    @staticmethod
    def is_palindrome(s):
        """
        Checks if a string is a palindrome (reads the same forward and backward).
        
        Args:
            s (str): The input string to check
            
        Returns:
            bool: True if the string is a palindrome, False otherwise
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string")
        # Clean the string by removing spaces and converting to lowercase
        clean_s = s.lower()
        return clean_s == clean_s[::-1]
    
    @staticmethod
    def count_substring(s, sub):
        """
        Counts occurrences of a substring in a string.
        
        Args:
            s (str): The main string to search in
            sub (str): The substring to count
            
        Returns:
            int: The number of occurrences of sub in s
            
        Raises:
            TypeError: If either input is not a string
        """
        if not isinstance(s, str) or not isinstance(sub, str):
            raise TypeError("Both inputs must be strings")
        if not sub:
            return 0
        
        count = 0
        pos = 0
        
        while True:
            pos = s.find(sub, pos)
            if pos == -1:
                break
            count += 1
            pos += 1
            
        return count
    
    @staticmethod
    def capitalize_words(s):
        """
        Capitalizes the first letter of each word in a string.
        
        Args:
            s (str): The input string
            
        Returns:
            str: String with each word capitalized
            
        Raises:
            TypeError: If input is not a string
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string")
        return ' '.join(word.capitalize() for word in s.split())


# For backward compatibility, providing function interfaces
def reverse_string(s):
    """Function interface to StringManipulator.reverse_string"""
    return StringManipulator.reverse_string(s)

def is_palindrome(s):
    """Function interface to StringManipulator.is_palindrome"""
    return StringManipulator.is_palindrome(s)

def count_substring(s, sub):
    """Function interface to StringManipulator.count_substring"""
    return StringManipulator.count_substring(s, sub)

def capitalize_words(s):
    """Function interface to StringManipulator.capitalize_words"""
    return StringManipulator.capitalize_words(s)