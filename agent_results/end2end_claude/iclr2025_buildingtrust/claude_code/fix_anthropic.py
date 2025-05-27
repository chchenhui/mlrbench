"""
Fix for Anthropic API.

This script provides compatibility fixes for the Anthropic Python SDK.
It's needed because the Anthropic API changed how messages are accessed in their SDK.
"""

async def fix_anthropic_response(message):
    """
    Extract text content from an Anthropic message response.
    
    This function handles different versions of the Anthropic SDK
    and provides consistent access to the message content.
    
    Args:
        message: The message response from Anthropic API
        
    Returns:
        The text content of the message
    """
    # For newer versions of the SDK, content is a list of blocks
    if hasattr(message, 'content') and isinstance(message.content, list):
        # Extract text from the first text block
        for block in message.content:
            if hasattr(block, 'text') and block.text:
                return block.text
            elif hasattr(block, 'type') and block.type == 'text':
                return block.text
                
    # Older versions had a direct .content attribute
    elif hasattr(message, 'content') and isinstance(message.content, str):
        return message.content
        
    # Another possible structure
    elif hasattr(message, 'completion'):
        return message.completion
        
    # Handle dictionary responses
    elif isinstance(message, dict):
        if 'content' in message:
            content = message['content']
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        return block['text']
            else:
                return content
        elif 'completion' in message:
            return message['completion']
    
    # If we can't find content in any expected format, convert the whole object to string
    return str(message)