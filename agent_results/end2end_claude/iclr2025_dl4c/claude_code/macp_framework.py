"""
Multi-Agent Collaborative Programming (MACP) framework with role-specialized agents.
"""

import os
import time
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
import anthropic
from collections import deque
import threading

class Agent:
    """Base class for role-specialized agents."""
    
    def __init__(
        self, 
        role: str, 
        model_name: str = "claude-3-7-sonnet-20250219",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize an agent with a specific role.
        
        Args:
            role: The agent's role (architect, implementer, tester, reviewer)
            model_name: Name of the LLM to use
            logger: Logger instance for tracking events
        """
        self.role = role
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize the anthropic client
        self.client = anthropic.Anthropic()
        
        # Initialize memory
        self.short_term_memory = []  # Recent messages and context
        self.knowledge = self._initialize_role_knowledge()
        
        # Message history
        self.message_history = []
    
    def _initialize_role_knowledge(self) -> Dict[str, Any]:
        """Initialize role-specific knowledge and expertise."""
        knowledge = {
            "role_description": self._get_role_description(),
            "responsibilities": self._get_role_responsibilities(),
            "best_practices": self._get_role_best_practices()
        }
        return knowledge
    
    def _get_role_description(self) -> str:
        """Get a description of the agent's role."""
        role_descriptions = {
            "architect": "System architect responsible for high-level design and component decomposition.",
            "implementer": "Developer responsible for translating designs into functional code.",
            "tester": "Quality assurance specialist responsible for creating test cases and verifying functionality.",
            "reviewer": "Code reviewer responsible for evaluating code quality and suggesting improvements.",
            "moderator": "Team coordinator responsible for managing workflow and resolving conflicts."
        }
        return role_descriptions.get(self.role, "Unknown role")
    
    def _get_role_responsibilities(self) -> List[str]:
        """Get a list of the agent's responsibilities."""
        role_responsibilities = {
            "architect": [
                "Create high-level system design",
                "Define component interfaces",
                "Make architecture decisions",
                "Ensure system meets requirements",
                "Create technical specifications"
            ],
            "implementer": [
                "Translate designs into code",
                "Implement algorithms and logic",
                "Write clean, maintainable code",
                "Add appropriate documentation",
                "Follow coding standards"
            ],
            "tester": [
                "Design test cases",
                "Identify edge cases",
                "Create unit and integration tests",
                "Verify code against requirements",
                "Report bugs and issues"
            ],
            "reviewer": [
                "Evaluate code quality",
                "Identify potential issues",
                "Suggest improvements",
                "Ensure adherence to standards",
                "Approve or reject code changes"
            ],
            "moderator": [
                "Coordinate team activities",
                "Track progress and tasks",
                "Facilitate communication",
                "Resolve conflicts",
                "Ensure project goals are met"
            ]
        }
        return role_responsibilities.get(self.role, [])
    
    def _get_role_best_practices(self) -> List[str]:
        """Get a list of best practices for the agent's role."""
        role_best_practices = {
            "architect": [
                "Follow SOLID principles",
                "Consider modularity and separation of concerns",
                "Design for extensibility and maintainability",
                "Document design decisions and rationale",
                "Consider performance, security, and scalability"
            ],
            "implementer": [
                "Write clear, self-documenting code",
                "Follow consistent naming conventions",
                "Minimize code duplication",
                "Handle edge cases and errors",
                "Write unit tests for your code"
            ],
            "tester": [
                "Test both normal and edge cases",
                "Write clear test descriptions",
                "Ensure good test coverage",
                "Make tests repeatable and deterministic",
                "Consider both positive and negative test cases"
            ],
            "reviewer": [
                "Focus on code quality and readability",
                "Look for potential bugs and edge cases",
                "Check for adherence to project standards",
                "Provide constructive feedback",
                "Verify test coverage"
            ],
            "moderator": [
                "Keep discussions focused and productive",
                "Ensure all team members are heard",
                "Document decisions and action items",
                "Track progress and identify blockers",
                "Balance team autonomy with coordination"
            ]
        }
        return role_best_practices.get(self.role, [])
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The incoming message
            
        Returns:
            The agent's response message
        """
        # Update short-term memory with this message
        self.short_term_memory.append(message)
        if len(self.short_term_memory) > 10:  # Keep only recent messages
            self.short_term_memory.pop(0)
        
        # Log the incoming message
        self.logger.info(f"{self.role} received message from {message.get('sender', 'unknown')}")
        
        # Generate response using LLM
        prompt = self._construct_prompt(message)
        response_content = self._get_llm_response(prompt)
        
        # Create response message
        response = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "sender": self.role,
            "receiver": message.get("sender"),
            "content": response_content,
            "reference_id": message.get("id"),
            "message_type": self._determine_message_type(response_content)
        }
        
        # Add to message history
        self.message_history.append({
            "sent": False,
            "message": message
        })
        self.message_history.append({
            "sent": True,
            "message": response
        })
        
        return response
    
    def _construct_prompt(self, message: Dict[str, Any]) -> str:
        """
        Construct a prompt for the LLM based on the message and role.
        
        Args:
            message: The incoming message
            
        Returns:
            The constructed prompt string
        """
        # Base prompt includes role information
        prompt = f"""
You are an AI agent with the role of {self.role.upper()} in a software development team.

## Your Role Description
{self._get_role_description()}

## Your Responsibilities
{', '.join(self._get_role_responsibilities())}

## Best Practices for Your Role
{', '.join(self._get_role_best_practices())}

## Recent Conversation History
"""
        
        # Add conversation history
        for hist_message in self.short_term_memory[-5:]:  # Last 5 messages
            sender = hist_message.get("sender", "Unknown")
            content = hist_message.get("content", "")
            prompt += f"\n{sender.upper()}: {content}\n"
        
        # Add current message
        prompt += f"\n## Current Message from {message.get('sender', 'Unknown').upper()}\n{message.get('content', '')}\n"
        
        # Add response instructions
        prompt += f"""
## Instructions
Please respond to this message based on your role as the {self.role.upper()}.
Focus on your specific responsibilities and expertise.
Provide a clear, helpful response that moves the development process forward.
"""
        
        return prompt
    
    def _get_llm_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        try:
            # Send the prompt to the model
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.5,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Return the response content
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Error getting response from model: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _determine_message_type(self, content: str) -> str:
        """
        Determine the type of message based on content.
        
        Args:
            content: The message content
            
        Returns:
            The message type
        """
        content_lower = content.lower()
        
        if "question" in content_lower or "?" in content:
            return "information_request"
        elif "approve" in content_lower or "approved" in content_lower:
            return "approval"
        elif "reject" in content_lower or "change" in content_lower:
            return "rejection"
        elif "status" in content_lower or "progress" in content_lower:
            return "status_update"
        elif "feedback" in content_lower or "suggestion" in content_lower:
            return "feedback"
        else:
            return "general"
    
    def create_message(self, receiver: str, content: str, message_type: str = "general") -> Dict[str, Any]:
        """
        Create a new message to send.
        
        Args:
            receiver: The recipient of the message
            content: The message content
            message_type: The type of message
            
        Returns:
            The created message
        """
        message = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "sender": self.role,
            "receiver": receiver,
            "content": content,
            "message_type": message_type
        }
        
        # Add to message history
        self.message_history.append({
            "sent": True,
            "message": message
        })
        
        return message


class ArchitectAgent(Agent):
    """Architect agent responsible for high-level design."""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        super().__init__("architect", model_name, logger)
        self.design_document = ""
        
    def create_design(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a design document for a programming task.
        
        Args:
            task: The programming task description
            
        Returns:
            A message containing the design document
        """
        prompt = f"""
You are an AI architect responsible for creating software designs.

# Task Description
{task['description']}

# Requirements
{', '.join(task['requirements'])}

# Expected Output
{json.dumps(task['expected_output'], indent=2)}

# Instructions
Please create a comprehensive design document for this programming task.
Your design document should include:

1. High-level architecture overview
2. Component breakdown with clear responsibilities
3. Interface definitions
4. Data structures and algorithms
5. Error handling approach
6. Considerations for extensibility and maintainability

Focus on creating a clean, modular design that addresses all requirements.
"""
        
        self.logger.info(f"Architect creating design for task: {task['name']}")
        
        # Get design from LLM
        self.design_document = self._get_llm_response(prompt)
        
        # Create message with design document
        message = self.create_message(
            receiver="all",
            content=self.design_document,
            message_type="design_document"
        )
        
        self.logger.info("Architect completed design document")
        
        return message


class ImplementerAgent(Agent):
    """Implementer agent responsible for translating designs into code."""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        super().__init__("implementer", model_name, logger)
        self.current_code = ""
        
    def implement_code(self, design_document: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement code based on a design document.
        
        Args:
            design_document: The design document
            task: The programming task description
            
        Returns:
            A message containing the implemented code
        """
        prompt = f"""
You are an AI implementer responsible for translating designs into functional code.

# Task Description
{task['description']}

# Requirements
{', '.join(task['requirements'])}

# Design Document
{design_document}

# Expected Output
{json.dumps(task['expected_output'], indent=2)}

# Instructions
Please implement Python code based on the provided design document.
Your implementation should:

1. Follow the architecture and component structure defined in the design
2. Implement all required functionality
3. Include clear comments explaining your code
4. Follow good coding practices and standards
5. Be complete and ready for testing

Return ONLY the Python code implementation. Start with ```python and end with ```.
"""
        
        self.logger.info(f"Implementer creating code for task: {task['name']}")
        
        # Get implementation from LLM
        response = self._get_llm_response(prompt)
        
        # Extract code between triple backticks
        code_blocks = response.split("```")
        if len(code_blocks) >= 3:
            # Get the content of the first code block
            implementation = code_blocks[1]
            # Remove language identifier if present
            if implementation.startswith("python"):
                implementation = implementation[len("python"):].strip()
            else:
                implementation = implementation.strip()
        else:
            # If no code blocks found, use entire response
            implementation = response
        
        self.current_code = implementation
        
        # Create message with implementation
        message = self.create_message(
            receiver="all",
            content=f"Here is my implementation based on the design:\n\n```python\n{self.current_code}\n```",
            message_type="implementation"
        )
        
        self.logger.info("Implementer completed code implementation")
        
        return message
    
    def update_code(self, feedback: str) -> Dict[str, Any]:
        """
        Update code based on feedback.
        
        Args:
            feedback: Feedback on the current implementation
            
        Returns:
            A message containing the updated code
        """
        prompt = f"""
You are an AI implementer responsible for updating code based on feedback.

# Current Code
```python
{self.current_code}
```

# Feedback
{feedback}

# Instructions
Please update the code based on the feedback provided.
Make specific changes to address all the points in the feedback.
Return ONLY the updated Python code. Start with ```python and end with ```.
"""
        
        self.logger.info("Implementer updating code based on feedback")
        
        # Get updated implementation from LLM
        response = self._get_llm_response(prompt)
        
        # Extract code between triple backticks
        code_blocks = response.split("```")
        if len(code_blocks) >= 3:
            # Get the content of the first code block
            updated_code = code_blocks[1]
            # Remove language identifier if present
            if updated_code.startswith("python"):
                updated_code = updated_code[len("python"):].strip()
            else:
                updated_code = updated_code.strip()
        else:
            # If no code blocks found, use entire response
            updated_code = response
        
        self.current_code = updated_code
        
        # Create message with updated implementation
        message = self.create_message(
            receiver="all",
            content=f"Here is my updated implementation based on the feedback:\n\n```python\n{self.current_code}\n```",
            message_type="implementation_update"
        )
        
        self.logger.info("Implementer completed code update")
        
        return message


class TesterAgent(Agent):
    """Tester agent responsible for creating tests and verifying functionality."""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        super().__init__("tester", model_name, logger)
        self.test_code = ""
        self.test_results = {}
        
    def create_tests(self, design_document: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create test cases based on a design document and task requirements.
        
        Args:
            design_document: The design document
            task: The programming task description
            
        Returns:
            A message containing the test cases
        """
        prompt = f"""
You are an AI tester responsible for creating test cases.

# Task Description
{task['description']}

# Requirements
{', '.join(task['requirements'])}

# Design Document
{design_document}

# Expected Output
{json.dumps(task['expected_output'], indent=2)}

# Test Cases from Requirements
{json.dumps(task.get('test_cases', []), indent=2)}

# Instructions
Please create comprehensive test cases for this task.
Your tests should:

1. Cover all requirements and functionality
2. Include both normal cases and edge cases
3. Be clear and easy to understand
4. Use pytest or unittest framework
5. Provide good test coverage

Return ONLY the Python test code. Start with ```python and end with ```.
"""
        
        self.logger.info(f"Tester creating tests for task: {task['name']}")
        
        # Get test code from LLM
        response = self._get_llm_response(prompt)
        
        # Extract code between triple backticks
        code_blocks = response.split("```")
        if len(code_blocks) >= 3:
            # Get the content of the first code block
            test_code = code_blocks[1]
            # Remove language identifier if present
            if test_code.startswith("python"):
                test_code = test_code[len("python"):].strip()
            else:
                test_code = test_code.strip()
        else:
            # If no code blocks found, use entire response
            test_code = response
        
        self.test_code = test_code
        
        # Create message with test cases
        message = self.create_message(
            receiver="all",
            content=f"Here are the test cases for the implementation:\n\n```python\n{self.test_code}\n```",
            message_type="test_cases"
        )
        
        self.logger.info("Tester completed test creation")
        
        return message
    
    def evaluate_implementation(self, implementation: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an implementation against requirements and test cases.
        
        Args:
            implementation: The code implementation
            task: The programming task description
            
        Returns:
            A message containing the evaluation results
        """
        prompt = f"""
You are an AI tester responsible for evaluating code implementations.

# Task Description
{task['description']}

# Requirements
{', '.join(task['requirements'])}

# Implementation
```python
{implementation}
```

# Test Cases
```python
{self.test_code}
```

# Instructions
Please evaluate this implementation against the requirements and test cases.
Your evaluation should:

1. Analyze if the code meets all requirements
2. Identify any bugs or issues
3. Check for edge cases that might not be handled
4. Assess code quality and best practices
5. Provide specific feedback on what needs to be fixed or improved

Return a detailed evaluation report with specific issues and suggested improvements.
"""
        
        self.logger.info(f"Tester evaluating implementation for task: {task['name']}")
        
        # Get evaluation from LLM
        evaluation = self._get_llm_response(prompt)
        
        # Create message with evaluation results
        message = self.create_message(
            receiver="all",
            content=f"Evaluation Results:\n\n{evaluation}",
            message_type="evaluation"
        )
        
        self.logger.info("Tester completed implementation evaluation")
        
        return message


class ReviewerAgent(Agent):
    """Reviewer agent responsible for evaluating code quality and suggesting improvements."""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        super().__init__("reviewer", model_name, logger)
        
    def review_code(self, implementation: str, design_document: str) -> Dict[str, Any]:
        """
        Review code implementation for quality and adherence to best practices.
        
        Args:
            implementation: The code implementation
            design_document: The design document
            
        Returns:
            A message containing the code review
        """
        prompt = f"""
You are an AI code reviewer responsible for evaluating code quality.

# Design Document
{design_document}

# Implementation
```python
{implementation}
```

# Instructions
Please review this code implementation for quality and best practices.
Your review should:

1. Evaluate adherence to design specifications
2. Identify code quality issues (readability, maintainability)
3. Point out potential bugs or edge cases
4. Suggest specific improvements
5. Comment on overall structure and organization
6. Provide feedback on documentation and comments

Return a detailed code review with specific issues and suggested improvements.
"""
        
        self.logger.info("Reviewer performing code review")
        
        # Get review from LLM
        review = self._get_llm_response(prompt)
        
        # Create message with code review
        message = self.create_message(
            receiver="all",
            content=f"Code Review:\n\n{review}",
            message_type="code_review"
        )
        
        self.logger.info("Reviewer completed code review")
        
        return message
    
    def approve_implementation(self, implementation: str) -> Dict[str, Any]:
        """
        Approve an implementation after all issues have been addressed.
        
        Args:
            implementation: The final code implementation
            
        Returns:
            A message containing the approval
        """
        # Create message with approval
        message = self.create_message(
            receiver="all",
            content="APPROVAL: The implementation meets all requirements and quality standards. The code is approved for release.",
            message_type="approval"
        )
        
        self.logger.info("Reviewer approved final implementation")
        
        return message


class ModeratorAgent(Agent):
    """Moderator meta-agent responsible for coordinating team activities."""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", logger: Optional[logging.Logger] = None):
        super().__init__("moderator", model_name, logger)
        self.task = None
        self.phase = "initialization"
        self.pending_issues = []
        
    def initialize_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Initialize a task and create briefing messages for all agents.
        
        Args:
            task: The programming task description
            
        Returns:
            A list of briefing messages for all agents
        """
        self.task = task
        self.phase = "initialization"
        
        self.logger.info(f"Moderator initializing task: {task['name']}")
        
        # Create briefing message for all agents
        briefing = f"""
# Task Briefing

## Task: {task['name']}
Complexity: {task['complexity']}

## Description
{task['description']}

## Requirements
{', '.join(task['requirements'])}

## Expected Outputs
{json.dumps(task['expected_output'], indent=2)}

## Development Plan
1. Architect will create a design document
2. Tester will prepare test cases based on the design
3. Implementer will develop code according to the design
4. Tester will evaluate the implementation
5. Reviewer will review code quality
6. Implementer will address feedback
7. Final approval and integration

Let's work together to create a high-quality solution.
"""
        
        # Create messages for each role
        messages = []
        for role in ["architect", "implementer", "tester", "reviewer"]:
            messages.append(self.create_message(
                receiver=role,
                content=briefing,
                message_type="task_briefing"
            ))
        
        self.logger.info("Moderator completed task initialization")
        
        return messages
    
    def update_phase(self, new_phase: str) -> Dict[str, Any]:
        """
        Update the current development phase.
        
        Args:
            new_phase: The new phase name
            
        Returns:
            A message announcing the phase change
        """
        self.phase = new_phase
        
        phase_descriptions = {
            "initialization": "Task initialization and team briefing",
            "design": "Architecture and system design",
            "test_planning": "Test case creation and planning",
            "implementation": "Code implementation based on design",
            "testing": "Implementation testing and evaluation",
            "review": "Code quality review",
            "refinement": "Addressing feedback and making improvements",
            "final_approval": "Final review and approval",
            "completion": "Task completion and wrap-up"
        }
        
        description = phase_descriptions.get(new_phase, new_phase)
        
        self.logger.info(f"Moderator updating phase to: {new_phase}")
        
        # Create message announcing phase change
        message = self.create_message(
            receiver="all",
            content=f"PHASE CHANGE: We are now moving to the {new_phase.upper()} phase. {description}.",
            message_type="phase_change"
        )
        
        return message
    
    def resolve_conflict(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts between team members.
        
        Args:
            messages: The conflicting messages
            
        Returns:
            A message with conflict resolution
        """
        prompt = f"""
You are an AI moderator responsible for resolving conflicts in a development team.

# Conflicting Messages
"""
        
        for idx, msg in enumerate(messages):
            prompt += f"\n## Message {idx+1} from {msg.get('sender', 'Unknown')}\n{msg.get('content', '')}\n"
        
        prompt += """
# Instructions
Please resolve this conflict by:

1. Summarizing the different viewpoints
2. Identifying the core issues
3. Evaluating the technical merits of each position
4. Making a final decision that best serves the project
5. Providing a clear path forward with specific recommendations

Return a detailed conflict resolution that addresses all points of contention.
"""
        
        self.logger.info("Moderator resolving conflict")
        
        # Get resolution from LLM
        resolution = self._get_llm_response(prompt)
        
        # Create message with conflict resolution
        message = self.create_message(
            receiver="all",
            content=f"Conflict Resolution:\n\n{resolution}",
            message_type="conflict_resolution"
        )
        
        self.logger.info("Moderator completed conflict resolution")
        
        return message
    
    def summarize_progress(self) -> Dict[str, Any]:
        """
        Create a summary of the current progress.
        
        Returns:
            A message with a progress summary
        """
        prompt = f"""
You are an AI moderator responsible for summarizing team progress.

# Current Task
Task: {self.task['name']}
Complexity: {self.task['complexity']}
Current Phase: {self.phase}

# Instructions
Please create a concise summary of the team's current progress, including:

1. What has been accomplished so far
2. Current status of each team member's work
3. Any outstanding issues or challenges
4. Next steps and priorities
5. Timeline assessment

Return a clear, informative progress summary.
"""
        
        self.logger.info("Moderator creating progress summary")
        
        # Get summary from LLM
        summary = self._get_llm_response(prompt)
        
        # Create message with progress summary
        message = self.create_message(
            receiver="all",
            content=f"Progress Summary:\n\n{summary}",
            message_type="progress_summary"
        )
        
        self.logger.info("Moderator completed progress summary")
        
        return message


class MACPFramework:
    """
    Multi-Agent Collaborative Programming Framework main class.
    """
    
    def __init__(
        self, 
        model_name: str = "claude-3-7-sonnet-20250219",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the MACP framework with specialized agents.
        
        Args:
            model_name: Name of the LLM to use for all agents
            logger: Logger instance for tracking events
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = {
            "architect": ArchitectAgent(model_name, self.logger),
            "implementer": ImplementerAgent(model_name, self.logger),
            "tester": TesterAgent(model_name, self.logger),
            "reviewer": ReviewerAgent(model_name, self.logger),
            "moderator": ModeratorAgent(model_name, self.logger)
        }
        
        # Initialize message queue and processing thread
        self.message_queue = deque()
        self.message_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.processing_thread = None
        
        # Collection of all messages
        self.all_messages = []
        
        # Current solution
        self.current_solution = ""
        
        self.logger.info("MACP Framework initialized")
    
    def solve_task(self, task: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a programming task using the collaborative framework.
        
        Args:
            task: The programming task description
            
        Returns:
            Tuple containing the solution code and metadata
        """
        self.logger.info(f"MACP Framework solving task: {task['name']}")
        
        # Record start time
        start_time = time.time()
        
        # Start message processing thread
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.start()
        
        try:
            # Initialize task with moderator
            initialization_messages = self.agents["moderator"].initialize_task(task)
            for message in initialization_messages:
                self._queue_message(message)
            
            # Phase 1: Design
            self._queue_message(self.agents["moderator"].update_phase("design"))
            design_message = self.agents["architect"].create_design(task)
            self._queue_message(design_message)
            
            # Wait for design processing
            time.sleep(2)
            
            # Phase 2: Test Planning
            self._queue_message(self.agents["moderator"].update_phase("test_planning"))
            design_document = next((m["content"] for m in self.all_messages if m.get("sender") == "architect" and m.get("message_type") == "design_document"), "")
            test_message = self.agents["tester"].create_tests(design_document, task)
            self._queue_message(test_message)
            
            # Wait for test planning processing
            time.sleep(2)
            
            # Phase 3: Implementation
            self._queue_message(self.agents["moderator"].update_phase("implementation"))
            implementation_message = self.agents["implementer"].implement_code(design_document, task)
            self._queue_message(implementation_message)
            
            # Wait for implementation processing
            time.sleep(2)
            
            # Phase 4: Testing
            self._queue_message(self.agents["moderator"].update_phase("testing"))
            implementation = next((m["content"] for m in self.all_messages if m.get("sender") == "implementer" and m.get("message_type") == "implementation"), "")
            
            # Extract code from the message
            code_blocks = implementation.split("```")
            if len(code_blocks) >= 3:
                code = code_blocks[1]
                if code.startswith("python"):
                    code = code[len("python"):].strip()
                else:
                    code = code.strip()
            else:
                code = implementation
            
            evaluation_message = self.agents["tester"].evaluate_implementation(code, task)
            self._queue_message(evaluation_message)
            
            # Wait for testing processing
            time.sleep(2)
            
            # Phase 5: Review
            self._queue_message(self.agents["moderator"].update_phase("review"))
            review_message = self.agents["reviewer"].review_code(code, design_document)
            self._queue_message(review_message)
            
            # Wait for review processing
            time.sleep(2)
            
            # Phase 6: Refinement
            self._queue_message(self.agents["moderator"].update_phase("refinement"))
            
            # Combine feedback from tester and reviewer
            evaluation = next((m["content"] for m in self.all_messages if m.get("sender") == "tester" and m.get("message_type") == "evaluation"), "")
            review = next((m["content"] for m in self.all_messages if m.get("sender") == "reviewer" and m.get("message_type") == "code_review"), "")
            combined_feedback = f"Feedback from Tester:\n{evaluation}\n\nFeedback from Reviewer:\n{review}"
            
            update_message = self.agents["implementer"].update_code(combined_feedback)
            self._queue_message(update_message)
            
            # Wait for refinement processing
            time.sleep(2)
            
            # Phase 7: Final Approval
            self._queue_message(self.agents["moderator"].update_phase("final_approval"))
            
            # Extract updated code
            updated_implementation = next((m["content"] for m in self.all_messages if m.get("sender") == "implementer" and m.get("message_type") == "implementation_update"), "")
            code_blocks = updated_implementation.split("```")
            if len(code_blocks) >= 3:
                updated_code = code_blocks[1]
                if updated_code.startswith("python"):
                    updated_code = updated_code[len("python"):].strip()
                else:
                    updated_code = updated_code.strip()
            else:
                updated_code = updated_implementation
            
            approval_message = self.agents["reviewer"].approve_implementation(updated_code)
            self._queue_message(approval_message)
            
            # Wait for approval processing
            time.sleep(2)
            
            # Phase 8: Completion
            self._queue_message(self.agents["moderator"].update_phase("completion"))
            summary_message = self.agents["moderator"].summarize_progress()
            self._queue_message(summary_message)
            
            # Final solution is the updated code
            self.current_solution = updated_code
            
            # Record end time
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Collect metadata
            metadata = {
                'execution_time': execution_time,
                'messages_count': len(self.all_messages),
                'model_name': model_name,
                'task_id': task['id'],
                'task_name': task['name'],
                'task_complexity': task['complexity'],
                'messages': self.all_messages
            }
            
            self.logger.info(f"MACP Framework completed task in {execution_time:.2f} seconds with {len(self.all_messages)} messages")
            
            return self.current_solution, metadata
            
        finally:
            # Stop message processing
            self.stop_event.set()
            if self.processing_thread:
                self.processing_thread.join()
    
    def _queue_message(self, message: Dict[str, Any]):
        """
        Add a message to the processing queue.
        
        Args:
            message: The message to queue
        """
        with self.message_lock:
            self.message_queue.append(message)
            self.all_messages.append(message)
    
    def _process_messages(self):
        """Process messages in the queue until stop event is set."""
        while not self.stop_event.is_set():
            message = None
            with self.message_lock:
                if self.message_queue:
                    message = self.message_queue.popleft()
            
            if message:
                self._process_single_message(message)
            
            time.sleep(0.1)  # Small delay to prevent CPU spinning
    
    def _process_single_message(self, message: Dict[str, Any]):
        """
        Process a single message by routing it to the appropriate agent.
        
        Args:
            message: The message to process
        """
        if message.get("receiver") == "all":
            # Broadcast message does not need a response
            self.logger.info(f"Broadcast message from {message.get('sender')}")
            return
        
        receiver = message.get("receiver")
        if receiver in self.agents:
            response = self.agents[receiver].process_message(message)
            self._queue_message(response)
        else:
            self.logger.warning(f"Message sent to unknown receiver: {receiver}")