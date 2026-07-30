"""
Base class for LLM runners
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import logging

class BaseLLMRunner(ABC):
    """Base class for all LLM runners"""
    
    def __init__(self, model_config: Dict[str, Any], api_key: str):
        """
        Initialize the runner
        
        Args:
            model_config: Configuration dict for the model
            api_key: API key for the service
        """
        self.model_config = model_config
        self.api_key = api_key
        self.model_name = model_config.get('model_name')
        self.temperature = model_config.get('temperature', 0.7)
        self.max_tokens = model_config.get('max_tokens', 2000)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def query(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the LLM with a prompt
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary containing:
                - response: The model's response text
                - usage: Token usage information
                - latency: Response time in seconds
                - error: Error message if any
        """
        pass
    
    def format_history_question(self, question_data: Dict[str, Any], language: str = "vi") -> str:
        """
        Format a history question for the model
        
        Args:
            question_data: Question data from dataset
            language: Language for the question (vi or en)
            
        Returns:
            Formatted prompt string
        """
        if language == "vi":
            prompt = f"""Câu hỏi về lịch sử Việt Nam:

{question_data['question']}

Ngữ cảnh từ tư liệu gốc:
{question_data.get('source_snippet', 'Không có')}

Hãy trả lời câu hỏi một cách chi tiết và chính xác dựa trên kiến thức lịch sử Việt Nam."""
        else:  # English
            prompt = f"""Question about Vietnamese history:

{question_data['question']}

Context from original sources:
{question_data.get('source_snippet', 'Not available')}

Please answer the question in detail and accurately based on Vietnamese historical knowledge."""
        
        return prompt
    
    def create_few_shot_prompt(self, question_data: Dict[str, Any], 
                               examples: list, language: str = "vi") -> str:
        """
        Create a few-shot learning prompt
        
        Args:
            question_data: The question to answer
            examples: List of example Q&A pairs
            language: Language for the prompt
            
        Returns:
            Formatted few-shot prompt
        """
        if language == "vi":
            prompt = "Dưới đây là một số ví dụ về cách trả lời câu hỏi lịch sử:\n\n"
            
            for i, ex in enumerate(examples, 1):
                prompt += f"Ví dụ {i}:\n"
                prompt += f"Câu hỏi: {ex['question']}\n"
                prompt += f"Trả lời: {ex['answer']}\n\n"
            
            prompt += "Bây giờ hãy trả lời câu hỏi sau theo cách tương tự:\n\n"
            prompt += self.format_history_question(question_data, language)
        else:
            prompt = "Here are some examples of how to answer history questions:\n\n"
            
            for i, ex in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Question: {ex['question']}\n"
                prompt += f"Answer: {ex['answer']}\n\n"
            
            prompt += "Now answer the following question in a similar way:\n\n"
            prompt += self.format_history_question(question_data, language)
        
        return prompt
