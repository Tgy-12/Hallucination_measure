# measure_hallucination/mitigation/prompt_engineering.py
from typing import Dict, List, Optional
import json

class PromptEngineer:
    """Anti-hallucination prompt engineering strategies"""
    
    def __init__(self):
        self.templates = {
            'strict_context_only': """
            Answer the question based ONLY on the provided context. 
            Follow these rules strictly:
            1. Only use information from the context
            2. If the answer is not in the context, say "I cannot answer based on the provided information"
            3. Do not add any external knowledge or assumptions
            4. Be concise and accurate
            5. Quote relevant parts when possible
            
            Context: {context}
            
            Question: {question}
            
            Answer: """,
            
            'confidence_based': """
            Based on the context below, answer the question.
            
            Important: 
            - If you are highly confident (>80% sure), provide a direct answer
            - If you are moderately confident (50-80% sure), add "Based on the available information: "
            - If you are less than 50% confident, say "I don't have enough information to answer confidently"
            
            Context: {context}
            
            Question: {question}
            
            First, think: What is my confidence level? Then answer accordingly:
            """,
            
            'step_by_step': """
            Follow these steps to answer:
            1. Read the context carefully
            2. Identify relevant information for the question
            3. Verify each piece of information is in the context
            4. If any required information is missing, note it
            5. Formulate answer using only verified information
            6. Add a confidence score (High/Medium/Low)
            
            Context: {context}
            
            Question: {question}
            
            Step-by-step reasoning: """,
            
            'medical_legal': """
            CRITICAL: This is for {domain} purposes. Accuracy is essential.
            
            Instructions:
            - Only use explicitly stated facts from the context
            - Do not infer, assume, or extrapolate
            - If uncertain, state the limitation clearly
            - Cite specific sections when possible
            - Flag any contradictions in the context
            
            Context: {context}
            
            Question: {question}
            
            Professional answer: """
        }
    
    def get_prompt(self, template_name: str, context: str, 
                   question: str, **kwargs) -> str:
        """
        Get anti-hallucination prompt
        
        Args:
            template_name: Name of template to use
            context: Source context
            question: User question
            **kwargs: Additional template parameters
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found. "
                           f"Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        formatted = template.format(context=context, question=question, **kwargs)
        
        return formatted
    
    def create_custom_prompt(self, rules: List[str], 
                            format_spec: Optional[Dict] = None) -> str:
        """
        Create custom anti-hallucination prompt
        
        Args:
            rules: List of anti-hallucination rules
            format_spec: Format specification
            
        Returns:
            Custom prompt template
        """
        if format_spec is None:
            format_spec = {
                'context_placeholder': '{context}',
                'question_placeholder': '{question}',
                'answer_placeholder': '{answer}'
            }
        
        rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(rules)])
        
        custom_template = f"""
        Answer the question following these anti-hallucination rules:
        
        {rules_text}
        
        Context: {format_spec['context_placeholder']}
        
        Question: {format_spec['question_placeholder']}
        
        Answer: {format_spec.get('answer_placeholder', '')}
        """
        
        return custom_template
    
    def add_system_message(self, prompt: str, 
                          system_role: str = "assistant") -> List[Dict]:
        """
        Convert prompt to chat format with system message
        
        Args:
            prompt: The prompt text
            system_role: Role for system message
            
        Returns:
            List of message dictionaries for chat models
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that strictly avoids hallucination. "
                          "Only use information from the provided context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        return messages
