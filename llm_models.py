import streamlit as st
import requests
from langchain.llms.base import LLM
from typing import Optional, List
import os
from execution_programs import *

# API key for llm model
GROQ_API_KEY = os.environ.get("GROQ_TOKENS")
class GroqLLM(LLM):
    model: str = "qwen-2.5-coder-32b"
    temperature: float = 0.3
    api_key: str = GROQ_API_KEY

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return "Sorry, there was an error processing your request."

class GroqTextLLM(LLM):
    model: str = "deepseek-r1-distill-llama-70b"
    temperature: float = 0.7
    api_key: str = GROQ_API_KEY

    @property
    def _llm_type(self) -> str:
        return "groq-text-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": self.temperature,
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return "Sorry, there was an error processing your request."

# Initialize LLMs
code_llm = GroqLLM()
text_llm = GroqTextLLM()