
import os
import logging
import requests
from typing import Optional
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.agent_state import AgentState

# ----- Configure logging -----
from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Load environment variables -----
# This loads environment variables from a .env file, typically containing sensitive information like API keys.
load_dotenv(dotenv_path='env')

# ----- Global LLM instance -----
# Initializes the language model for currency extraction. 
# The model used here is GPT-3.5-turbo, which is ideal for text processing and extraction tasks.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ----- Prompt Template for currency extraction -----
# This is the template used to instruct the language model to extract currency codes (ISO 4217 format) from the given text.
# The prompt is written in Spanish but will be used to parse any input text in the same format.
currency_extraction_template = """
Eres un asistente que extrae dos códigos de divisas (ISO 4217) desde el texto dado. 
Responde únicamente con los códigos separados por coma. No uses símbolos ni explicaciones.

Ejemplo:
Texto: "¿Cuánto vale un dólar en pesos mexicanos?" -> USD, MXN
Texto: "{text}"
"""

currency_extraction_prompt = PromptTemplate(
    input_variables=["text"],  # The input variable that will be passed to the template is 'text'.
    template=currency_extraction_template  # The prompt template for extraction.
)

# ----- Function to extract currencies using the language model (LLM) -----
def extract_currencies_with_llm(text: str) -> Optional[tuple[str, str]]:
    """
    Extracts two currency codes from a given text using the pre-defined language model prompt.
    
    Parameters:
    text (str): The input text containing the currencies to be extracted.
    
    Returns:
    Optional[tuple[str, str]]: A tuple containing two ISO 4217 currency codes (or None if extraction fails).
    """
    try:
        # Format the prompt with the provided text
        prompt = currency_extraction_prompt.format(text=text)
        logger.debug(f"Prompt sent to LLM: {prompt}")
        
        # Get the response from the LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        logger.debug(f"LLM response: {result}")

        # Split the result into two parts (currency codes)
        parts = [p.strip().upper() for p in result.split(",")]

        # Check if the result contains exactly two 3-letter currency codes
        if len(parts) == 2 and all(len(code) == 3 for code in parts):
            logger.info(f"Successfully extracted currency codes: {parts}")
            return parts[0], parts[1]
        else:
            logger.warning(f"Unexpected format in the response: {result}")

    except Exception as e:
        # Log the error if extraction fails
        logger.exception("Error during currency extraction")

    return None

# ----- Function to get exchange rate -----
def get_exchange_rate(state: AgentState) -> AgentState:
    """
    Processes the user's message to detect currencies and fetches the exchange rate between them.
    
    Parameters:
    state (dict): The current state of the agent, which contains the user's message and other context.
    
    Returns:
    dict: The updated state dictionary containing either the results or error messages.
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_exchange")
    try:
        # Get the input text from the state (the user's message)
        input_text = state["messages"][-1].content
        logger.info(f"Processing user message: {input_text}")

        # Extract the currency codes from the user's input
        currencies = extract_currencies_with_llm(input_text)
        
        # If no currencies are detected, return an error
        if not currencies:
            msg = "No currencies detected in the message."
            logger.warning(msg)
            return {
                "error": {"exchange": msg},
                "task_completed":{"exchange": True} 
            }

        # Extract base and target currencies
        base_currency, target_currency = currencies
        logger.info(f"Detected currencies: {base_currency} -> {target_currency}")

        # Retrieve the API key for the exchange rate service from environment variables
        api_key = os.getenv("EXCHANGE_API_KEY")
        if not api_key:
            logger.error("API Key not set in the environment.")
            return {
                "error": {"exchange": "API Key not configured in the system."},
                "task_completed":{"exchange": True} 
            }

        # Construct the API URL to get the exchange rates
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
        logger.debug(f"Querying external API: {url}")
        response = requests.get(url)

        # Check if the response from the API is successful
        if response.status_code != 200:
            logger.error(f"Error in API response: {response.status_code}")
            return {
                "error": {"exchange": f"API error: {response.status_code}"},
                "task_completed":{"exchange": False} 
            }

        # Parse the JSON data from the API response
        data = response.json()

        # Check if the target currency's exchange rate is available
        if target_currency not in data.get('conversion_rates', {}):
            logger.warning(f"Exchange rate for {target_currency} not available in the response.")
            return {
                "error": {"exchange": f"Exchange rate for {target_currency} not found."},
                "task_completed":{"exchange": True} 
            }

        # Get the exchange rate for the target currency and format the response message
        rate = data['conversion_rates'][target_currency]
        message = f"1 {base_currency} = {rate} {target_currency}"
        logger.info(f"Exchange rate obtained: {message}")

        # Return the exchange rate result in the updated state
        return {
            "results": {"exchange": [message]},
            "task_completed":{"exchange": True} 
        }

    except Exception as e:
        # Handle any unexpected errors and return them in the state
        logger.exception("Unexpected error while getting exchange rate")
        return {
            "error": {"exchange": f"Error obtaining exchange rate: {str(e)}"},
            "task_completed":{"exchange": False} 
        }
