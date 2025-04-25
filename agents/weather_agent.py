
from dotenv import load_dotenv
import os
import requests
import logging
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from core.agent_state import AgentState

from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Load environment variables -----
load_dotenv(dotenv_path='env')

# Global LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ----- Prompt Template -----
city_extraction_template = """
Eres un asistente que extrae el nombre de la ciudad en inglés americano del siguiente texto. 
Responde solo con el nombre de la ciudad, sin comillas ni símbolos extra.

Ejemplo:
Texto: "¿Cómo está el clima en Nueva York?" -> New York
Texto: "{text}"
"""

city_extraction_prompt = PromptTemplate(input_variables=["text"], template=city_extraction_template)

# ----- LLM-based city extractor -----
def extract_city_with_llm(text: str) -> Optional[str]:
    """
    Extracts the name of the city from the given text using a language model.

    Args:
    - text (str): The input text that may contain a city name.

    Returns:
    - str or None: Returns the city name if successfully extracted, otherwise None.
    """
    logger.debug(f"Extracting city from text: '{text}'")
    prompt = city_extraction_prompt.format(text=text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        city = response.content.strip()
        logger.info(f"City extracted: '{city}'")
    except Exception as e:
        logger.exception("Error invoking the model for city extraction.")
        return None

    if not city or len(city) < 2 or any(c in city for c in ['{', '}', '[', ']']):
        logger.warning(f"Invalid city detected: '{city}'")
        return None

    return city

# ----- Weather Node -----
def get_weather(state: AgentState) -> AgentState:
    """
    Handles weather-related queries using an LLM to extract the city and the OpenWeatherMap API to fetch weather data.

    Returns:
    - 'results': If successful, a dictionary with the weather report.
    - 'error': If an issue occurs, a dictionary with the error message.
    - 'task_completed': A boolean flag indicating if the task was completed.
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_weather")

    try:
        input_text = state["messages"][-1].content
        logger.debug(f"Received weather message: '{input_text}'")

        city = extract_city_with_llm(input_text)
        logger.debug(f"Respose llm: '{city}'")

        if not city:
            msg = "City could not be identified in the message."
            logger.warning(msg)
            return {
                "error": {"weather": msg},
                "task_completed": {"weather": False}
            }

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            msg = "API Key not configured in the system."
            logger.error(msg)
            return {
                "error": {"weather": msg},
                "task_completed": {"weather": False}
            }

        logger.info(f"Fetching weather for: {city}")
        location_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        location_response = requests.get(location_url, params=params)

        if location_response.status_code != 200:
            msg = f"City '{city}' not found or not correctly written in English."
            logger.warning(msg)
            return {
                "error": {"weather": msg},
                "task_completed": {"weather": False}
            }

        location_data = location_response.json()

        try:
            weather_desc = location_data["weather"][0]["description"]
            temperature = location_data["main"]["temp"]
        except (KeyError, IndexError, TypeError) as e:
            msg = "Unexpected weather data format received from API."
            logger.exception(msg)
            return {
                "error": {"weather": msg},
                "task_completed": {"weather": False}
            }

        weather_report = f"The weather in {city} is {weather_desc} with a temperature of {temperature}°C."
        logger.info(f"Generated weather report: {weather_report}")

        return {
            "results": {"weather": [weather_report]},
            "task_completed": {"weather": True}
        }

    except Exception as e:
        msg = f"Error obtaining weather: {str(e)}"
        logger.exception(msg)
        return {
            "error": {"weather": msg},
            "task_completed": {"weather": False}
        }
