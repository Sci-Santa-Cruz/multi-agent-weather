
import os
import logging
import requests
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
# Loads environment variables from the .env file, typically containing API keys like the News API key.
load_dotenv(dotenv_path='env')

# ----- Global LLM instance -----
# This initializes the LLM (language model) from OpenAI with the "gpt-3.5-turbo" model.
# It's used for natural language understanding and extracting country codes from text.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ----- Prompt Template for country extraction -----
# This is the prompt used by the LLM to extract the country code (ISO 3166-1 alpha-2) from the provided text.
# The prompt asks for a 2-letter country code and returns only that code.
country_extraction_template = """
You are an assistant that extracts the country (in ISO 3166-1 alpha-2 code, like 'MX', 'US', 'FR') from the following text.
Respond only with the country code. If no country is mentioned, respond with ' '.

Text: "{text}"
"""

# Create a prompt template that the LLM will use.
country_extraction_prompt = PromptTemplate(
    input_variables=["text"],  # The input variable for the template is 'text'.
    template=country_extraction_template  # The actual template for extraction.
)

# ----- Function to extract country from the text using the LLM -----
def extract_country_with_llm(text: str) -> str:
    """
    Extracts the country code from the provided text using the LLM.

    Parameters:
    text (str): The input text containing a country mention.

    Returns:
    str: The ISO 3166-1 alpha-2 country code extracted from the text (or ' ' if no country is mentioned).
    """
    try:
        # Format the prompt with the provided text
        prompt = country_extraction_prompt.format(text=text)
        logger.debug(f"Prompt sent to LLM: {prompt}")
        
        # Get the response from the LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        country = response.content.strip().lower()  # Normalize the country code (convert to lowercase)
        logger.debug(f"LLM response: {country}")

        # Validate that the response is a valid 2-letter country code
        if len(country) != 2 or not country.isalpha():
            logger.warning("Invalid response from LLM")
            return None  # Return a blank space if the response is invalid
        return country

    except Exception as e:
        # Log any exception that occurs during country extraction
        logger.exception("Error during country extraction")
        return None  # Return 'None' as a default fallback

# ----- News fetching function -----
def get_news(state: AgentState) -> AgentState:
    """
    Processes the input text to detect the country and fetches news headlines for that country.

    Parameters:
    input_text (str): The input text containing user query or message.

    Returns:
    dict: A dictionary containing the news headlines or error information.
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_news")
    
    # Get the input text from the state (the user's message)
    input_text = state["messages"][-1].content
    try:
        logger.info(f"Processing user message: {input_text}")

        # Extract the country code from the user's message
        country_code = extract_country_with_llm(input_text)
        logger.info(f"Detected country code: {country_code}")

        # Retrieve the News API key from the environment variables
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            msg = "News API Key is not configured."
            logger.error(msg)
            return {
                    "error": {"news": msg},
                   "task_completed": {"news": False}
            }

        # Construct the API URL to get news headlines for the detected country
        url = f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={api_key}"
        logger.debug(f"Querying News API: {url}")
        response = requests.get(url)

        # Handle non-200 HTTP responses from the News API
        if response.status_code != 200:
            msg = f"Error in News API: {response.status_code}"
            logger.error(msg)
            return {
                "error": {"news": msg},
                "task_completed": {"news": False}
            }

        # Parse the JSON response from the News API
        data = response.json()

        # Check if the 'articles' key exists in the response and has data
        if "articles" not in data or not data["articles"]:
            msg = f"No news found for {country_code}."
            logger.warning(msg)
            return {
                "error": {"news": msg},
                "task_completed": {"news": False}
                   }

        # Extract the titles of the top 3 articles from the response
        titles = ", ".join([article["title"] for article in data["articles"][:3]])
        headlines = f"Headlines in {country_code.upper()}: {titles}"

        logger.info(f"Found headlines: {headlines}")

        # Return the news headlines as a dictionary
        return {"results": {"news": [headlines]},
                "task_completed": {"news": True}
}

    except Exception as e:
        # Handle any unexpected errors during the news retrieval process
        logger.exception("Unexpected error while fetching news")
        return {
            "error": {"news": f"Error fetching news: {str(e)}"},
            "task_completed": {"news" : False}
            }
