from dotenv import load_dotenv

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing import cast
from core.agent_state import AgentState
import json

from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Cargar variables de entorno -----
load_dotenv(dotenv_path='env')

# Global LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ----- System message (prompt) -----
system_prompt = SystemMessage(
    content="""
Eres un asistente que clasifica tareas en tres categorías: weather, exchange y news.
Dado un mensaje del usuario, responde con un JSON con claves: "weather", "exchange", "news",
y valores booleanos indicando si la tarea está presente.
Ejemplo de respuesta: {"weather": true, "exchange": false, "news": true}
"""
)

# ----- Node: classify_tasks -----
def classify_tasks(state: AgentState) -> AgentState:
    """
    Clasifica la intención del usuario en categorías predefinidas
    usando un modelo de lenguaje y actualiza el estado.
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("classify_tasks")
    try:
        logger.debug("Buscando el último mensaje del usuario...")
        user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        logger.info(f"Mensaje recibido: {user_msg.content}")

        full_prompt = [system_prompt, user_msg]

        logger.debug("Enviando prompt al modelo...")
        response = llm.invoke(full_prompt)
        logger.debug(f"Respuesta del modelo: {response.content}")

        classification = json.loads(response.content)
        classification = cast(dict, classification)
        logger.info(f"Tareas clasificadas: {classification}")

        new_state = {
            "tasks_to_do": classification,
            "results": state.get("results", {}),
            "task_completed": {},
            "error": {},
            "order_task": {},
            "ready_to_aggregate": False,
        }

        logger.debug("Estado actualizado correctamente.")
        return new_state

    except Exception as e:
        error_msg = f"classify: {str(e)}"
        logger.exception("Error al clasificar el mensaje del usuario.")
        return {
            "messages": state["messages"] + [SystemMessage(content=error_msg)],
            "results": state.get("results", {}),
            "tasks_to_do": {},
            "error": {"classify": error_msg},
            "order_task": {},
            "ready_to_aggregate": False,
            "task_completed": {}
        }
