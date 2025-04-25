
import json
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from core.agent_state import AgentState
from langchain_openai import ChatOpenAI

# ----- Configurar logging -----
from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Cargar variables de entorno -----
load_dotenv(dotenv_path='env')

# Global LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ----- Task Ordering Prompt -----
order_tasks_template = PromptTemplate(
    input_variables=["tasks", "user_input"],
    template="""
Eres un asistente experto en coordinar tareas de un sistema que puede obtener información sobre clima, noticias y divisas.

El usuario ha solicitado información sobre las siguientes tareas: {tasks}.

Consulta original del usuario:
"{user_input}"

Analiza la intención del usuario y determina el orden más lógico y útil en el que se deben ejecutar estas tareas. 
Devuelve únicamente un objeto JSON donde cada clave sea el nombre de la tarea y su valor sea su posición en el orden en el que aparece en el texto ,
si no aparece alguna tarea omitela de la respuesta
por ejemplo: {{"weather": 1, "exchange": 2, "noticias": 3}}

No incluyas ningún otro texto ni explicaciones.
"""
)

# ----- Task Ordering Node -----
def order_tasks(state: AgentState) -> AgentState:
    """
    Orders tasks based on the user's input and their intent, helping to prioritize actions.

    Parameters:
        state (AgentState): The shared state that includes tasks and the user's query.

    Returns:
        dict: {"results": {"order": ...}, "task_completed": {"order": True}}
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_order")
    tasks = {k: v for k, v in state.get("tasks", {}).items() if v}
    user_input = state["messages"][-1].content if state.get("messages") else ""

    logger.info(f"Tareas detectadas: {list(tasks.keys())}")
    logger.debug(f"Consulta del usuario: {user_input}")

    try:
        prompt = order_tasks_template.format(
            tasks=", ".join(tasks.keys()),
            user_input=user_input
        )
        logger.debug(f"Prompt generado para el LLM:\n{prompt}")

        response = llm.invoke([HumanMessage(content=prompt)])
        ordered_dict = json.loads(response.content.strip())

        logger.info(f"Orden propuesto por LLM: {ordered_dict}")

        return {
            "order_task":  ordered_dict,
            "task_completed": {'order':True}
        }

    except Exception as e:
        error_msg = f"Error al ordenar tareas: {str(e)}"
        logger.exception(error_msg)
        return {
            "task_completed": {'order':False},
            "error": {"order": error_msg}
        }
