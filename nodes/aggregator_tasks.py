
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.agent_state import AgentState  # Ajustar según sea necesario

# ----- Configurar logging -----
from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Cargar variables de entorno -----
load_dotenv(dotenv_path='env')

# Instancia global de LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Plantilla para "enchulamiento"
enchulador_template = PromptTemplate(
    input_variables=["mensaje"],
    template="""
Enchula el siguiente mensaje para hacerlo amigable para el usuario:

{mensaje}
"""
)

def aggregator(state: AgentState) -> AgentState:
    """
    Reformula resultados exitosos con el LLM y agrega errores directamente.
    Devuelve mensajes listos para mostrar al usuario.

    Args:
        state (dict): Contiene 'order_task', 'results', 'error'.

    Returns:
        dict: {"results": {"aggregator": [messages]}, "task_completed":  {}}
    """
    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_aggregator")
    processed_messages = []

    logger.info("Iniciando agregación de tareas...")

    # Ordenar las tareas según el orden dado en el diccionario order_task
    # Ordenamos el diccionario order_task por los valores (el orden de las tareas)
    sorted_order = sorted(state.get("order_task", {}).items(), key=lambda item: item[1])
    logger.info("Orden detectado  %s", sorted_order)

    # Iterar en el orden correcto
    for task, order in sorted_order:
        result = state.get("results", {}).get(task)
        error = state.get("error", {}).get(task)

        logger.debug(f"Tarea: {task} | Orden: {order} | Resultado: {result} | Error: {error}")


        if result:
            mensaje_bruto = f"{task.capitalize()}: {result}"
            try:
                prompt_text = enchulador_template.format(mensaje=mensaje_bruto)
                prompt = HumanMessage(content=prompt_text)
                response = llm.invoke([prompt])
                friendly_text = response.content.strip()
                logger.info(f"Mensaje procesado para '{task}': {friendly_text}")
                processed_messages.append(friendly_text)
            except Exception as e:
                logger.exception(f"No se pudo reformular el mensaje para '{task}': {str(e)}")
                processed_messages.append(mensaje_bruto)

    # Retornar solo el formato correcto, sin modificar el state
    return {
        "results": {"aggregator": processed_messages},
        "task_completed": {'aggregator':True}
    }
