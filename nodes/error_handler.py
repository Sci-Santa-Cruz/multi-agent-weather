
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from core.agent_state import AgentState  # Adjust if needed
from langchain_openai import ChatOpenAI

# ----- Configurar logging -----
from utils.logging import setup_logging

# Initialize logger using the setup_logging function
logger = setup_logging()

# ----- Cargar variables de entorno -----
load_dotenv(dotenv_path='env')

# Global LLM instance
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# ----- Error Interpretation Prompt -----
error_handler_template = PromptTemplate(
    input_variables=["error", "original_text"],
    template="""
Eres un asistente experto en interpretar errores de sistemas que consultan datos sobre clima, noticias y divisas.

Mensaje original del usuario:
"{original_text}"

Mensaje de error del sistema:
"{error}"

1. Si hay nombres de ciudades, países o monedas abreviados (como 'UK', 'US', 'EUR'), proporciónalos en su forma completa y clara.
2. Genera una explicación amigable del error para el usuario.
3. Sugiere una alternativa. Por ejemplo, si no se puede obtener el clima, sugiere obtener noticias o divisas, y viceversa.
reiterando que vuleva a hacer la peticion con la recomenacion asociada
Devuelve solo el texto final para el usuario, no incluyas explicaciones adicionales ni estructuras.
"""
)

# ----- Error Handler Node -----
def error_handler(state: AgentState) -> AgentState:
    """
    Uses LLM to transform technical error messages into user-friendly suggestions.

    Parameters:
        state (AgentState): The shared graph state including error and last message.

    Returns:
        dict: {"results": {...}, "task_completed": {...}, "error": {...}}
    """

    # Añadir trazabilidad del nodo
    state.setdefault("history", []).append("task_error")
    try:
        user_input = state["messages"][-1].content if state.get("messages") else ""

        errores = state.get("error", {})
        nodo_source = next(iter(errores), "desconocido")  # Tomamos la primera clave
        raw_error = errores.get(nodo_source, "Error no especificado")

        logger.info(f"Procesando error desde el nodo '{nodo_source}': {raw_error}")

        prompt = error_handler_template.format(
            error=raw_error,
            original_text=user_input
        )
        logger.debug(f"Prompt generado para el LLM:\n{prompt}")
        response = llm.invoke([HumanMessage(content=prompt)])
        friendly_message = response.content.strip()

        # Remover la clave procesada
        errores.pop(nodo_source, None)

        logger.info(f"Mensaje amigable generado: {friendly_message}")
        return {
            "results": {nodo_source: [friendly_message]},
            "task_completed": {nodo_source: True},
            "error": { "error": errores } # Retornamos el dict sin la clave ya procesada
        }

    except Exception as e:
        logger.exception("Error en el manejador de errores")
        fallback_message = f"No se pudo procesar el error automáticamente. Detalles: {str(e)}"
        return {
            "results": {"error": [fallback_message]},
            "task_completed": {'error': True}
        }
