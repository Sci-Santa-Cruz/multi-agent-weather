
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# Cargar variables de entorno
load_dotenv(dotenv_path='env')

# Función para combinar diccionarios
def merge_dicts(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError(f"Ambos argumentos deben ser diccionarios. Recibido: {type(dict1)} y {type(dict2)}")
    return {**dict1, **dict2}

def add_history_update(history_old: List[str], history_new: List[str]) -> List[str]:
    return history_old + history_new

# Clase AgentState
class AgentState(TypedDict):
    """
    Representa el estado que fluye a través del agente LangGraph.

    Atributos:
        messages (list[BaseMessage]):
            Lista de mensajes intercambiados entre el sistema y el usuario,
            gestionados automáticamente con el paso de mensajes de LangGraph.
        
        order_task (Optional[List[str]]):
            Lista opcional que representa el orden en el que las tareas identificadas 
            deben ser procesadas o presentadas.

        results (Dict[str, Any]):
            Diccionario que contiene los resultados de varios nodos de tareas.
            Cada clave es el nombre de la tarea, y el valor es el resultado o error.

        error (Optional[str]):
            Mensaje de error general para representar cualquier problema en el flujo del agente.

        tasks_to_do (Dict[str, bool]):
            Diccionario que representa qué tareas han sido identificadas para ejecutar.
            Ejemplo: {"weather": True, "currencies": True, "news": False}

        ready_to_aggregate (bool):
            Indica si todas las tareas esperadas están listas y el paso final de agregación puede ejecutarse.

        history (List[str]):
            Lista para realizar un seguimiento de los nombres de los nodos por los que pasa el flujo.
    """
    
    messages: Annotated[List[BaseMessage], add_messages]  # Mensajes intercambiados
    order_task: Dict[str, Any]  # Orden de las tareas
    error: Annotated[Dict[str, str], merge_dicts]  # Manejo de errores
    results: Annotated[Dict[str, str], merge_dicts]  # Resultados de las tareas
    task_completed: Annotated[Dict[str, str], merge_dicts]  # Tareas completadas
    tasks_to_do: Dict[str, bool]  # Tareas pendientes
    ready_to_aggregate: bool  # Indicador de si está listo para agregarse
    history: Annotated[List[str], add_history_update]  # Historial de nodos procesados

