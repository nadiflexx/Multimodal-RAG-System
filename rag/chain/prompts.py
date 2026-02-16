"""
Centralized Prompt Templates for the RAG System.

This module contains all the prompts used across different components:
- Generation (Chat)
- Contextualization (History resolution)
- Semantic Cache (Intent normalization)
- Semantic Router (Intent classification)
- HyDE (Hypothetical Document Generation)
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate

# ═══════════════════════════════════════════════════════
# 1. GENERATION PROMPTS (Chat)
# ═══════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an Expert Document Analysis Assistant.
Your mission is to answer questions strictly based on the provided context.

Rules:
1. If the answer is not in the context, say: "I'm sorry, I don't have information about that in the document."
2. Do not invent information (Zero Hallucination).
3. Cite the reference page if available in the metadata.
4. Answer in the same language as the user's question.
5. Be concise and professional.
6. If the user asks for a long list (like bibliography or index), summarize key points or show the first few items and mention the page range where the rest can be found. DO NOT try to output lists longer than 10 items.
"""

USER_PROMPT = """Context:
{context}

User Question:
{question}
"""

# ═══════════════════════════════════════════════════════
# 2. CONTEXTUALIZER PROMPTS
# ═══════════════════════════════════════════════════════

CONTEXTUALIZER_TEMPLATE = """
Dada una conversación y una pregunta de seguimiento, tu ÚNICA tarea 
es hacer explícitas las referencias implícitas (pronombres como "eso", "ahí", "él").

REGLAS DE ORO:
1. Si la pregunta NO tiene pronombres ni referencias ambiguas, DEVUÉLVELA EXACTAMENTE IGUAL.
2. NO asumas que el tema sigue siendo el mismo si la pregunta introduce un concepto nuevo (ej: "índice", "bibliografía", "conclusiones").
3. NO añadas información, definiciones ni explicaciones.
4. Solo modifica si es gramaticalmente necesario para que la pregunta tenga sentido por sí sola.

EJEMPLOS:

Historial: User preguntó sobre LIME.
Input: "¿En qué página está?" 
Output: "¿En qué página está LIME?" (Correcto: referencia ambigua resuelta)

Historial: User preguntó sobre la bibliografía.
Input: "tabla de contenidos"
Output: "tabla de contenidos" (Correcto: es un nuevo tema, NO mezclar con bibliografía)

Historial: User preguntó sobre Random Forest.
Input: "¿Y qué ventajas tiene?"
Output: "¿Qué ventajas tiene Random Forest?" (Correcto)

Historial: User preguntó sobre los autores.
Input: "resumen"
Output: "resumen" (Correcto: cambio de tema)

Historial:
{chat_history}

Pregunta Original: {question}

Pregunta Contextualizada (o la original si no hay ambigüedad):
"""

# ═══════════════════════════════════════════════════════
# 3. SEMANTIC CACHE PROMPTS (Intent Normalization)
# ═══════════════════════════════════════════════════════

CACHE_INTENT_TEMPLATE = """Extrae el TEMA CENTRAL de esta pregunta en 2-4 palabras.

INSTRUCCIONES:
- Responde SOLO con sustantivos y el tema, sin verbos de acción.
- NO uses verbos como "definicion", "ubicacion", "listado", "resumen".
- Extrae directamente el SUJETO de la pregunta.
- Si hay un nombre propio o técnico, mantenlo exacto.

EJEMPLOS:
"¿qué es LIME?" → "lime"
"explícame qué es LIME" → "lime"
"¿dónde aparece LIME?" → "lime"
"¿en qué página está LIME?" → "lime"
"apartado de agradecimientos" → "agradecimientos"
"muestra los agradecimientos" → "agradecimientos"
"¿qué dice el apartado de agradecimientos?" → "agradecimientos"
"¿cuáles son las técnicas de explicabilidad?" → "tecnicas explicabilidad"
"explica las técnicas de explicabilidad" → "tecnicas explicabilidad"
"resume el capítulo de explicabilidad" → "explicabilidad"
"¿qué es la inteligencia artificial explicable?" → "inteligencia artificial explicable"
"háblame de XAI" → "xai"
"¿qué es XAI?" → "xai"
"diferencias entre LIME y SHAP" → "lime shap"
"compara LIME con SHAP" → "lime shap"
"¿quién es el tutor del proyecto?" → "tutor proyecto"
"¿qué regulación se menciona?" → "regulacion"
"¿qué dice el RGPD?" → "rgpd"
"sector financiero" → "sector financiero"
"¿qué modelos de caja negra existen?" → "modelos caja negra"
"bibliografía" → "bibliografia"
"índice del documento" → "indice"
"conclusiones del proyecto" → "conclusiones"

Pregunta: {query}

Tema central:"""


# ═══════════════════════════════════════════════════════
# 4. SEMANTIC ROUTER PROMPTS (Intent Classification)
# ═══════════════════════════════════════════════════════

ROUTER_TEMPLATE = """
Tu tarea es clasificar la intención del usuario en una de estas 3 categorías:

1. "SEARCH": El usuario está preguntando sobre información, datos, el documento
2. "GREETING": El usuario está saludando/despidiéndose(hola, adios, buenos dias)
3. "OFF_TOPIC": El usuario está insultando, hablando de temas personales...

Pregunta: {question}

Responde SOLO con la palabra clave (SEARCH, GREETING, u OFF_TOPIC).
"""

# ═══════════════════════════════════════════════════════
# 5. HYDE PROMPTS (Hypothetical Document)
# ═══════════════════════════════════════════════════════

HYDE_TEMPLATE = """
Tu tarea es generar un fragmento hipotético de un documento técnico que responda a la pregunta.
Usa terminología precisa. No inventes datos específicos (fechas, nombres propios falsos), usa generalidades técnicas.

Pregunta: {question}

Fragmento Hipotético:
"""

# ═══════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════

TemplateType = Literal["chat", "contextualizer", "cache", "router", "hyde"]


def get_template(template_type: TemplateType) -> ChatPromptTemplate:
    """
    Factory function to retrieve the appropriate prompt template.

    Args:
        template_type: The type of template required.
                       Options: "chat", "contextualizer", "cache", "router", "hyde"

    Returns:
        A configured ChatPromptTemplate instance.
    """
    if template_type == "chat":
        return ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
        )

    elif template_type == "contextualizer":
        return ChatPromptTemplate.from_template(CONTEXTUALIZER_TEMPLATE)

    elif template_type == "cache":
        return ChatPromptTemplate.from_template(CACHE_INTENT_TEMPLATE)

    elif template_type == "router":
        return ChatPromptTemplate.from_template(ROUTER_TEMPLATE)

    elif template_type == "hyde":
        return ChatPromptTemplate.from_template(HYDE_TEMPLATE)

    else:
        raise ValueError(f"Unknown template type: {template_type}")
