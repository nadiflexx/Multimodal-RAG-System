# 6. Fase 3: GENERATION (Generación de Respuesta)

---

## Paso 3.1: Formateo de Contexto

```python
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([
        f"[Página {d.metadata.get('page', '?')}]: {d.page_content}"
        for d in docs
    ])

# Resultado:
# [Página 3]: La técnica LIME (Local Interpretable Model-agnostic
# Explanations) permite generar explicaciones locales...
#
# [Página 4]: LIME funciona creando perturbaciones alrededor de
# la instancia a explicar y ajustando un modelo simple...
#
# [Página 5]: Las ventajas de LIME incluyen su independencia
# del modelo subyacente...
```

---

## Paso 3.2: Prompt Engineering (prompts.py)

```python
SYSTEM_PROMPT = """Eres un Asistente Experto en Análisis Documental.
Tu misión es responder preguntas basándote ESTRICTAMENTE en el contexto
proporcionado.

Reglas:
1. Si la respuesta no está en el contexto, di:
   "Lo siento, no tengo información sobre eso en el documento."
2. No inventes información (Alucinación 0).
3. Cita la página de referencia si está disponible.
4. Responde en el mismo idioma de la pregunta.
5. Sé conciso y profesional.
"""

USER_PROMPT = """Contexto:
{context}

Pregunta del usuario:
{question}
"""
```

### Anatomía del Prompt

```text
┌─────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT (Quién eres + Reglas)                         │
│                                                             │
│ Define la PERSONALIDAD y los LÍMITES del asistente:         │
│                                                             │
│ "Eres un Asistente Experto..."                              │
│ → Le dice al LLM QUÉ rol interpretar                        │
│                                                             │
│ "basándote ESTRICTAMENTE en el contexto"                    │
│ → GUARDRAIL contra alucinaciones                            │
│                                                             │
│ "Si la respuesta no está... di: Lo siento..."               │
│ → Le da una SALIDA SEGURA cuando no sabe                    │
│                                                             │
│ "No inventes información"                                   │
│ → Refuerzo del guardrail                                    │
│                                                             │
│ "Cita la página"                                            │
│ → Trazabilidad de la respuesta                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ USER PROMPT (Contexto + Pregunta)                           │
│                                                             │
│ Contexto:                                                   │
│ [Página 3]: La técnica LIME permite...                      │
│ [Página 4]: LIME funciona creando...                        │
│                                                             │
│ Pregunta del usuario:                                       │
│ ¿Qué es LIME?                                               │
│                                                             │
│ → El LLM tiene TODO lo que necesita para responder          │
│ → Si la respuesta no está en el contexto, dirá "no sé"      │
└─────────────────────────────────────────────────────────────┘
```

---

## Paso 3.3: Invocación del LLM (Groq + Llama 3.3)

```python
chain = get_chat_template() | self.llm | StrOutputParser()

response = chain.invoke({
    "context": context_text,
    "question": query
})
```

### ¿Qué es esta sintaxis | (pipe)?

```text
Es la "LCEL" de LangChain (LangChain Expression Language):

  get_chat_template()  → Crea el prompt con las variables
         |
      self.llm         → Envía el prompt a Groq/Llama 3.3
         |
  StrOutputParser()    → Extrae solo el texto de la respuesta

  Es equivalente a:
    prompt = template.format(context=..., question=...)
    llm_response = llm.call(prompt)
    text = llm_response.content
```

---

## ¿Por qué Groq + Llama 3.3 y no OpenAI?

```text
OpenAI GPT-4:
  ✅ Mejor calidad de respuestas
  ❌ ~$30/millón de tokens
  ❌ Latencia: 2-5 segundos

Groq + Llama 3.3 70B:
  ✅ Gratis (con rate limits)
  ✅ Latencia: 0.3-1 segundo (MUCHO más rápido)
  ✅ Calidad muy cercana a GPT-4
  ✅ Modelo open-source
  ❌ Rate limits en el free tier
```

---

## ¿Por qué temperature=0?

```text
Temperature controla la ALEATORIEDAD del LLM:

  temperature=0:   Siempre elige la palabra más probable
                   → Respuestas DETERMINISTAS y PRECISAS
                   → Ideal para: RAG, Q&A, clasificación

  temperature=0.7: Introduce variabilidad
                   → Respuestas más CREATIVAS
                   → Ideal para: escritura, brainstorming

  temperature=1.0: Máxima aleatoriedad
                   → Respuestas impredecibles
                   → Riesgo de incoherencia
```
