# 1. ¿Qué es RAG y por qué existe?

## El Problema Fundamental

Los LLMs (como GPT-4, Llama 3.3) tienen dos limitaciones críticas:

```text
┌─────────────────────────────────────────────────────────┐
│  PROBLEMA 1: Conocimiento Congelado                     │
│                                                         │
│  El LLM fue entrenado con datos hasta fecha X.          │
│  No sabe nada sobre TU documento, TU empresa,           │
│  TU proyecto específico.                                │
│                                                         │
│  PROBLEMA 2: Alucinaciones                              │
│                                                         │
│  Si le preguntas algo que no sabe, INVENTA una          │
│  respuesta que PARECE correcta pero es falsa.           │
└─────────────────────────────────────────────────────────┘
```

## La Solución: RAG

**Retrieval Augmented Generation = Buscar primero, generar después.**

```text
SIN RAG:
  Usuario: "¿Qué dice el documento sobre LIME?"
  LLM: "LIME es una técnica de..." ← Inventa basándose en conocimiento general

CON RAG:
  Usuario: "¿Qué dice el documento sobre LIME?"
  Sistema: 1. Busca en el documento → Encuentra fragmentos relevantes
           2. Le pasa esos fragmentos al LLM como contexto
  LLM: "Según el documento (página 5), LIME se define como..." ← Respuesta fundamentada
```

## Analogía Simple

```text
RAG es como un estudiante en un examen con apuntes permitidos:

  1. Lee la pregunta del examen (query del usuario)
  2. Busca en sus apuntes los fragmentos relevantes (retrieval)
  3. Redacta la respuesta usando esos fragmentos (generation)

  Sin apuntes → Inventa o dice "no sé"
  Con apuntes → Respuesta precisa y citable
```
