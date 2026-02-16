# Flujo Final Completo (Todo junto)

```text
┌─────────────────────────────────────────────────────────────┐
│  USUARIO ESCRIBE: "¿Y dónde aparece eso?"                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    ┌─────▼──────┐
                    │  ROUTER    │ → SEARCH ✅ (no es saludo)
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ NORMALIZE  │ → "y donde aparece eso"
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  CACHE     │ → MISS ❌ (primera vez)
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ CONTEXT    │ → "¿Dónde aparece LIME?"
                    │ UALIZER    │   (resolvió "eso" → "LIME")
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │   HyDE     │ → "LIME aparece en la sección
                    │            │    de explicabilidad donde..."
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  HYBRID    │ → 10 candidatos
                    │  SEARCH    │   (5 vector + 5 BM25)
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ RERANKER   │ → 3 más relevantes
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  PARENT    │ → 5-7 páginas completas
                    │ EXPANSION  │   (con vecinos)
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │   LLM      │ → "Según el documento
                    │ GENERATION │    (página 3), LIME
                    │            │    aparece en la sección..."
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │ CACHE SAVE │ → Guardar para futuro
                    │ + MEMORY   │ → Añadir al historial
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  RESPONSE  │ → Se muestra en Streamlit
                    └────────────┘
```
