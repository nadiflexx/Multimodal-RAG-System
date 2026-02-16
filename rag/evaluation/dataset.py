"""
Define a golden dataset to evaluate the RAG pipeline.
"""

GOLDEN_DATASET = [
    {
        "query": "Cual es el objetivo del proyecto?",
        "expected_content": "Programar un asistente virtual en Python",
    },
    {
        "query": "Que libreria principal se debe utilizar?",
        "expected_content": "utilizar la librería nltk",
    },
    {
        "query": "Como responde si no sabe la respuesta?",
        "expected_content": "no tengo información sobre eso",
    },
    {
        "query": "Cual es el nombre del archivo que se carga al inicio?",
        "expected_content": "Clase0_NLP_dataset.txt",
    },
    {
        "query": "Como se introduce las preguntas?",
        "expected_content": "consola (terminal)",
    },
]
