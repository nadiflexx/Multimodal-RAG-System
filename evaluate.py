"""
RAG Evaluation

Runs the retrieval evaluation for the RAG pipeline.
"""

from rag.evaluation.evaluator import RetrievalEvaluator

if __name__ == "__main__":
    eval = RetrievalEvaluator()
    eval.run()
