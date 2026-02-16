"""
RAG Assistant - Entry Point

Mode CLI:     uv run main.py --cli
Mode Web:     streamlit run main.py
"""

import sys

from rag.cli import run
from rag.pipeline import RAGPipeline


def main_cli():
    """Executes the RAG pipeline in CLI mode (legacy)."""

    pipeline = RAGPipeline()
    pipeline.run_ingestion("data.pdf")

    print("\n💬 Chat Ready! Type 'exit' to quit.")
    print("-" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "q"):
                break

            response, docs = pipeline.run_conversation_flow(query)
            print(f"\n🤖 Assistant:\n{response}")

            if docs:
                print("\n📚 Sources:")
                for i, d in enumerate(docs):
                    page = d.metadata.get("page", "?")
                    preview = d.page_content[:50].replace("\n", " ")
                    print(f"  - Ref {i + 1} (Page {page}): {preview}...")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n👋 Goodbye!")


def main_streamlit():
    """Executes the RAG pipeline in Streamlit mode."""
    run()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        main_cli()
    else:
        main_streamlit()
