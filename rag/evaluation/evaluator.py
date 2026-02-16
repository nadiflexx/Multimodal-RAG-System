from loguru import logger

from rag.evaluation.dataset import GOLDEN_DATASET
from rag.exceptions import RAGException
from rag.pipeline import RAGPipeline


class RetrievalEvaluator:
    """Hit Rate Evaluator.
    Evaluates the performance of the retrieval pipeline by checking if expected
    content is retrieved.
    """

    def __init__(self):
        self.pipeline = RAGPipeline()

        try:
            print("⚙️  Initializing Pipeline for Evaluation...")
            self.vector_db = self.pipeline.run_ingestion("evaluation.pdf")
        except RAGException as e:
            print(f"❌ Critical Error: {e}")
            logger.error(f"Evaluation setup failed: {e}")
            self.vector_db = None

    def run(self):
        """Runs the evaluation.
        Evaluates the retrieval pipeline against a golden dataset.
        """
        if not self.vector_db:
            return

        print("\n🧪 Starting Evaluation: Hit Rate @ 3")
        print("=" * 60)

        hits = 0
        total = len(GOLDEN_DATASET)

        for i, item in enumerate(GOLDEN_DATASET):
            query = item["query"]
            expected = item["expected_content"]

            try:
                retrieved_docs = self.pipeline.run_retrieval(query)
            except RAGException as e:
                print(f"❌ ERROR | Q{i + 1}: {query} → {e}")
                continue

            is_hit = False
            found_in_chunk = -1

            for index, doc in enumerate(retrieved_docs):
                if expected.lower() in doc.page_content.lower():
                    is_hit = True
                    found_in_chunk = index + 1
                    break

            if is_hit:
                hits += 1
                icon = "✅ HIT "
                detail = f"(Found at Rank #{found_in_chunk})"
            else:
                icon = "❌ MISS"
                print(f"{icon} | Q{i + 1}: {query:<40}")
                print(f"   ⚠️ EXPECTED: '{expected}'")
                if retrieved_docs:
                    best_match_content = retrieved_docs[0].page_content
                    print(f"   👀 RECEIVED: '{best_match_content[:150]}...'")

                    if expected.lower().replace(
                        " ", ""
                    ) in best_match_content.lower().replace(" ", ""):
                        print(
                            "   💡 HINT: It matches if we ignore spaces! "
                            "Check your whitespace cleaning."
                        )
                detail = f"(Expected: '{expected}')"

            print(f"{icon} | Q{i + 1}: {query:<40} {detail}")

        score = (hits / total) * 100 if total > 0 else 0
        print("=" * 60)
        print(f"📊 Final Hit Rate Score: {score:.2f}%")

        self._analyze_score(score)

    def _analyze_score(self, score):
        """Analyzes the score and provides feedback."""
        if score == 100:
            print("🏆 PERFECT! Your RAG is fully grounded.")
        elif score >= 80:
            print("🚀 Great performance. Ready for production.")
        elif score >= 60:
            print("⚠️  Acceptable, but needs tuning (Check chunk_size or overlap).")
        else:
            print("🛑 Critical issues. The system cannot find simple answers.")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run()
