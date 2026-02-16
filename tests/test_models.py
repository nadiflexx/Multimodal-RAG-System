from rag.models import Intent, PipelineState, RetrievalResult


def test_intent_enum():
    assert Intent.SEARCH == "SEARCH"
    assert Intent.GREETING == "GREETING"


def test_retrieval_result():
    res = RetrievalResult(query_used="test")
    assert res.documents == []
    assert res.query_used == "test"


def test_pipeline_state():
    state = PipelineState()
    assert not state.is_initialized
