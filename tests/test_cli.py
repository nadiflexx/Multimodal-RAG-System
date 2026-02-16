from unittest.mock import MagicMock, mock_open, patch  # FIX: Añadido mock_open

from rag.cli import initialize_pipeline, process_user_query, save_uploaded_file


@patch("rag.cli.st")
@patch("rag.cli.RAGPipeline")
def test_initialize_pipeline(mock_pipeline, mock_st):
    # Setup session state mock
    mock_st.session_state = MagicMock()
    mock_st.session_state.document_loaded = False

    success = initialize_pipeline("test.pdf", "semantic", {})

    assert success is True
    assert mock_st.session_state.document_loaded is True
    mock_pipeline.return_value.run_ingestion.assert_called()


@patch("rag.cli.st")
def test_process_user_query(mock_st):
    mock_st.session_state = MagicMock()
    mock_st.session_state.messages = []

    # Mock pipeline in session
    mock_pipe = MagicMock()
    mock_pipe.run_conversation_flow.return_value = ("Response", [])
    mock_st.session_state.pipeline = mock_pipe

    process_user_query("Hello")

    assert len(mock_st.session_state.messages) == 2  # User + AI
    mock_pipe.run_conversation_flow.assert_called_with("Hello")


@patch("rag.cli.settings")
def test_save_uploaded_file(mock_settings):
    mock_file = MagicMock()
    mock_file.name = "test.pdf"
    mock_file.getbuffer.return_value = b"content"

    mock_settings.DATA_DIR = MagicMock()
    mock_settings.DATA_DIR.__truediv__.return_value = MagicMock()

    with patch("builtins.open", mock_open()):
        path = save_uploaded_file(mock_file)
        assert path is not None
