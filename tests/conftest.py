import pytest
from langchain.schema import Document

@pytest.fixture
def sample_docs():
    return [Document(page_content="Sample text for testing")]