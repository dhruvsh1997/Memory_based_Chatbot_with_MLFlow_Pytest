import pytest
from MLFlow_Pytest_Chatbot.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_chat_with_message(client):
    # Test if the chat endpoint can provide an answer
    resp = client.post('/chat', json={"message": "Hello, how are you?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0  # Ensure there's a non-empty response