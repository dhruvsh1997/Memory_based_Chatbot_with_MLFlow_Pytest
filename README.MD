# 🤖 MLFlow_Pytest_Chatbot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Testing-red.svg)](https://pytest.org/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-purple.svg)](https://groq.com/)

A production-ready chatbot application that combines **Flask web framework**, **Groq's LLaMA 3.3 70B model**, **MLflow experiment tracking**, and **Pytest testing** to deliver a robust, monitored, and tested conversational AI experience.

## 🌟 Key Features

- **🧠 AI-Powered Conversations**: Leverages Groq's LLaMA 3.3 70B Versatile model for intelligent responses
- **📊 MLflow Experiment Tracking**: Comprehensive logging of chat sessions, metrics, and artifacts
- **🧪 Pytest Testing Framework**: Automated testing to ensure reliability and functionality
- **💾 Persistent Chat History**: CSV-based storage for conversation logging
- **🐳 Docker Support**: Containerized deployment with Docker and docker-compose
- **🎨 Modern UI**: Clean, responsive chat interface

---

## 🏗️ Project Architecture

```
MLFlow_Pytest_Chatbot/
├── templates/
│   └── index.html          # Chat interface UI
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── app.py                 # Main Flask application
├── test_app.py            # Pytest test suite
├── Dockerfile             # Docker configuration
└── docker-compose.yml     # Multi-container setup
```

---

## 📈 MLflow Integration Deep Dive

### Why MLflow?

MLflow transforms this chatbot from a simple application into a **production-grade ML system** with comprehensive experiment tracking, making it ideal for:

- **Performance Monitoring**: Track response times and token usage
- **Conversation Analytics**: Analyze chat patterns and model performance
- **Reproducibility**: Maintain detailed logs of all interactions
- **Model Comparison**: Compare different prompts or model configurations

### MLflow Components in Action

#### 🎯 **Experiment Organization**
```python
mlflow.set_experiment("Groq_Chatbot_Experiments")
```
- Creates a dedicated experiment space for all chatbot runs
- Organizes multiple chat sessions under a single experiment umbrella

#### 📊 **Comprehensive Metrics Tracking**
Every chat interaction logs:

| Metric | Description | Purpose |
|--------|-------------|---------|
| `response_time` | Time taken to generate response | Performance monitoring |
| `input_tokens` | Tokens in user question | Cost analysis |
| `output_tokens` | Tokens in AI response | Response complexity tracking |
| `total_tokens` | Combined token usage | Total resource consumption |

#### 🔧 **Parameter Logging**
```python
mlflow.log_param("question", user_msg)
mlflow.log_param("answer", answer)
mlflow.log_param("model", "llama-3.3-70b-versatile")
```
- Captures the context of each conversation
- Enables search and filtering of specific interactions

#### 📁 **Artifact Management**
MLflow automatically saves four types of artifacts for each chat session:

1. **📋 Current Q&A Pair** (`current_qa/`)
   ```json
   {
     "timestamp": "2025-09-02 14:30:15",
     "question": "What is machine learning?",
     "answer": "Machine learning is...",
     "response_time": 1.23,
     "input_tokens": 15,
     "output_tokens": 150,
     "total_tokens": 165
   }
   ```

2. **🔐 Credentials Info** (`credentials/`)
   ```json
   {
     "api_key": "gsk_abcd...",
     "model": "llama-3.3-70b-versatile"
   }
   ```

3. **📜 Complete Chat History** (`chat_history/`)
   - Full CSV export of all conversations
   - Enables historical analysis and reporting

4. **🤖 Model Information** (`model_info/`)
   ```json
   {
     "model_name": "llama-3.3-70b-versatile",
     "provider": "Groq",
     "api_endpoint": "https://api.groq.com"
   }
   ```

### MLflow Dashboard Benefits

Access the MLflow UI at `http://localhost:5002` to:

- **📈 Visualize Performance Trends**: Response time distribution, token usage patterns
- **🔍 Search Conversations**: Find specific questions or topics
- **📊 Generate Reports**: Export data for business intelligence
- **🔄 Compare Sessions**: Analyze different conversation patterns

---

## 🧪 Pytest Testing Strategy

### Testing Philosophy

The Pytest integration ensures **reliability** and **quality assurance** by validating that the chatbot can:
- ✅ Handle user inputs correctly
- ✅ Generate meaningful responses
- ✅ Maintain API connectivity
- ✅ Return proper response formats

### Test Structure

#### **Fixture Setup**
```python
@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c
```
- Creates an isolated testing environment
- Ensures tests don't interfere with production data

#### **Core Functionality Test**
```python
def test_chat_with_message(client):
    resp = client.post('/chat', json={"message": "Hello, how are you?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0
```

**What this test validates:**
- 🌐 **HTTP Endpoint Functionality**: POST request handling
- 📡 **API Integration**: Groq API connectivity
- 📝 **Response Format**: JSON structure validation
- 💬 **Content Quality**: Non-empty response generation

### Testing Benefits

| Benefit | Description |
|---------|-------------|
| **🛡️ Quality Assurance** | Catches integration issues before deployment |
| **🔄 Continuous Integration** | Automated testing in CI/CD pipelines |
| **📊 Reliability Metrics** | Quantifiable success rates |
| **🐛 Bug Prevention** | Early detection of API or code issues |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (optional)
- Groq API Key

### Method 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MLFlow_Pytest_Chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   echo "MLFLOW_TRACKING_URI=./mlruns" >> .env
   echo "SECRET_KEY=your_secret_key_here" >> .env
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

### Method 2: Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build
```

**Service Endpoints:**
- 🌐 **Chatbot Application**: `http://localhost:5000`
- 📊 **MLflow Dashboard**: `http://localhost:5002`

---

## 🧪 Running Tests

### Execute Test Suite
```bash
# Run all tests
pytest test_app.py -v

# Run with coverage
pytest test_app.py --cov=app

# Run specific test
pytest test_app.py::test_chat_with_message -v
```

### Expected Test Output
```
======================== test session starts ========================
collected 1 item

test_app.py::test_chat_with_message PASSED             [100%]

======================== 1 passed in 2.34s ========================
```

---

## 📊 MLflow Experiment Tracking

### Starting MLflow UI
```bash
# Local MLflow server
mlflow ui --backend-store-uri ./mlruns --port 5002

# Or access via Docker
docker-compose up mlflow
```

### Key MLflow Features

#### **🎛️ Experiment Dashboard**
- View all chat sessions in chronological order
- Filter by parameters (questions, model, timestamps)
- Compare performance metrics across sessions

#### **📈 Metrics Visualization**
- Response time trends
- Token usage analytics
- Performance distribution charts

#### **🗂️ Artifact Management**
- Download conversation logs
- Access detailed session information
- Export data for external analysis

#### **🔍 Search & Filter**
- Find specific conversations by keywords
- Filter by date ranges or performance metrics
- Export filtered datasets

---

## 🎯 Architecture Benefits

### MLflow + Pytest Synergy

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **MLflow** | Production monitoring | 📊 Data-driven insights into chatbot performance |
| **Pytest** | Quality assurance | 🛡️ Confidence in deployment reliability |
| **Combined** | DevOps excellence | 🚀 Production-ready ML application |

### Production Readiness Features

- **🔄 Error Handling**: Graceful degradation when MLflow or API fails
- **📱 Responsive Design**: Mobile-friendly chat interface
- **⚡ Performance Tracking**: Real-time metrics collection
- **🔒 Security**: API key masking in logs
- **📈 Scalability**: Docker-based deployment ready for cloud

---

## 🎨 User Interface

The chatbot features a modern, intuitive interface with:

- **💙 Clean Design**: Professional blue theme with smooth animations
- **💬 Message Bubbles**: Distinct styling for user and bot messages
- **⏳ Loading Animation**: Engaging dot animation during response generation
- **📱 Responsive Layout**: Works seamlessly across devices
- **⌨️ Keyboard Support**: Send messages with Enter key

---

## 📁 Data Storage

### Chat History Management
- **Format**: CSV files for easy analysis
- **Structure**: Timestamp, question, answer, performance metrics
- **Persistence**: Automatic saving after each interaction
- **Integration**: Seamless MLflow artifact logging

### MLflow Experiment Data
- **Local Storage**: `./mlruns` directory
- **Organization**: Experiments → Runs → Artifacts
- **Backup**: Easy to backup and migrate
- **Sharing**: Export capabilities for team collaboration

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API authentication key | Required |
| `MLFLOW_TRACKING_URI` | MLflow storage location | `./mlruns` |
| `SECRET_KEY` | Flask session security key | `secret_key` |

### Customization Points

- **🎨 UI Styling**: Modify CSS in `templates/index.html`
- **🤖 System Prompt**: Adjust chatbot personality in `GroqChatbot.__init__()`
- **📊 Metrics**: Add custom metrics in the MLflow logging section
- **🧪 Tests**: Extend test coverage in `test_app.py`

---

## 🚀 Deployment

### Local Development
```bash
python app.py
# Access: http://localhost:5000
```

### Docker Production
```bash
docker-compose up -d
# Chatbot: http://localhost:5000
# MLflow: http://localhost:5002
```

### Cloud Deployment
- Compatible with AWS, GCP, Azure
- Requires external MLflow tracking server for production
- Environment variable configuration for cloud APIs

---

## 🔍 Monitoring & Analytics

### Real-time Monitoring
- **Response Time Tracking**: Identify performance bottlenecks
- **Token Usage Analysis**: Cost optimization insights
- **Error Rate Monitoring**: System health indicators
- **User Interaction Patterns**: Usage analytics

### Business Intelligence
- **📈 Usage Trends**: Daily/weekly conversation volumes
- **🎯 Popular Topics**: Most frequently asked questions
- **⚡ Performance Metrics**: Average response times and token efficiency
- **📊 Export Capabilities**: Data export for external BI tools

---

## 🤝 Contributing

### Development Workflow
1. **🔧 Setup**: Install dependencies and configure environment
2. **🧪 Test**: Run pytest suite before making changes
3. **📊 Monitor**: Check MLflow for performance impacts
4. **🔄 Iterate**: Use tracking data to guide improvements

### Adding New Features
- Extend pytest coverage for new functionality
- Add relevant MLflow metrics for monitoring
- Update documentation and examples

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🎉 Conclusion

This chatbot project demonstrates **modern ML engineering practices** by combining:

- **🤖 Advanced AI**: State-of-the-art language model integration
- **📊 Experiment Tracking**: Production-grade monitoring with MLflow
- **🧪 Quality Assurance**: Automated testing with Pytest
- **🐳 DevOps**: Containerized deployment ready for any environment

The integration of MLflow and Pytest transforms a simple chatbot into a **professional-grade ML application** suitable for production deployment, continuous monitoring, and iterative improvement.

---
MLFLOW UI Port Switch Code : mlflow ui --port 5002

*Built with ❤️ using Flask, MLflow, Pytest, and Groq LLaMA 3.3 70B*

