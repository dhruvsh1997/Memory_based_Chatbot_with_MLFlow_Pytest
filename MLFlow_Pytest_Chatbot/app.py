# Import necessary libraries
import os
import time
import mlflow
import mlflow.pyfunc
from flask import Flask, render_template, request, jsonify
from groq import Groq
import dotenv
import pandas as pd
import tempfile
import json

# Load environment variables from .env file
dotenv.load_dotenv()

# ==================== STEP 1: SET UP MLFLOW TRACKING ====================
# MLflow is a tool for tracking machine learning experiments
# We'll set up where to store the tracking information (a local directory by default)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

# Try to initialize MLflow tracking
try:
    # Set the tracking URI (where MLflow will store experiment data)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get an experiment named "Groq_Chatbot_Experiments"
    # An experiment is a collection of related runs (chat sessions in our case)
    mlflow.set_experiment("Groq_Chatbot_Experiments")
    
    # Flag to indicate MLflow is available
    MLFLOW_AVAILABLE = True
    print(f"MLflow tracking initialized at: {MLFLOW_TRACKING_URI}")
except Exception as e:
    # If MLflow setup fails, print a warning and continue without MLflow
    print(f"Warning: MLflow tracking not available: {str(e)}")
    MLFLOW_AVAILABLE = False

# ==================== STEP 2: SET UP API KEY ====================
# Get the Groq API key from environment variables
# This key is needed to access the Groq language model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
breakpoint()
# ==================== STEP 3: DEFINE THE CHATBOT CLASS ====================
class GroqChatbot:
    """
    A chatbot that uses the Groq LLM API to generate responses.
    This class handles the conversation with the AI model.
    """
    
    def __init__(self, system_prompt="You are a helpful AI assistant."):
        """
        Initialize the chatbot with a system prompt.
        
        Args:
            system_prompt (str): Instructions that guide the AI's behavior
        """
        self.system_prompt = system_prompt  # Store the system prompt
        self.client = None  # Will hold the Groq API client
        self.message_history = []  # Will store the conversation history
        
    def initialize_client(self, api_key):
        """
        Initialize the Groq API client.
        
        Args:
            api_key (str): The API key for accessing Groq
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if api_key:
            # Create a Groq client with the provided API key
            self.client = Groq(api_key=api_key)
            
            # Start the conversation with the system prompt
            self.message_history = [{"role": "system", "content": self.system_prompt}]
            return True
        return False
    
    def predict(self, question):
        """
        Generate a response to the user's question.
        
        Args:
            question (str): The user's question
            
        Returns:
            dict: A dictionary containing the response and metrics
        """
        # Check if the client was initialized
        if not self.client:
            return {"error": "Client not initialized"}
            
        try:
            # Add the user's question to the conversation history
            self.message_history.append({"role": "user", "content": question})
            
            # Record the start time to measure response time
            start_time = time.time()
            
            # Send the conversation history to the Groq API and get a response
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # The model to use
                messages=self.message_history  # The conversation so far
            )
            
            # Calculate how long the response took
            response_time = time.time() - start_time
            
            # Extract the answer from the response
            final_answer = response.choices[0].message.content
            
            # Add the AI's response to the conversation history
            self.message_history.append({"role": "assistant", "content": final_answer})
            
            # Return the answer and metrics
            return {
                "answer": final_answer,
                "response_time": response_time,
                "input_tokens": response.usage.prompt_tokens,  # Tokens in the question
                "output_tokens": response.usage.completion_tokens,  # Tokens in the answer
                "total_tokens": response.usage.total_tokens  # Total tokens used
            }
        except Exception as e:
            # If something goes wrong, return the error message
            return {"error": str(e)}

# ==================== STEP 4: CHAT HISTORY MANAGER ====================
class ChatHistoryManager:
    """
    Manages the chat history by saving it to a CSV file.
    This allows us to maintain a record of all conversations.
    """
    
    def __init__(self, csv_path="chat_history.csv"):
        """
        Initialize the chat history manager.
        
        Args:
            csv_path (str): Path to the CSV file for storing chat history
        """
        self.csv_path = csv_path  # Path to the CSV file
        self.history_df = self._load_history()  # Load existing history if available
        
    def _load_history(self):
        """
        Load chat history from the CSV file.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the chat history
        """
        if os.path.exists(self.csv_path):
            try:
                # If the file exists, read it into a DataFrame
                return pd.read_csv(self.csv_path)
            except Exception as e:
                # If there's an error reading the file, print a warning
                print(f"Error loading chat history: {e}")
                # Return an empty DataFrame with the correct columns
                return pd.DataFrame(columns=["timestamp", "question", "answer", 
                                           "response_time", "input_tokens", 
                                           "output_tokens", "total_tokens"])
        else:
            # If the file doesn't exist, return an empty DataFrame
            return pd.DataFrame(columns=["timestamp", "question", "answer", 
                                       "response_time", "input_tokens", 
                                       "output_tokens", "total_tokens"])
    
    def add_entry(self, timestamp, question, answer, response_time, input_tokens, output_tokens, total_tokens):
        """
        Add a new entry to the chat history.
        
        Args:
            timestamp (str): When the conversation happened
            question (str): The user's question
            answer (str): The AI's answer
            response_time (float): How long it took to generate the answer
            input_tokens (int): Number of tokens in the question
            output_tokens (int): Number of tokens in the answer
            total_tokens (int): Total number of tokens used
        """
        # Create a new entry as a dictionary
        new_entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
        
        # Add the new entry to the DataFrame
        self.history_df = pd.concat([self.history_df, pd.DataFrame([new_entry])], ignore_index=True)
        
        # Save the updated DataFrame to the CSV file
        self._save_history()
    
    def _save_history(self):
        """
        Save the chat history to the CSV file.
        """
        try:
            # Save the DataFrame to a CSV file without the index
            self.history_df.to_csv(self.csv_path, index=False)
        except Exception as e:
            # If there's an error saving the file, print a warning
            print(f"Error saving chat history: {e}")
    
    def get_history(self):
        """
        Get the chat history.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the chat history
        """
        return self.history_df

# ==================== STEP 5: CREATE FLASK APP ====================
def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application
    """
    # Create a Flask app
    app = Flask(__name__)
    
    # Set a secret key for the app (used for session management)
    app.secret_key = os.getenv("SECRET_KEY", "secret_key")
    
    # Initialize the chatbot
    chatbot = GroqChatbot(system_prompt="You are a helpful AI assistant.")
    
    # If we have an API key, initialize the chatbot client
    if GROQ_API_KEY:
        chatbot.initialize_client(GROQ_API_KEY)
    
    # Initialize the chat history manager
    history_manager = ChatHistoryManager()
    
    # Define the home page route
    @app.route("/")
    def index():
        # Render the HTML template for the chat interface
        return render_template("index.html")
    
    # Define the chat route (where messages are sent)
    @app.route("/chat", methods=["POST"])
    def chat():
        # Get the user's message from the request
        user_msg = request.json.get("message")
        
        # Check if a message was provided
        if not user_msg:
            return jsonify({"error": "No message provided"}), 400
            
        # Get a response from the chatbot
        result = chatbot.predict(user_msg)
        
        # Check if there was an error
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Extract the response data
        answer = result["answer"]
        response_time = result["response_time"]
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        total_tokens = result["total_tokens"]
        
        # Add the conversation to the history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        history_manager.add_entry(
            timestamp=timestamp,
            question=user_msg,
            answer=answer,
            response_time=response_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
        
        # Log to MLflow if it's available
        if MLFLOW_AVAILABLE:
            try:
                # Start a new MLflow run
                # A run represents a single execution of the code (in our case, one chat session)
                with mlflow.start_run(run_name="chat_session"):
                    # ==================== LOG PARAMETERS ====================
                    # Parameters are input values that don't change during the run
                    # In our case, we log the question, answer, and model name
                    
                    # Log the user's question
                    mlflow.log_param("question", user_msg)
                    
                    # Log the AI's answer
                    mlflow.log_param("answer", answer)
                    
                    # Log the model name
                    mlflow.log_param("model", "llama-3.3-70b-versatile")
                    
                    # ==================== LOG METRICS ====================
                    # Metrics are numerical values that can change during the run
                    # In our case, we log response time and token counts
                    
                    # Log how long it took to generate the response
                    mlflow.log_metric("response_time", response_time)
                    
                    # Log the number of tokens in the question
                    mlflow.log_metric("input_tokens", input_tokens)
                    
                    # Log the number of tokens in the answer
                    mlflow.log_metric("output_tokens", output_tokens)
                    
                    # Log the total number of tokens used
                    mlflow.log_metric("total_tokens", total_tokens)
                    
                    # ==================== LOG ARTIFACTS ====================
                    # Artifacts are files associated with a run
                    # We'll log several files: credentials, chat history, current Q&A, and model info
                    
                    # Log credentials as a JSON file (with masked API key for security)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        # Create a dictionary with credential information
                        credentials = {
                            # Only show the first 8 characters of the API key for security
                            "api_key": GROQ_API_KEY[:8] + "..." if GROQ_API_KEY else "Not provided",
                            "model": "llama-3.3-70b-versatile"
                        }
                        # Write the credentials to the temporary file
                        json.dump(credentials, f)
                        # Log the file as an artifact in the "credentials" directory
                        mlflow.log_artifact(f.name, "credentials")
                    
                    # Log the entire chat history as a CSV file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        # Get the chat history DataFrame
                        history_df = history_manager.get_history()
                        # Write the DataFrame to a CSV file
                        history_df.to_csv(f.name, index=False)
                        # Log the file as an artifact in the "chat_history" directory
                        mlflow.log_artifact(f.name, "chat_history")
                    
                    # Log the current question-answer pair as a JSON file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        # Create a dictionary with the current Q&A
                        qa_pair = {
                            "timestamp": timestamp,
                            "question": user_msg,
                            "answer": answer,
                            "response_time": response_time,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens
                        }
                        # Write the Q&A to the temporary file with nice formatting
                        json.dump(qa_pair, f, indent=2)
                        # Log the file as an artifact in the "current_qa" directory
                        mlflow.log_artifact(f.name, "current_qa")
                    
                    # Log model information as a JSON file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        # Create a dictionary with model information
                        model_info = {
                            "model_name": "llama-3.3-70b-versatile",
                            "provider": "Groq",
                            "api_endpoint": "https://api.groq.com"
                        }
                        # Write the model info to the temporary file with nice formatting
                        json.dump(model_info, f, indent=2)
                        # Log the file as an artifact in the "model_info" directory
                        mlflow.log_artifact(f.name, "model_info")
                    
            except Exception as e:
                # If there's an error with MLflow, print a warning
                print(f"Warning: Could not log to MLflow: {str(e)}")
            
        # Return the AI's response to the user
        return jsonify({"response": answer})
    
    return app

# ==================== STEP 6: RUN THE APP ====================
if __name__ == "__main__":
    # Create the Flask app
    app = create_app()
    
    # Run the app on all available network interfaces at port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)

# pytest test_app.py -v
# mlflow ui --port 5002 