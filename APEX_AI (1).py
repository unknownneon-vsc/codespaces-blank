import os
import requests
import logging  # Add logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dqn_agent import DQNAgent
import gym
import numpy as np
import hashlib
import random
from serpapi import GoogleSearch

# GitHub token - Hardcoded for simplicity (not recommended for production)
GITHUB_TOKEN = "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"

# SerpAPI key (you need to sign up at https://serpapi.com/ to get an API key)
SERPAPI_KEY = "0a2f24f1da84dd4b918227026f68bbcc85b24097875a0259f5e4243a82696689"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define transformations for the training data and testing data
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

# Download and load the test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define the classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the network, define the criterion and optimizer
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
def train(net, trainloader, criterion, optimizer, epochs=2):  # Reduced epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

# Test the network on the test data
def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

# Function to display some sample test images and predictions
def display_predictions(net, testloader, classes, num_images=5):
    dataiter = iter(testloader)
    images, labels = next(dataiter)  # Use the next() function
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(num_images)))
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(num_images)))

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge (question TEXT, answer TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS searches (query TEXT, timestamp TEXT)''')  # Add searches table
    c.execute('''CREATE TABLE IF NOT EXISTS settings (user_id TEXT, setting1 TEXT, setting2 TEXT, theme TEXT, timestamp TEXT)''')  # Add settings table
    conn.commit()
    conn.close()

# Store question and answer in the database
def store_knowledge(question, answer):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("INSERT INTO knowledge (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

# Store search query in the database
def store_search(query):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("INSERT INTO searches (query, timestamp) VALUES (?, datetime('now'))", (query,))
    conn.commit()
    conn.close()

# Store settings change in the database
def store_settings(user_id, setting1, setting2, theme):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("INSERT INTO settings (user_id, setting1, setting2, theme, timestamp) VALUES (?, ?, ?, ?, datetime('now'))", (user_id, setting1, setting2, theme))
    conn.commit()
    conn.close()

# Retrieve answer from the database
def retrieve_knowledge(question):
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute("SELECT answer FROM knowledge WHERE question=?", (question,))
    answer = c.fetchone()
    conn.close()
    return answer[0] if answer else None

# Define the GitHub class
class GitHubKnowledgeBase:
    def __init__(self):
        self.knowledge_base = {}

    def answer(self, question):
        logging.info(f"Received question: {question}")
        # Check if the answer is already in the knowledge base
        answer = retrieve_knowledge(question)
        if answer:
            logging.info("Answer found in knowledge base.")
            return answer

        # Generate a response using GitHub API
        try:
            headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
            search_url = f"https://api.github.com/search/issues?q={question}+in:title,body"
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()
            if data.get("items"):
                answer = data["items"][0]["title"]  # Get the title of the first issue that matches
                logging.info("Answer found using GitHub API.")
            else:
                answer = "No relevant GitHub issues found."
                logging.warning("No relevant GitHub issues found.")
        except requests.exceptions.HTTPError as e:
            answer = f"GitHub API request failed: {e}"
            logging.error(f"GitHub API request failed: {e}")
        except Exception as e:
            answer = f"An error occurred: {e}"
            logging.error(f"An error occurred: {e}")

        store_knowledge(question, answer)
        return answer

# Enhanced Conversational AI with Google Search integration
class DeepSeekAI:
    def __init__(self):
        self.conversation_history = []

    def generate_response(self, user_input):
        logging.info(f"User input: {user_input}")
        # Add user input to conversation history
        self.conversation_history.append(f"User: {user_input}")

        # Simple rule-based response generation for specific commands
        if "hello" in user_input.lower():
            response = "Hello! How can I assist you today?"
        elif "how are you" in user_input.lower():
            response = "I'm just a program, but I'm here to help you!"
        else:
            # Treat all other user inputs as questions or things to be answered
            query = user_input.strip()
            store_search(query)  # Store the search query
            search_result = self.search_serpapi(query)
            response = f"Here is what I found on Google: {search_result}"

        # Add AI response to conversation history
        self.conversation_history.append(f"AI: {response}")
        logging.info(f"AI response: {response}")

        return response

    def search_serpapi(self, query):
        logging.info(f"Searching SerpAPI for query: {query}")
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_KEY
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            if 'organic_results' in results:
                return results['organic_results'][0]['snippet']
            else:
                return "No results found."
        except Exception as e:
            logging.error(f"An error occurred while searching: {e}")
            return f"An error occurred while searching: {e}"

# Flask app setup
app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Hardcoded for simplicity
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create instances of the GitHubKnowledgeBase and DeepSeekAI classes
github_kb = GitHubKnowledgeBase()
deepseek_ai = DeepSeekAI()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# In-memory user store
users = {'admin': bcrypt.generate_password_hash('password').decode('utf-8')}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            user = User('admin')
            login_user(user)
            return redirect(url_for('guest'))
        elif username in users and bcrypt.check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('guest'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/guest')
@login_required
def guest():
    return render_template('guest.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
@login_required
def ask():
    try:
        data = request.get_json()
        question = data.get('question')
        response = deepseek_ai.generate_response(question)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error in /ask route: {e}")
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        setting1 = request.form.get('setting1')
        setting2 = request.form.get('setting2')
        theme = request.form.get('theme')
        # Save settings logic here
        store_settings(current_user.id, setting1, setting2, theme)  # Store settings change
        session['theme'] = theme  # Save theme in session
        return redirect(url_for('settings'))  # Redirect to settings page
    else:
        conn = sqlite3.connect('knowledge_base.db')
        c = conn.cursor()
        c.execute("SELECT setting1, setting2, theme FROM settings WHERE user_id=?", (current_user.id,))
        settings = c.fetchone()
        conn.close()
        if settings:
            setting1, setting2, theme = settings
        else:
            setting1 = setting2 = theme = ''
        return render_template('settings.html', setting1=setting1, setting2=setting2, theme=theme)

@app.context_processor
def inject_theme():
    return dict(theme=session.get('theme', 'black'))  # Default to black theme

@app.before_request
def before_request():
    if current_user.is_authenticated:
        active_users.add(current_user.id)
        # Ensure theme is loaded from the database if not in session
        if 'theme' not in session:
            conn = sqlite3.connect('knowledge_base.db')
            c = conn.cursor()
            c.execute("SELECT theme FROM settings WHERE user_id=?", (current_user.id,))
            theme = c.fetchone()
            session['theme'] = theme[0] if theme else 'black'
            conn.close()

@app.teardown_request
def teardown_request(exception):
    if current_user.is_authenticated:
        active_users.discard(current_user.id)

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        feedback_text = request.form.get('feedback')
        # Save feedback logic here
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

# Track active users
active_users = set()

@app.route('/proxy', methods=['POST'])
@login_required
def proxy():
    data = request.get_json()
    url = data.get('url')
    method = data.get('method', 'GET')
    headers = data.get('headers', {})
    payload = data.get('payload', {})

    try:
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, params=payload)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload)
        elif method.upper() == 'PUT':
            response = requests.put(url, headers=headers, json=payload)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=headers, json=payload)
        else:
            return jsonify({'error': 'Unsupported HTTP method'}), 400

        return jsonify({
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content': response.json() if response.headers.get('Content-Type') == 'application/json' else response.text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    def test_torch():
        try:
            logging.info(f"Torch version: {torch.__version__}")
            logging.info("Torch library imported successfully.")
        except Exception as e:
            logging.error(f"Error importing torch library: {e}")

    test_torch()
    
    # Initialize the database
    init_db()  # Add this line to initialize the database
    
    env = gym.make('CartPole-v1')

    # First determine state size from environment
    state = env.reset()
    if isinstance(state, tuple):
        state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in state])
    else:
        state = np.array(state, dtype=np.float32).flatten() if not isinstance(state, dict) else np.array(list(state.values()), dtype=np.float32).flatten()
    state_size = state.shape[0]  # Get actual state size

    # Initialize agent with correct state size
    agent = DQNAgent(state_size=state_size, action_size=env.action_space.n)
    batch_size = 32

    # Train the neural network
    print("Starting neural network training...")  # Debug statement
    train(net, trainloader, criterion, optimizer, epochs=2)  # Reduced epochs
    test(net, testloader)
    display_predictions(net, testloader, classes)
    print("Neural network training completed.")  # Debug statement

    # Q-learning parameters
    num_episodes = 100  # Reduced episodes
    learning_rate = 0.1
    discount_factor = 0.99

    # Initialize Q-table with a fixed size
    Q_table_size = 10000  # Example size, adjust as needed
    Q = np.zeros([Q_table_size, env.action_space.n])

    # Function to convert state to a unique index
    def state_to_index(state, table_size):
        state_str = str(state)
        hash_object = hashlib.md5(state_str.encode())
        return int(hash_object.hexdigest(), 16) % table_size

    # Training loop
    print("Starting Q-learning training...")  # Debug statement
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in state])
        else:
            state = np.array(state, dtype=np.float32).flatten() if not isinstance(state, dict) else np.array(list(state.values()), dtype=np.float32).flatten()
        state_index = state_to_index(state, Q_table_size)
        done = False

        while not done:
            # Choose action based on epsilon-greedy policy
            if random.uniform(0, 1) < 0.1:  # Explore
                action = env.action_space.sample()
            else:  # Exploit
                action = np.argmax(Q[state_index, :])

            # Take action and observe new state and reward
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if isinstance(new_state, tuple):
                new_state = np.concatenate([np.array(s, dtype=np.float32).flatten() if not isinstance(s, dict) else np.array(list(s.values()), dtype=np.float32).flatten() for s in new_state])
            else:
                new_state = np.array(new_state, dtype=np.float32).flatten() if not isinstance(new_state, dict) else np.array(list(new_state.values()), dtype=np.float32).flatten()
            new_state_index = state_to_index(new_state, Q_table_size)

            # Update Q-value
            Q[state_index, action] = Q[state_index, action] + learning_rate * (reward + discount_factor * np.max(Q[new_state_index, :]) - Q[state_index, action])

            state_index = new_state_index

    # Close the environment
    env.close()
    print("Q-learning training completed.")  # Debug statement

    # Run Flask app with gunicorn for production
    logging.info("Starting Flask app with gunicorn for 24/7 availability")
    from gunicorn.app.wsgiapp import run
    run(["gunicorn", "-w", "4", "wsgi:app:app"])