import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Implementations of Chatbot using NLP")
    st.markdown(
    """
    <style>
    /* Smooth background with a soft gradient */
    body, .stApp {
        background: linear-gradient(135deg, #1E1E2F, #252641);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        text-align: center;
    }

    /* Chatbot container - ensures proper layout */
    .block-container {
        max-width: 700px;
        margin: auto;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Proper title alignment and spacing */
    .stApp h1 {
        color: #7dc9ff !important;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Sidebar Styling */
    .css-1d391kg, .css-1v3fvcr {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px;
        padding: 15px;
    }

    /* Input Box - Now with Black Text */
    .stTextInput>div>div>input {
        width: 100%;
        background: rgba(255, 255, 255, 0.9) !important; /* Light Background */
        color: black !important; /* Black Text */
        font-size: 16px;
        padding: 12px;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease-in-out;
    }
    .stTextInput>div>div>input:focus {
        border-color: #7dc9ff !important;
        box-shadow: 0px 0px 10px rgba(125, 201, 255, 0.6);
    }

    /* Chatbot Response Box - Now with Black Text */
    .stTextArea>div>textarea {
        width: 100%;
        background: rgba(255, 255, 255, 0.9) !important; /* Light Background */
        color: black !important; /* Black Text */
        font-size: 16px;
        border-radius: 10px;
        padding: 12px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    /* Buttons - rounded, smooth transition */
    .stButton>button {
        background: linear-gradient(to right, #7dc9ff, #5372ff) !important;
        color: white !important;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px;
        transition: 0.3s ease-in-out;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #5372ff, #7dc9ff) !important;
        box-shadow: 0px 0px 10px rgba(125, 201, 255, 0.6);
    }

    /* Chat History */
    .stText {
        font-size: 15px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
