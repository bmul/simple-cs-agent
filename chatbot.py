# Description: A simple chatbot that uses OpenAI's text embeddings to match user queries with predefined FAQ answers. 
# The chatbot computes the embeddings for the user input and the FAQ questions, then calculates the cosine similarity between them to find the best match. 
# If the similarity score is above a certain threshold, the chatbot returns the corresponding answer; otherwise, it provides a default response. 
# The chatbot is implemented as a Gradio interface, allowing users to interact with it through a web-based chat interface.
import gradio as gr
import openai
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Predefined dataset
faq_data = {
    "questions": [
        {
            "question": "What does the eligibility verification agent (EVA) do?",
            "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
        },
        {
            "question": "What does the claims processing agent (CAM) do?",
            "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
        },
        {
            "question": "How does the payment posting agent (PHIL) work?",
            "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
        },
        {
            "question": "Tell me about Thoughtful AI's Agents.",
            "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
        },
        {
            "question": "What are the benefits of using Thoughtful AI's agents?",
            "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
        }
    ]
}

# Get OpenAI text embeddings
def get_embedding(text):
    """Get OpenAI embedding for a given text using the latest OpenAI API."""
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(response.data[0].embedding)

# Precompute embeddings for questions
def compute_question_embeddings():
    return {q: get_embedding(q) for q in questions}

# Extract questions and answers
questions = [item["question"] for item in faq_data["questions"]]
answers = {item["question"]: item["answer"] for item in faq_data["questions"]}
question_embeddings = compute_question_embeddings()

# Chatbot response function
def chatbot_response(user_input, history):
    """Returns the best matching FAQ answer or a default response."""
    if not OPENAI_API_KEY:
        return "Error: OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable."
    
    try:
        # Compute user input embedding
        user_embedding = get_embedding(user_input)
        # Calculate cosine similarity with each question embedding
        similarities = {q: cosine_similarity([user_embedding], [emb])[0][0] for q, emb in question_embeddings.items()}
        # Find the best matching question
        best_match = max(similarities, key=similarities.get)
        # Get the corresponding answer
        best_score = similarities[best_match]
        print(f"Best match: {best_match}, Score: {best_score}")
        
        if best_score > 0.75:  # Threshold for a good match, determined by experimentation
            return answers[best_match]
        else:
            return "I'm sorry, I don't have an answer for that question. Please visit our website for more information."
    except Exception as e:
        return "An unexpected error occurred. Please try again later."

# Create Gradio interface
chatbot = gr.ChatInterface(fn=chatbot_response, title="Simple Customer Service Chatbot")

# Launch the chatbot on default port 7860
if __name__ == "__main__":
    chatbot.launch()
