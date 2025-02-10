# Simple Customer Service Chatbot

This is a simple AI-powered customer service chatbot built with Gradio and OpenAI's embedding model. It matches user questions with predefined FAQ responses using cosine similarity.

## Features
- Uses OpenAI embeddings for semantic similarity
- Matches user queries with predefined FAQ questions
- Provides relevant answers based on similarity scores
- Runs a Gradio-based chat UI for interaction

## Installation

### Prerequisites
- Python 3.7+
- OpenAI API Key (set as an environment variable: `OPENAI_API_KEY`)

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Running the Chatbot
```sh
python chatbot.py
```
Once the script runs, you will get a Gradio link to interact with the chatbot on http://127.0.0.1:7860

### Setting the OpenAI API Key
Before running the script, set your OpenAI API key:
```sh
export OPENAI_API_KEY="your-api-key-here"  # macOS/Linux
set OPENAI_API_KEY="your-api-key-here"     # Windows
```

## How It Works
1. **Predefined FAQs:** The chatbot has a set of predefined questions and answers.
2. **Embedding Matching:** When a user asks a question, the chatbot computes its embedding using OpenAI's API and compares it with stored embeddings.
3. **Cosine Similarity:** The chatbot selects the closest matching FAQ based on cosine similarity.
4. **Response Selection:** If the similarity score is above a threshold, it returns the corresponding answer; otherwise, it provides a default response.

## Dependencies
- `gradio`
- `openai`
- `numpy`
- `scikit-learn`

## License
This project is open-source and available under the MIT License.

## Author
Developed by Brian Mullins