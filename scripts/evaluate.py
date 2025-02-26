from transformers import pipeline

# Load the QA model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

# Define multiple contexts and corresponding questions
qa_pairs = [
    {
        "context": "OpenAI developed ChatGPT, a language model for AI conversations.",
        "question": "Who developed ChatGPT?"
    },
    {
        "context": "Isaac Newton formulated the laws of motion and universal gravitation.",
        "question": "Who formulated the laws of motion?"
    },
    {
        "context": "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "question": "Where is the Eiffel Tower located?"
    },
    {
        "context": "Python is a high-level programming language created by Guido van Rossum in 1991.",
        "question": "Who created Python?"
    },
    {
        "context": "The Great Wall of China was built to protect against invasions and stretches over 13,000 miles.",
        "question": "Why was the Great Wall of China built?"
    },
    {
        "context": "The Pacific Ocean is the largest ocean on Earth, covering more than 63 million square miles.",
        "question": "Which is the largest ocean on Earth?"
    },
    {
        "context": "Albert Einstein developed the theory of relativity, which revolutionized modern physics.",
        "question": "Who developed the theory of relativity?"
    },
    {
        "context": "The Amazon Rainforest is the largest rainforest in the world, spanning nine countries.",
        "question": "Which is the largest rainforest in the world?"
    },
    {
        "context": "The capital of Japan is Tokyo, which is one of the most populated cities in the world.",
        "question": "What is the capital of Japan?"
    },
    {
        "context": "The Mona Lisa, painted by Leonardo da Vinci, is displayed in the Louvre Museum in Paris.",
        "question": "Who painted the Mona Lisa?"
    }
]

# Process each question-context pair
for pair in qa_pairs:
    result = qa_model(question=pair["question"], context=pair["context"])

    # Print the result with error handling
    if "answer" in result and "score" in result:
        print(f"Question: {pair['question']}")
        print(f"Answer: {result['answer']}, Confidence: {result['score']:.2f}")
        print("-" * 50)
    else:
        print(f"Error: Model did not return a valid answer for '{pair['question']}'")
