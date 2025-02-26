# from flask import Flask, render_template, request
# from transformers import pipeline

# app = Flask(__name__)

# # Load the Question Answering model
# qa_pipeline = pipeline("question-answering")

# @app.route("/", methods=["GET", "POST"])
# def home():
#     answer = None
#     if request.method == "POST":
#         context = request.form["context"]
#         question = request.form["question"]
        
#         # Get the answer from the model
#         result = qa_pipeline(question=question, context=context)
#         answer = result["answer"]
    
#     return render_template("index.html", answer=answer)

# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
from transformers import pipeline

# Load the Question Answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering")

qa_pipeline = load_qa_model()

# Streamlit UI
st.title("Question Answering System")
st.write("Enter a passage and ask a question based on it.")

# Input fields
context = st.text_area("Enter the passage:")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.write(f"**Answer:** {result['answer']}")
    else:
        st.warning("Please enter both a passage and a question.")
