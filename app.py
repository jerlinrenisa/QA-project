from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the Question Answering model
qa_pipeline = pipeline("question-answering")

@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    if request.method == "POST":
        context = request.form["context"]
        question = request.form["question"]
        
        # Get the answer from the model
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"]
    
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
