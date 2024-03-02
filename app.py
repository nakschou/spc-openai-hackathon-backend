from flask import Flask, request, jsonify
import dspy
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)

@app.route('/question_replies', methods=['GET'])
def question_replies():
    text = request.args.get('text', 'What is the meaning of life?')  # Default to 'World' if 'name' not provided
    class Question_Three_Replies(dspy.Signature):
        """Given a tweet with a question, generate three possible unique replies"""

        question = dspy.InputField()
        reply1 = dspy.OutputField(desc="1-5 words")
        reply2 = dspy.OutputField(desc="1-5 words")
        reply3 = dspy.OutputField(desc="1-5 words")
    q_3 = dspy.ChainOfThought(Question_Three_Replies)
    try:
        answer = q_3(question=text)
    except Exception as e:
        return jsonify({'error': str(e)}, status=500)
    return jsonify(answer.toDict(), status=200)