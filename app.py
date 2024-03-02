from flask import Flask
import dspy
import os
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)

@app.route('/test')
def hello_world():
    class SystemOfEquations(dspy.Signature):
        """Solve for x and y in the system of equations"""

        equations = dspy.InputField(desc="Comma-separated list of equations")
        answer = dspy.OutputField(desc="JSON containing x and y")
    qa = dspy.ChainOfThought(SystemOfEquations)
    answer = qa(equations="3x+3y=42,2x-2y=0")
    return answer.answer
    