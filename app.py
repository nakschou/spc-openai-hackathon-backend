from flask import Flask, request, json
import dspy
import os
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)
client = OpenAI()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)

@app.route('/question_replies', methods=['GET'])
def question_replies():
    text = request.args.get('text', 'What is the meaning of life?')
    class Question_Three_Replies(dspy.Signature):
        """Given a tweet with a question, generate three possible unique replies"""

        question = dspy.InputField()
        reply1 = dspy.OutputField(desc="1-5 words")
        reply2 = dspy.OutputField(desc="1-5 words")
        reply3 = dspy.OutputField(desc="1-5 words")
    q_3 = dspy.Predict(Question_Three_Replies)
    try:
        answer = q_3(question=text)
        answer = answer.toDict()
        response = app.response_class(
            response=json.dumps(answer),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        return response
    
@app.route('/filter_image', methods=['GET'])
def filter_image(image_url, new_filter):
    image_url = request.args.get('image_url', 'None')
    new_filter = request.args.get('new_filter', 'None')
    try:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
            messages=[
                {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Briefly describe the contents of the image."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
        )
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting from image to URL"}),
            status=500,
            mimetype='application/json'
        )
        return response
    prompt = response.dict()["choices"][0]["message"]["content"]
    filtered_prompt = f"Reimagine the following prompt if it were filtered like {new_filter}: {prompt}"
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=filtered_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = response.dict()["data"][0]["url"]
        response = app.response_class(
            response=json.dumps({'url': url}),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in filtering image"}),
            status=500,
            mimetype='application/json'
        )
        return response