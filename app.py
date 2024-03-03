from flask import Flask, request, json
import dspy
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import urllib.parse
import yfinance as yf
import redis
from PIL import Image
from io import BytesIO
import base64
import redis

app = Flask(__name__)
client = OpenAI()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)
r = redis.from_url(os.environ['REDIS_URL'])

@app.route('/question_replies', methods=['GET'])
def question_replies():
    text = request.args.get('text', 'What is the meaning of life?')
    adjective = request.args.get('adjective', 'None')
    class Question_Three_Replies(dspy.Signature):
        """Given a tweet with a question, generate three possible unique replies in the voice of the given adjective."""

        question = dspy.InputField()
        adjective = dspy.InputField(desc="Adjective to describe the replies")
        reply1 = dspy.OutputField(desc="1-5 words")
        reply2 = dspy.OutputField(desc="1-5 words")
        reply3 = dspy.OutputField(desc="1-5 words")
    q_3 = dspy.Predict(Question_Three_Replies)
    try:
        answer = q_3(question=text, adjective=adjective)
        answer = answer.toDict()
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    class Controversial(dspy.Signature):
        """Given a tweet, determine whether it is controversial or not."""

        tweet = dspy.InputField()
        controversial = dspy.OutputField(desc="Y or N")
    contro = dspy.Predict(Controversial)
    try:
        cont = contro(tweet=text)
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    if cont.controversial == "Y":
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                },
                data=json.dumps({
                    "model": "perplexity/sonar-small-online", # Optional
                    "messages": [
                    {"role": "user", "content": "Give me the one top link to a news article that disproves the following tweet: " + "The insurrection was a peaceful protest."}
                    ]
                })
            )
        except Exception as e:
            response = app.response_class(
                response=json.dumps({'error': str(e)}),
                status=500,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        class CommunityNote(dspy.Signature):
            """Given a tweet and an explanation of why it is incorrect, generate a correction of the base statement without referring to oneself or the other user. Humor is a plus."""

            tweet = dspy.InputField()
            explanation = dspy.InputField(desc="Explanation of why the base statement is incorrect")
            correction = dspy.OutputField(desc="Informative correction of the base statement and mildly humorous if possible. Cite sources.")
        cn = dspy.Predict(CommunityNote)
        try:
            cn_ret = cn(tweet=text, explanation = response.json()["choices"][0]["message"]["content"])
            response = app.response_class(
                response=json.dumps({'reply1': answer["reply1"], 
                                    'reply2': answer["reply2"],
                                    'reply3': answer["reply3"],
                                    'community_note': cn_ret.correction}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            response = app.response_class(
                response=json.dumps({'error': str(e)}),
                status=500,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    else:
        response = app.response_class(
            response=json.dumps({'reply1': answer["reply1"], 
                                'reply2': answer["reply2"],
                                'reply3': answer["reply3"],
                                    'community_note': "None"}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    
@app.route('/filter_image', methods=['GET'])
def filter_image():
    image_url = request.args.get('image_url', 'None')
    new_filter = request.args.get('new_filter', 'None')
    if new_filter == 'None':
        response = app.response_class(
            response=json.dumps({'url': image_url}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    if image_url == 'None':
        response = app.response_class(
            response=json.dumps({'error': "No image URL provided"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        if r.exists(image_url+new_filter):
            print("Here")
            the_image = r.get(image_url+new_filter).decode('utf-8')
            response = app.response_class(
                response=json.dumps({'image': the_image}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
            messages=[
                {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Briefly describe the contents of the image. Two sentences max."},
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
        response.headers.add('Access-Control-Allow-Origin', '*')
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
        # Step 1: Download the image
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.

        # Step 2: Compress the image
        image = Image.open(BytesIO(response.content))
        compressed_image_io = BytesIO()
        image.save(compressed_image_io, format='JPEG', quality=20)
        compressed_image_io.seek(0)  # Rewind the file-like object to its beginning
        image_base64 = base64.b64encode(compressed_image_io.read()).decode('utf-8')
        print(type(image_base64))
        r.set(image_url+new_filter, image_base64)
        response = app.response_class(
            response=json.dumps({'image': image_base64}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/text_to_image', methods=['GET'])
def text_to_image():
    text = request.args.get('text', 'None')
    new_filter = request.args.get('new_filter', 'None')
    if text == 'None':
        response = app.response_class(
            response=json.dumps({'error': "No text provided"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    if r.exists(text+new_filter):
        the_image = r.get(text+new_filter).decode('utf-8')
        response = app.response_class(
            response=json.dumps({'image': the_image}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        if new_filter == "None":
            prompt = "Convert the following text prompt to an image: " + text
        else:
            prompt = f"Reimagine the following text prompt if it were filtered like {new_filter}: {text}"
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        url = response.dict()["data"][0]["url"]
        # Step 1: Download the image
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the request returned an unsuccessful status code.

        # Step 2: Compress the image
        image = Image.open(BytesIO(response.content))
        compressed_image_io = BytesIO()
        image.save(compressed_image_io, format='JPEG', quality=20)
        compressed_image_io.seek(0)  # Rewind the file-like object to its beginning
        image_base64 = base64.b64encode(compressed_image_io.read()).decode('utf-8')
        r.set(text+new_filter, image_base64)
        response = app.response_class(
            response=json.dumps({'image': image_base64}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting text to image"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
@app.route('/text_to_coords', methods=['GET'])
def text_to_coords():
    text = request.args.get('text', 'None')
    if text == 'None':
        response = app.response_class(
            response=json.dumps({'error': "No text provided"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    class Address_Finder(dspy.Signature):
        """Given a tweet, derive an address or location from the tweet that could be geocoded"""

        tweet = dspy.InputField()
        location = dspy.OutputField(desc="Address or location, or 'None' if no location found.")
    try:
        add = dspy.Predict(Address_Finder)
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in deriving location from text"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    answer = add(tweet=text)
    try:
        encoded_location = urllib.parse.quote(answer.location, safe='')
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_location}.json"
        params = {
            "access_token": MAPBOX_API_KEY,
        }
        response = requests.get(url, params=params)
        lon, lat = response.json()["features"][0]["center"]
        response = app.response_class(
            response=json.dumps({'latitude': lat, 'longitude': lon}),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting text to coordinates"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
@app.route('/text_to_weather', methods=['GET'])
def text_to_weather():
    location = request.args.get('location', 'None')
    try:
        encoded_location = urllib.parse.quote(location, safe='')
        url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{encoded_location}.json"
        params = {
            "access_token": MAPBOX_API_KEY,
        }
        response = requests.get(url, params=params)
        lon, lat = response.json()["features"][0]["center"]
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting text to coordinates"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=imperial"
        response = requests.get(url)
        weather = response.json()
        returnjson = {
            "location": weather["name"],
            "description": weather["weather"][0]["description"],
            "temperature": weather["main"]["temp"],
            "feels_like": weather["main"]["feels_like"],
            "humidity": weather["main"]["humidity"],
            "wind_speed": weather["wind"]["speed"],
            "wind_direction": weather["wind"]["deg"],
            "icon": f"https://openweathermap.org/img/wn/{weather["weather"][0]["icon"]}@2x.png",
        }
        response = app.response_class(
            response=json.dumps(returnjson),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in fetching weather data"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/text_to_finance_data', methods=['GET'])
def text_to_finance_data():
    ticker = request.args.get('ticker', 'None')
    if ticker == "None":
        response = app.response_class(
            response=json.dumps({'error': "No ticker found"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        data = yf.download(ticker, period="1mo")
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in fetching finance data"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        percent_today = (data["Close"][-1] - data["Open"][-1]) / data["Open"][-1] * 100
        current_price = data["Close"][-1]
        amount_today = data["Close"][-1] - data["Open"][-1]
        close_prices = data["Close"].to_dict()
        high = data["High"][-1]
        low = data["Low"][-1]
        volume = data["Volume"][-1]
        close_prices = {timestamp.to_pydatetime().isoformat() + 'Z': price for timestamp, price in close_prices.items()}
        returnjson = {
            "ticker": ticker,
            "percent_today": percent_today,
            "amount_today": amount_today,
            "current_price": current_price,
            "close_prices": close_prices,
            "high": high,
            "low": low,
            "volume": int(volume),
        }
        response = app.response_class(
            response=json.dumps(returnjson),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

@app.route('/text_to_politics', methods=['GET'])
def text_to_politics():
    text = request.args.get('text', 'None')
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": "perplexity/sonar-small-online", # Optional
            "messages": [
            {"role": "user", "content": "Give me the top three links to current news articles in a numbered list related to this tweet: " + text \
                + "make sure to included the title  followed by a colon and then the link to the article"}
            ]
        })
    )
    data = json.loads(response.text)
    message_content = data['choices'][0]['message']['content']

    def get_title_link(input_text):
        class get_title(dspy.Signature):
            """Given a string with a title in it, give me just the title as a string"""
            text = dspy.InputField()
            title = dspy.OutputField(desc="title of article")
        title = dspy.Predict(get_title)
        title_out = title(text=input_text).toDict()['title']

        class get_link(dspy.Signature):
            """Given a string with a link in it, give me just the link as a string"""
            text = dspy.InputField()
            link = dspy.OutputField(desc="link to article")
        link = dspy.Predict(get_link)
        link_out = link(text=input_text).toDict()['link']

        return {"title": title_out, "link": link_out}

    class format_output(dspy.Signature):
        """Given a string of articles, give me the title and link of each article"""
        text = dspy.InputField()
        formatted = dspy.OutputField(desc="title of article and link to article")
    formatted = dspy.Predict(format_output)
    answer = formatted(text=message_content)
    answer = answer.toDict()['formatted']
    lines = answer.split('\n')

    articles = []

    for line in lines:
        articles.append(get_title_link(line))

    class classify_party(dspy.Signature):
        """Given tweet text, classify the political party of the author"""
        text = dspy.InputField()
        party = dspy.OutputField(desc="Right, Left, or Center")
    party_output = dspy.Predict(classify_party)
    party = party_output(text=text).toDict()['party']
    
    response = app.response_class(
        response=json.dumps({"party": party, "articles": articles}),
        status=200,
        mimetype='application/json'
    )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response