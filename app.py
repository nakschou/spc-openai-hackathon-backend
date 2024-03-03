from flask import Flask, request, json
import dspy
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import urllib.parse
import yfinance as yf

app = Flask(__name__)
client = OpenAI()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
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
def filter_image():
    image_url = request.args.get('image_url', 'None')
    new_filter = request.args.get('new_filter', 'None')
    if new_filter == 'None':
        response = app.response_class(
            response=json.dumps({'url': image_url}),
            status=200,
            mimetype='application/json'
        )
        return response
    if image_url == 'None':
        response = app.response_class(
            response=json.dumps({'error': "No image URL provided"}),
            status=500,
            mimetype='application/json'
        )
        return response
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
        response = app.response_class(
            response=json.dumps({'url': url}),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting text to image"}),
            status=500,
            mimetype='application/json'
        )
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
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in converting text to coordinates"}),
            status=500,
            mimetype='application/json'
        )
        return response
    
@app.route('/text_to_weather', methods=['GET'])
def text_to_weather():
    text = request.args.get('text', 'None')
    class Address_Finder(dspy.Signature):
        """Given a tweet, derive an address or location from the tweet that could be geocoded"""

        tweet = dspy.InputField()
        location = dspy.OutputField(desc="Address or location, or 'None' if no location found.")
    add = dspy.Predict(Address_Finder)
    try:
        answer = add(tweet=text)
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in deriving location from text"}),
            status=500,
            mimetype='application/json'
        )
        return response
    try:
        encoded_location = urllib.parse.quote(answer.location, safe='')
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
            "icon": weather["weather"][0]["icon"],
        }
        response = app.response_class(
            response=json.dumps(returnjson),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in fetching weather data"}),
            status=500,
            mimetype='application/json'
        )
        return response

@app.route('/text_to_finance_data', methods=['GET'])
def text_to_finance_data():
    text = request.args.get('text', 'None')
    class Ticker_Finder(dspy.Signature):
        """Given a tweet, derive a stock ticker from the tweet"""

        tweet = dspy.InputField()
        ticker = dspy.OutputField(desc="Ticker such as 'AAPL' or 'GOOGL', or 'None' if no ticker found.")
    tck = dspy.Predict(Ticker_Finder)
    try:
        answer = tck(tweet=text)
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in deriving ticker from text"}),
            status=500,
            mimetype='application/json'
        )
        return response
    if answer.ticker == "None":
        response = app.response_class(
            response=json.dumps({'error': "No ticker found"}),
            status=500,
            mimetype='application/json'
        )
        return response
    try:
        data = yf.download(answer.ticker, period="1mo")
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': "Error in fetching finance data"}),
            status=500,
            mimetype='application/json'
        )
        return response
    percent_today = (data["Close"][-1] - data["Open"][-1]) / data["Open"][-1] * 100
    current_price = data["Close"][-1]
    amount_today = data["Close"][-1] - data["Open"][-1]
    close_prices = data["Close"].to_dict()
    high = data["High"][-1]
    low = data["Low"][-1]
    volume = data["Volume"][-1]
    close_prices = {timestamp.to_pydatetime().isoformat() + 'Z': price for timestamp, price in close_prices.items()}
    returnjson = {
        "ticker": answer.ticker,
        "percent_today": percent_today,
        "amount_today": amount_today,
        "current_price": current_price,
        "close_prices": close_prices,
        "high": high,
        "low": low,
        "volume": volume,
    }
    response = app.response_class(
        response=json.dumps(returnjson),
        status=200,
        mimetype='application/json'
    )
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
            "model": "perplexity/sonar-small-chat", # Optional
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
    
    
    return {"party": party, "articles": articles}