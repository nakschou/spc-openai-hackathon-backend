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
from pinecone import Pinecone

app = Flask(__name__)
client = OpenAI()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_API_KEY = os.getenv("MAPBOX_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)
r = redis.from_url(os.environ['REDIS_URL'])
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("spchack")

@app.route('/question_replies', methods=['GET'])
def question_replies():
    try:
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
        answer = q_3(question=text, adjective=adjective)
        answer = answer.toDict()
        class Controversial(dspy.Signature):
            """Given a tweet, determine whether it is controversial or not."""

            tweet = dspy.InputField()
            controversial = dspy.OutputField(desc="Y or N")
        contro = dspy.Predict(Controversial)
        cont = contro(tweet=text)
        if cont.controversial == "Y":
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
            class CommunityNote(dspy.Signature):
                """Given a tweet and an explanation of why it is incorrect, generate a correction of the base statement without referring to oneself or the other user. Humor is a plus."""

                tweet = dspy.InputField()
                explanation = dspy.InputField(desc="Explanation of why the base statement is incorrect")
                correction = dspy.OutputField(desc="Informative correction of the base statement and mildly humorous if possible. Cite sources.")
            cn = dspy.Predict(CommunityNote)
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
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    
@app.route('/filter_image', methods=['GET'])
def filter_image():
    image_url = request.args.get('image_url', 'None')
    new_filter = request.args.get('new_filter', 'None')
    text = request.args.get('text', 'None')
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
            the_image = r.get(image_url+new_filter).decode('utf-8')
            response = app.response_class(
                response=json.dumps({'image': the_image}),
                status=200,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
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
        prompt = response.dict()["choices"][0]["message"]["content"]
        class AssistClassifier(dspy.Signature):
            """Given a message from a large language model, determine whether or not it provides information other than being unable to assist with our request."""

            message = dspy.InputField()
            denied = dspy.OutputField(desc="Y or N based on whether information other than being denied is shared.")
        assistclass = dspy.Predict(AssistClassifier)
        assist = assistclass(message=prompt)
        if assist.denied == "Y":
            filtered_prompt = f"Reimagine the following prompt if it were filtered like {new_filter}: {text}"
        else:
            filtered_prompt = f"Reimagine the following prompt if it were filtered like {new_filter}: {prompt}"
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
            response=json.dumps({'error': str(e)}),
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

@app.route('/get_clothing', methods=['GET'])
def get_clothing_items():
    url = request.args.get('url', 'None')
    if url == "None":
        response = app.response_class(
            response=json.dumps({'error': "No url found"}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    elif r.exists(url):
        items = r.get(url)
        response = app.response_class(
            response=items,
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    def get_link(url):
        items = get_image_description(url)
        items = strip_newlines(items)

        #for each item in the items, check how many items there are in the value
        parsed_items = process_json(items)
        return parsed_items


    def get_image_description(url):
        response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are all of the articles of clothing in this image? Format the response in JSON object format, \
                with the following categories as keys: 'pants, shorts, hats, shirts, tshirts, shoes, jackets'. For each entry in any of the \
                    categories, make sure to include a two sentence description including things like the color, material used, and any other relevant details."},
                {
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
                },
            ],
            }
        ],
        max_tokens=300,
        )

        return response.choices[0].message.content


    def process_json(json_object):
        for key, value in json_object.items():
            if isinstance(value, list):
                for item in value:
                    # Assuming each item in the list has a 'description' key
                    if isinstance(item, dict) and 'description' in item:
                        item['description'] = embed_text(item['description'])
                    elif isinstance(item, str):
                        # If the list contains strings, embed those
                        value[value.index(item)] = embed_text(item)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                json_object[key] = process_json(value)
            elif isinstance(value, str):
                # Embed the string value
                json_object[key] = embed_text(value)
        
        return json_object


    def strip_newlines(desc):
        try:
            desc = desc.replace('\n', '')
            desc = desc.replace('`', '')
            desc = desc.replace('json', '')
            desc = json.loads(desc)
            return desc
        except:
            return 'Error stripping newlines'

    def embed_text(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    def item_query(vector, item):
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("spchack")
        match = index.query(
            vector=vector,
            filter={
                "item_type": {"$eq": item},
            },
            top_k=1,
            include_metadata=True
        )
        return match

    def replace_vectors_with_search_results(json_object):
        for key, value in json_object.items():
            if isinstance(value, list):
                for item in value:
                    # Assuming each item in the list has a 'description' key
                    if isinstance(item, dict) and 'description' in item:
                        # Call vector search function and replace the vector with the result
                        item['description'] = item_query(item['description'], key)
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                json_object[key] = replace_vectors_with_search_results(value)
        
        return json_object

    def extract_clothing_info(input_data):
        result = {}

        for category, items in input_data.items():
            category_info = []

            for item in items:
                description = item.get('description', {})
                matches = description.get('matches', [])

                for match in matches:
                    metadata = match.get('metadata', {})
                    article_info = {
                        'img_src': metadata.get('img_src', ''),
                        'item_link': metadata.get('item_link', ''),
                        'description': metadata.get('description', ''),
                        'item_type': metadata.get('item_type', ''),
                    }

                    category_info.append(article_info)

            result[category] = category_info

        return result
    
    error = ""
    attempts = 0
    while attempts < 2:
        try:
            items = get_link(url)
            items = replace_vectors_with_search_results(items)
            items = extract_clothing_info(items)
            response = app.response_class(
                response=json.dumps(items),
                status=200,
                mimetype='application/json'
            )
            r.set(url, json.dumps(items))
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as e:
            error = str(e)
            attempts += 1

    response = app.response_class(
            response=json.dumps({'error': error}),
            status=500,
            mimetype='application/json'
        )
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response




@app.route('/text_to_politics', methods=['GET'])
def text_to_politics():
    text = request.args.get('text', 'None')
    try:
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
    except Exception as e:
        response = app.response_class(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response