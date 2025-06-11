import os
import numpy as np
from keras.preprocessing import image
import os
from dotenv import load_dotenv
from keras.models import load_model
from extra.evaluate_image import evaluate_rotten_vs_fresh, print_fresh
from flask import Flask, request, render_template, send_from_directory
from langchain_community.chat_models import ChatOllama
import subprocess
from datetime import datetime
import base64
load_dotenv()


app = Flask(__name__)
# app = Flask(__name__, static_folder="images")
llm = ChatOllama(model="gemma:2b", temperature=0)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Fresh Apple','Fresh Banana','Fresh Orange','Rotten Apple','Rotten Banana','Rotten Orange','......']


def print_fresh(res):
    threshold_fresh = 0.10  # set according to standards
    threshold_medium = 0.35  # set according to standards
    if res < threshold_fresh:
        return "FRESH"
    elif threshold_fresh < res < threshold_medium:
        return "The item is MEDIUM FRESH"
    else:
        return "Rotten"


@app.route("/")
def index():
    try:
        #default behaviour
        return render_template("index.html") # renders the index.html template
    except Exception as e:
        return str(e)

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')  # Directory to save uploaded images
    if not os.path.isdir(target):
        os.mkdir(target)

    if request.is_json:
        # Handle JSON data (from camera capture)
        data = request.get_json()
        image_data = data['file'].split(',')[1]  # Remove data:image/png;base64,
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(target, filename)
        
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(image_data))
    else:
        # Handle form data (from file upload)
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(target, filename)
            file.save(filepath)
        else:
            return "No file uploaded", 400

    # Process the saved image
    path_of_image = os.path.join('images', filename)
    is_rotten = evaluate_rotten_vs_fresh(path_of_image)
    freshness_result = print_fresh(is_rotten)

    if freshness_result.lower() == "fresh":
        prompt = f"""
        The item in the image ({path_of_image}) is identified as: {freshness_result}.

        Since it is fresh, suggest the top 3 healthy and practical ways this fruit or vegetable can be consumed or used. Focus on:
        - Nutritional benefits
        - Simple recipes or salad ideas
        - Quick preparation tips

        Only return the 3 best options.
        """

    elif freshness_result.lower() == "medium fresh":
        prompt = f"""
        The item in the image ({path_of_image}) is identified as: {freshness_result}.

        Since it is medium fresh, it may not be ideal for salads or raw consumption, but still good for blending. Suggest the top 3 ways to use it in:
        - Smoothies
        - Shakes
        - Purees or blended recipes

        Mention any nutritional benefits if relevant. Only return the 3 best options.
        """

    else:  # Assumes "rotten"
        prompt = f"""
        The item in the image ({path_of_image}) is identified as: {freshness_result}.

        Since it is rotten, it is unsuitable for direct consumption. Suggest the top 3 creative and sustainable reuse methods such as:
        - Composting techniques
        
        - Non-edible recycling or eco-friendly disposal solutions

        Only return the 3 best options.
        dont include art and crafts or any other use cases that are not related to food waste management.
        """

    answer = llm.invoke(prompt)



    return render_template("template.html", image_name=filename, text=freshness_result, content=answer.content)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)

