import numpy as np
import cv2

from pipeline import read_page, DetectorConfig
from pipeline.reader import read

from flask import Flask, request, render_template, make_response

# with open('pipeline/reader/model/corpus.txt') as f:
#     word_list = [w.strip().upper() for w in f.readlines()]
# prefix_tree = PrefixTree(word_list)


app = Flask(__name__, static_folder='templates')
# CORS(app)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image in request
    if 'image' not in request.files:
        return "No image file found\n", 400
    # Get the image from the request
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    rType =request.form.get('rType')
    prediction = ''
    if rType == "0":
        print("Processing letter")
        # Make predictions
        prediction = read(img)[:1]
    elif rType == "1":

        print("Processing word")
        # Make predictions
        prediction = read(img)

    elif rType == "2":
        print("Processing text")
        read_lines = read_page(img, DetectorConfig(height=1000))
        for read_line in read_lines:
            prediction += ' '.join(read_word.text for read_word in read_line) + '\n'
    else:
        return "Invalid recognition type\n", 400
              
    # Return the predicted text
    return make_response(prediction+'\n')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

