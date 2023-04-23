from typing import Tuple, List
import numpy as np
import cv2

from dataloader_iam import Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

from flask import Flask, request, render_template, make_response

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = 'model/charList.txt'
    fn_summary = 'model/summary.json'
    fn_corpus = 'model/corpus.txt'


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())
    
def get_img_height() -> int:
    """Fixed height for NN."""
    return 32

def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def infer(model: Model, img):
    """Recognizes text in image provided by file path."""
    # img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f'Recognized: "{recognized[0]}"')
    print(f'Probability: {probability[0]}')
    return recognized[0]


app = Flask(__name__)
decoder_mapping = {'bestpath': DecoderType.BestPath,
                    'beamsearch': DecoderType.BeamSearch,
                    'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping['bestpath']

model = Model(char_list_from_file(), decoder_type, must_restore=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if image in request
    print(request.content_type)
    if 'image' not in request.files:
        return "No image file found", 400
    # Get the image from the request
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    print("Processing recognition")
    # Make predictions
    prediction = infer(model, img)

    # Return the predicted text
    return make_response(prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

