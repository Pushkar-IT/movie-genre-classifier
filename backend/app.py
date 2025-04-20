import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model("movie_genre_model.keras")


# Define genres (must match your model's training labels)
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror',
       'Music', 'Musical', 'Mystery', 'N/A', 'News', 'Reality-TV', 'Romance',
       'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

# Root route
@app.route('/')
def index():
    return "Welcome to the Movie Genre Prediction API!"

@app.route('/predict', methods=['POST'])
def predict_genre():
    try:
        if 'poster' not in request.files:
            return jsonify({'error': 'No poster uploaded'}), 400

        poster = request.files['poster']

        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)

        # Save the uploaded poster image
        img_path = os.path.join("uploads", poster.filename)
        poster.save(img_path)

        print("Saved image at:", img_path)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("Image shape:", img_array.shape)

        # Make prediction
        prediction = model.predict(img_array)
        print("Raw prediction:", prediction)

        # Get the predicted genre
        predicted_genre = genres[np.argmax(prediction)]
        print("Predicted genre:", predicted_genre)

        # Clean up the uploaded image file
        os.remove(img_path)

        # Return the prediction result
        return jsonify({'genre': predicted_genre})

    except Exception as e:
        print("Error occurred:", e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

