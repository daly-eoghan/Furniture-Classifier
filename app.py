import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_from_directory, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from flask_uploads import UploadSet, IMAGES, configure_uploads


# Load up the best model we trained in furniture_classifier.ipynb
model = tf.keras.models.load_model('./best_network.h5')

def prepare_image(img):
    # Takes the image and fixes its dimensions so that it works with the
    # pretrained network.
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    return np.expand_dims(np.array(img), 0)

def predict_result(img):
    predictions = model.predict(img)
    predicted_class = [["Bed", "Chair", "Sofa"][i] for i in predictions.argmax(1)]
    return predicted_class[0]

# Creates a flask instance.
app = Flask(__name__)
app.config['SECRET_KEY'] = "Welcome to the Jungle"
app.config['UPLOADED_PHOTOS_DEST'] = "./predicted_images"

# Collection of files.
pictures = UploadSet("photos", IMAGES)

# Goes through all Upload sets, gets the configuration and stores them
# on the app.
configure_uploads(app, pictures)

class UploadForm(FlaskForm):
    picture = FileField( validators = [
            FileAllowed(pictures, "Pictures Only"), 
            FileRequired("Field must not be empty!")
        ]
    )
    submit = SubmitField("Submit")
    
@app.route('/predicted_images/<file>')
def get_file(file):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], file)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    # True if the form is valid and submitted.
    if form.validate_on_submit():
        filename = pictures.save(form.picture.data)
        file_url = url_for("get_file", file = filename)

        with open(app.config['UPLOADED_PHOTOS_DEST'] + "/" + filename, "rb") as image:
            img = image.read()
            prediction = predict_result(prepare_image(img))
            
    else:
        file_url = None
        prediction = None
    return render_template('index.html', form = form, file_url = file_url, prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')