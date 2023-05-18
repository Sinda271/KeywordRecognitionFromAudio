import random
import os
from flask import Flask, request, jsonify
from keyword_spotting import Keyword_Spotting_Service

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():

    #  get audio file from client and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,10000))
    audio_file.save(file_name)

    #  make Keyword_Spotting_Service object
    kss = Keyword_Spotting_Service()

    #  make prediction
    predicted_keyword = kss.predict(file_name)

    #  remove audio file
    os.remove(file_name)

    #  send back the predicted word to the client in json format
    data = {"keyword": predicted_keyword}

    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
