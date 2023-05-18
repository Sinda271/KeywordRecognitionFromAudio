import tensorflow.keras as keras
import numpy as np
import librosa

MODEL_PATH = "model.h5"
SAMPLE_RATE = 22050


class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "stop",
        "up",
        "down",
        "left",
        "right"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # ( # segments, # MFCCs)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # convert 2d array to 4d array -> (# samples, # segments, # MFCCs, 1)

        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]
        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) > SAMPLE_RATE:
            signal = signal[:SAMPLE_RATE]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure that we have only 1 instance of Keyword spotting service
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    keyword1 = kss.predict("../../local/test/Down.wav")
    keyword2 = kss.predict("../../local/test/Left.wav")
    keyword3 = kss.predict("../../local/test/Stop.wav")
    keyword4 = kss.predict("../../local/test/Right.wav")
    keyword5 = kss.predict("../../local/test/Up.wav")
    print(f"Predicted words are : Down: {keyword1}, Left: {keyword2}, Stop: {keyword3}, Right: {keyword4}, Up: {keyword5}")
