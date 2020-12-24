import requests

URL = "http://34.229.71.99/predict"
TEST_AUDIO_FILE_PATH = "test/Up.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}

    # get predicted word from server
    response = requests.post(URL, files=values)
    data = response.json()
    print(f"The predicted keyword is {data['keyword']}")

