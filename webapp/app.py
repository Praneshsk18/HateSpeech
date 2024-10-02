from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import speech_recognition as sr
from pydub import AudioSegment
import os

app = Flask(__name__)
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Clean text function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopwords_set]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Load and preprocess data
data = pd.read_csv("twitter.csv")
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Speech", 2: "Normal"})
data = data[["tweet", "labels"]]
data["tweet"] = data["tweet"].apply(clean)
x = np.array(data["tweet"])
y = np.array(data["labels"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Text input
        if 'text_input' in request.form:
            user_input = request.form['text_input']
            return redirect(url_for('predict', input_text=user_input))
        
        # Audio input
        if 'audio_input' in request.files:
            audio_file = request.files['audio_input']
            if audio_file:
                recognizer = sr.Recognizer()
                audio_path = "temp." + audio_file.filename.split('.')[-1]
                audio_file.save(audio_path)
                
                # Convert to WAV format
                audio = AudioSegment.from_file(audio_path)
                audio.export("temp.wav", format="wav")
                
                # Perform recognition
                with sr.AudioFile("temp.wav") as source:
                    audio_data = recognizer.record(source)
                    user_input = recognizer.recognize_google(audio_data)
                
                os.remove(audio_path)
                os.remove("temp.wav")
                return redirect(url_for('predict', input_text=user_input))
    
    return render_template("index.html")

@app.route("/predict")
def predict():
    normal_string = request.args.get('input_text')
    data = cv.transform([normal_string]).toarray()
    output_text = clf.predict(data)
    output_text = output_text[0]
    return render_template("result.html", input_text=normal_string, output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True, port=5001)