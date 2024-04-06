import pandas as pd
import requests
import os
from datetime import datetime
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path



@st.cache_data
def getListofAllIDs(csv_path="CoinsListsData.csv"):
    if os.path.exists(csv_path):
        print(f"Loading coins list from {csv_path}.")
        df = pd.read_csv(csv_path)
    else:
        print("Fetching coins list from CoinGecko API.")
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data)
        df_rowsWithNumbers = df.map(lambda x: any(char.isdigit() for char in str(x))).any(axis=1)
        df = df[~df_rowsWithNumbers]
        df.to_csv(csv_path, index=False)
        print(f"Coins list saved to {csv_path} for future use.")
    return df



@st.cache_data
def get_historical_data(coin_id, start_date, end_date, currency="cad"):
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    headers = {"x-cg-demo-api-key": "CG-YG83hELn8uTajFVAJ8nvA2nB"}
    params = {
        "vs_currency": currency,
        "from": start_timestamp,
        "to": end_timestamp
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"Failed to fetch historical data for {coin_id}.")
        return pd.DataFrame()



def load_or_train_model():
    model_path = Path("mnist_model.keras")

    if model_path.exists():
        print("Loading model...")
        return tf.keras.models.load_model(model_path)
    
    else:
        print("Training model...")
        mnist = tf.keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0

        train_images = train_images[..., tf.newaxis].astype("float32")
        test_images = test_images[..., tf.newaxis].astype("float32")

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        st.info("Model is Training. This is a one time activity. Model will be stored once this is done!")
        model.fit(train_images, train_labels, epochs=1, validation_split=0.1)

        model.save(model_path)
        return model



def predict_digit(image, model):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.split()[-1] 
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg

    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    predictions = tf.nn.softmax(predictions).numpy()
    label = np.argmax(predictions)
    confidence = np.max(predictions)
    return label, confidence