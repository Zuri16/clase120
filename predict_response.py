from urllib import response
from matplotlib.style import use
import nltk
import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

import tensorflow
from data_preprocessing import get_stem_words

model=tensorflow.keras.models.load_model('./chatbot_model.h5')
intense=json.loads(open('./intents.json').read())
word=pickle.load(open('./words.pkl','rb'))
clases=pickle.load(open('./classes.pkl','rb'))

def preprocessing_input(user_input):
    token_input=nltk.word_tokenize(user_input)
    word_root=get_stem_words(token_input,ignore_words)
    input_ordenado=sorted(list(set(word_root)))
    
    bag=[]
    words_bag=[]

    for w in word:
        if w in input_ordenado:
            words_bag.append(1)
        else:
            words_bag.append(0)
    
    bag.append(words_bag)
    return np.array(bag)

def predicion(user_input):
    input_pro=preprocessing_input(user_input)
    prediction=model.predict(input_pro)
    predicted_class=np.argmax(prediction[0])
    
    return predicted_class

def bot_response(user_input):
    prediction=predicion(user_input)
    prediction_class=clases[prediction]
    for int in intense['intents']:
        if int['tag'] == prediction_class:
            respuesta_bot=random.choice(int['responses'])
            return respuesta_bot

print('Hola soy bot')
while True:
    input=input("Escribe aqui")
    print('user',input)
    response_bot=bot_response(input)
    print('bot',response_bot)
