print("Initializing Campus Information Chatbot...")

print("importing libraries...")
import json
import random
import re
import string
import numpy as np
import torch
import pickle
import spacy
from transformers import BertTokenizer, BertModel
from tensorflow.keras.models import load_model
print("all libraries have been imported")

print("loading resources...")
print("-- loading model")
model = load_model("chatbot_campus_lstm.h5")

print("-- loading label encoder...")
with open("chatbot_campus_lstm_label_encoder.pkl", "rb") as enc:
    label_encoder = pickle.load(enc)

print("-- loading intents data...")
with open("../intents.json") as file:
    data = json.load(file)

print("-- loading spaCy model...")
nlp = spacy.load("en_core_web_lg")

print("-- loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(f"chatbot_campus_lstm_custom_tokenizer/")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()
print("done loading resources...")

# region ===== data preparation =====
def get_bert_embedding(sentence, max_len=30):
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state.squeeze(0).numpy()
# endregion

# region === data preprocessing ===
def apply_ner_tags(text):
    doc = nlp(text)
    
    # Get all entities (no filtering, like notebook)
    entities = [(ent.start, ent.end, ent.label_) for ent in doc.ents]
    
    # Reconstruct text with tags
    tokens = [token.text for token in doc]
    
    # Process entities in reverse to avoid index shifting
    for start, end, label in sorted(entities, reverse=True):
        tokens[start:end] = [f"<{label}>"]
    
    return ' '.join(tokens)

def preprocess_text(text):
    words = text.split()
    processed_words = [word.lower() for word in words]
    processed_words = [word.strip() for word in processed_words]
    processed_words = [re.sub('\s+',' ', word) for word in processed_words]
    processed_words = [word for word in processed_words if not all(char in string.punctuation for char in word.replace(' ',''))]
    
    processed_words = ' '.join(processed_words)
    return processed_words
# endregion 
# === Chat function ===

def chat():
    print(r""" 
    =================================================      
    Campus Information Chatbot - President University
    =================================================
    """)
    print("Type 'quit' at any time to exit")
    
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            print("Exiting chat. Goodbye!")
            break
        
 
        inp = preprocess_text(inp)
        tagged_input = apply_ner_tags(inp)
        
        embedded_input = np.expand_dims(get_bert_embedding(tagged_input), axis=0)
        prediction = model.predict(embedded_input)[0]
        predicted_class_index = np.argmax(prediction)
        tag = label_encoder.inverse_transform([predicted_class_index])[0]

        for intent in data['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print("Bot:", response)
                print("="*50)
                break

if __name__ == "__main__":    
    chat()
