print("Initializing Campus Information Chatbot...")

import json
import random
import numpy as np
import pickle
import spacy
import string
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

# variable initialization
EMBEDDING_DIM = 300
MAX_LEN = 20

print("loading resources...")
print("-- loading model")
model = load_model("chatbot_campus_lstm_original.h5")

print("-- loading label mapping")
with open("chatbot_campus_lstm_original_label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)
    label_encoder = label_mapping["label_encoder"]

print("-- loading tokenizer")
with open("chatbot_campus_lstm_original_glove_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("-- loading intents dataset")
with open("../intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

print("-- loading spaCy model...")
nlp = spacy.load("en_core_web_lg")

print("done loading resources...")

# region ===== text preprocessing =====
def preprocess_text(text):
    words = text.split()
    processed_words = [word.lower() for word in words]
    processed_words = [word.strip() for word in processed_words]
    processed_words = [re.sub('\s+',' ', word) for word in processed_words]
    processed_words = [word for word in processed_words if not all(char in string.punctuation for char in word.replace(' ',''))]
    
    processed_words = ' '.join(processed_words)
    
    return ' '.join(words)

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

# endregion

# region ===== data preparation =====
def create_padded_seq(text, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')[0]
# endregion

# region ===== chat app =====
def chat():
    print(r""" 
    =================================================      
    Campus Information Chatbot - President University
    =================================================
    """)
    print("Type 'quit' at any time to exit")

    while True:
        try:
            user_input = input("You: ").strip() 
            
            if user_input.lower() == 'quit':
                print("\nBot: Thank you for using the campus chatbot. Goodbye!")
                break
                
            if not user_input:
                print("Bot: Please enter a valid question.")
                continue
            
            # preprocessing
            cleaned_input = preprocess_text(user_input)
            tagged_input = apply_ner_tags(cleaned_input)
            
            # data preparation
            seq = create_padded_seq(tagged_input, tokenizer, MAX_LEN)
            model_input = np.expand_dims(seq, axis=0)
            
            # predict tag
            prediction = model.predict(model_input, verbose=0)[0]
            tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            
            # get answer baesd on tag
            for intent in intents_data['intents']:
                if intent['tag'] == tag:
                    response = random.choice(intent['responses'])
                    print("Bot:", response)
                    print("-"*50)  # Visual separator
                    break
                    
        except KeyboardInterrupt:
            print("\nBot: Session ended by user. Goodbye!")
            break
            
        except Exception as e:
            print(f"\nBot: I encountered an error. Please try rephrasing your question. (Error: {str(e)})")

# endregion

if __name__ == "__main__":
    chat()
    
