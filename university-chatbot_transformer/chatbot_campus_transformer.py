print("Initializing Campus Information Chatbot...")

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import spacy
import re 
import string

# variable initialization
TOKENIZER_PATH = 'chatbot_campus_transformer_tokenizer'
MODEL_PATH = 'chatbot_campus_transformer_model'
MAX_LENGTH = 40

print("loading resources...")
print("-- loading tokenizer")
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
VOCAB_SIZE = tokenizer.vocab_size + 2

print("-- loading spaCy model...")
nlp = spacy.load("en_core_web_lg")
print("done loading resources...")

# region ===== data preprocessing =====
def preprocess_text_per_word(text):
    words = text.split()
    processed_words = [word.lower() for word in words]
    processed_words = [word.strip() for word in processed_words]
    processed_words = [re.sub('\s+',' ', word) for word in processed_words]
    processed_words = [word for word in processed_words if not all(char in string.punctuation for char in word.replace(' ',''))]
    
    processed_words = ' '.join(processed_words)
    return processed_words

def apply_ner_tags(text):
    doc = nlp(text)
    
    tagged_tokens = []
    for token in doc:
        replaced = False
        for ent in doc.ents:
            if token.text == ent.text:
                tagged_tokens.append(f"<{ent.label_}>")
                replaced = True
                break
        if not replaced:
            tagged_tokens.append(token.text)
    return " ".join(tagged_tokens)
# endregion

# region ===== data preparation =====
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(seq):
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(seq)
    return tf.maximum(look_ahead_mask, padding_mask)
# endregion

# region ===== model components =====
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps

    def get_config(self):
        return {"d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
# endregion

# region ===== model loading  =====
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionalEncoding": PositionalEncoding,
        "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
        "CustomSchedule": CustomSchedule,
    },
    compile=False
)
# endregion

# region ===== prediction =====
def evaluate(sentence):
    sentence = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    encoder_input = tf.expand_dims(sentence, 0)

    decoder_input = [START_TOKEN[0]]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[encoder_input, output], training=False)
        predictions = predictions[:, -1:, :]  # last token
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == END_TOKEN[0]:
            break

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence):
    prediction = evaluate(sentence).numpy()
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    return predicted_sentence
# endregion

if __name__ == "__main__":
    print(r""" 
    =================================================      
    Campus Information Chatbot - President University
    =================================================
    """)
    
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Bye!")
            print("="*50)
            break
        
        # preprocessing
        preprocessed_question = preprocess_text_per_word(user_input)
        preprocessed_question = apply_ner_tags(preprocessed_question)

        response = predict(preprocessed_question)
        print(f"Bot: {response}")
