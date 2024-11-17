from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and necessary files (word_to_index and index_to_word)
model = load_model('C:/Users/Asus/OneDrive/Desktop/My-Project/lstm_model.h5')

with open('C:/Users/Asus/OneDrive/Desktop/My-Project/word_to_index.pkl', 'rb') as f:
  word_to_index = pickle.load(f)

with open('C:/Users/Asus/OneDrive/Desktop/My-Project/index_to_word.pkl', 'rb') as f:
  index_to_word = pickle.load(f)


# Function to generate a story based on the seed text
def generate_story(seed_text, model, word_to_index, index_to_word, max_sequence_length=5, num_words_to_generate=50, temperature=1.0, prob_tolerance=1e-5):
    tokenized_seed = [word_to_index.get(word, 0) for word in seed_text.split()]
    padded_seed = tokenized_seed
    if len(padded_seed) < max_sequence_length:
        padded_seed = [0] * (max_sequence_length - len(padded_seed)) + padded_seed
    padded_seed = np.array(padded_seed).reshape(1, -1)
    
    generated_story = seed_text
    for _ in range(num_words_to_generate):
        predicted_probs = model.predict(padded_seed, verbose=0)
        predicted_probs = predicted_probs[0, -1, :]  # Probabilities for the next word
        
        # Apply temperature scaling
        predicted_probs = predicted_probs / temperature
        predicted_probs = np.exp(predicted_probs)
        predicted_probs = predicted_probs / np.sum(predicted_probs)
        
        if np.abs(np.sum(predicted_probs) - 1.0) > prob_tolerance:
            predicted_probs = predicted_probs / np.sum(predicted_probs)

        predicted_word_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        predicted_word = index_to_word.get(predicted_word_index, '')

        if predicted_word == '':
            break
        
        generated_story += ' ' + predicted_word
        padded_seed = np.roll(padded_seed, -1, axis=1)
        padded_seed[0, -1] = predicted_word_index

    return generated_story

# Route for rendering the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for rendering the test page
@app.route('/test')
def test():
    return render_template('test.html')

# Route for generating a story based on the user input
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required to generate a story.'})

    try:
        generated_story = generate_story(prompt, model, word_to_index, index_to_word)
        return jsonify({'story': generated_story})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

