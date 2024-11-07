from flask import Flask, request, render_template
import joblib

# Load the model and vectorizer
model = joblib.load('crime_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a preprocessing function (same as used during training)
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    
    nltk.download('stopwords', quiet=True)
    stemmer = PorterStemmer()
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word.lower() not in stopwords.words('english')]
    return " ".join(filtered_words)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        crime_description = request.form['crime_description']
        
        # Preprocess and transform the input text
        preprocessed_text = preprocess_text(crime_description)
        transformed_text = vectorizer.transform([preprocessed_text])
        
        # Make a prediction
        prediction = model.predict(transformed_text)[0]
        
        # Render the result back to the UI
        return render_template('index.html', prediction=prediction, crime_description=crime_description)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
