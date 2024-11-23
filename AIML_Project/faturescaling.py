from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target
feature_names = wine_data.feature_names
target_names = wine_data.target_names

# Define the scaler and encoder for scaling features
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False)

# Convert the dataset to a pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Feature scaling: Apply standard scaling to the features
scaled_X = scaler.fit_transform(X)
scaled_df = pd.DataFrame(scaled_X, columns=feature_names)

# One-Hot Encoding: The target variable is a categorical label, so we need to one-hot encode it
encoded_target = encoder.fit_transform(y.reshape(-1, 1))

# Function to create a graph (plot)
def create_plot():
    # Plotting the scaled features (showing the first two features for simplicity)
    plt.figure(figsize=(8, 6))
    plt.scatter(scaled_df[feature_names[0]], scaled_df[feature_names[1]], c=y, cmap='viridis')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Scaled Features Plot")
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return plot_url

# Function to generate the table in HTML format
def generate_table():
    return df.head().to_html(classes='table table-striped')

# Function to generate scaled table in HTML format
def generate_scaled_table():
    return scaled_df.head().to_html(classes='table table-striped')

# Function to generate one-hot encoded table in HTML format
def generate_encoded_table():
    encoded_df = pd.DataFrame(encoded_target, columns=[f"Class {i}" for i in target_names])
    return encoded_df.head().to_html(classes='table table-striped')

# Function to get bot responses based on input
def get_bot_response(user_input):
    user_input = user_input.lower()

    # Matching keywords for different queries
    if "features" in user_input or "columns" in user_input or "wine dataset" in user_input:
        return "The features in the wine dataset include 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total Phenols', 'Flavanoids', 'Nonflavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315 of Diluted Wines', 'Proline'."
    
    elif "scaling" in user_input or "standardization" in user_input:
        return f"Feature scaling normalizes the data to a standard range. Standardization ensures the features have a mean of 0 and a standard deviation of 1. Here's a graph for the first two scaled features below."
    
    elif "encoding" in user_input or "one-hot encoding" in user_input:
        return "One-hot encoding is applied to the target variable. Here's an example of the one-hot encoded target:\n" + str(encoded_target[:5])
    
    elif "show" in user_input and "data" in user_input:
        return generate_table()  # Return the first 5 rows as a table in HTML
    
    elif "show" in user_input and "graph" in user_input:
        plot_url = create_plot()
        return f'<img src="data:image/png;base64,{plot_url}" alt="Scaled Feature Plot">'
    
    elif "scaled" in user_input and "table" in user_input:
        return generate_scaled_table()  # Return the first 5 rows of the scaled data
    
    elif "encoded" in user_input and "table" in user_input:
        return generate_encoded_table()  # Return the first 5 rows of the one-hot encoded data
    
    elif "overfitting" in user_input:
        return "Overfitting happens when a model learns the training data too well, including its noise, which harms its performance on unseen data."
    
    else:
        return "I'm here to assist you with feature scaling, one-hot encoding, and the wine dataset. Ask me questions about features, scaling, encoding, or the model!"

# Home route for the webpage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle chatbot responses
@app.route('/get', methods=['GET', 'POST'])
def chatbot():
    user_input = request.args.get('msg')
    response = get_bot_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
