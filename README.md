Overview
The AI Data Preprocessing Assistant is a web-based chatbot application designed to assist with data preprocessing tasks such as feature scaling, one-hot encoding, and exploratory data analysis using the Wine dataset from Scikit-learn. The chatbot offers an interactive way to query the dataset, visualize data, and understand preprocessing techniques.

Features
Interactive Chatbot
A conversational chatbot to answer queries related to the Wine dataset, feature scaling, and one-hot encoding.

Feature Scaling
Standardizes numerical features using StandardScaler to ensure all features have a mean of 0 and a standard deviation of 1.

One-Hot Encoding
Encodes the categorical target variable into a binary matrix for machine learning compatibility.

Data Visualization
Generates graphs for scaled features and displays sample data tables.

Tech Stack
Backend: Flask (Python)
Frontend: HTML, CSS, JavaScript, Bootstrap
Libraries Used:
scikit-learn for preprocessing
pandas for data handling
matplotlib for data visualization
Commands Supported
Queries about the dataset:
"What are the features in the Wine dataset?"
"List the columns in the dataset."
"Tell me about the Wine dataset."
Feature scaling and encoding:
"What is feature scaling?"
"Explain scaling and standardization."
"Show a graph of scaled features."
"What is one-hot encoding?"
"Show an example of one-hot encoded data."
Data display:
"Show me the first few rows of the data."
"Show the scaled table."
"Show the encoded table."
General questions:
"What is overfitting?"
"Help"
Setup Instructions
Prerequisites
Python 3.8 or above installed
Required Python packages:
Flask
scikit-learn
pandas
matplotlib
