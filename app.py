from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pickle
import pandas as pd

app = Flask(__name__)
# Load the model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))  # Load the trained model from 'model.pkl'
    scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the scaler from 'scaler.pkl' to scale input data
    model_loaded = True
except Exception as e:
    model_loaded = False  # Set flag if model or scaler failed to load
    load_error = str(e)  # Store the error message for debugging

# Serve index.html for GET requests and handle predictions for POST requests
@app.route('/')
def home():
    return render_template('home.html')  # Render the home page (assumes 'home.html' exists)

# Route to serve the team page
@app.route('/Team', methods=['GET'])
def team():
    return render_template('team.html')  # Render the team page (assumes 'team.html' exists)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')  # Serve the prediction form page
    elif request.method == 'POST':
        try:
            # Get JSON data from request (in case the request comes from an API client)
            data = request.get_json() or request.form  # Supports both JSON payload and form submission

            # Define the required features for prediction
            features = ['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                        'Cuisine_Type', 'Average_Customer_Spending', 
                        'Promotions', 'Reviews']

            # Check if all required features are present in the request
            if not all(key in data for key in features):
                return jsonify({"error": "Missing input data"}), 400

            # Convert input data to a DataFrame to enable scaling
            input_data = pd.DataFrame([data])

            # Separate binary and non-binary columns for appropriate handling
            binary_data = input_data[['Promotions']]
            non_binary_data = input_data.drop(['Promotions'], axis=1)

            # Scale non-binary columns using the pre-trained scaler
            nb_scaled = scaler.transform(non_binary_data)
            scaled_df = pd.DataFrame(nb_scaled, columns=non_binary_data.columns)

            # Combine scaled non-binary data with binary data and ensure proper column order
            final_input = pd.concat([scaled_df, binary_data], axis=1)
            final_input = final_input[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                                       'Cuisine_Type', 'Average_Customer_Spending', 
                                       'Promotions', 'Reviews']]

            # Make prediction using the loaded model
            prediction = model.predict(final_input)

            # Return the predicted value in the response
            return jsonify({"predicted_Monthly_Revenue": prediction[0][0]})

        except Exception as e:
            # Handle any exceptions that occur during the prediction process
            return jsonify({"error": str(e)}), 500

# Route to serve the about page
@app.route('/About', methods=['GET'])
def about():
    return render_template('about.html')  # Render the about page (assumes 'about.html' exists)

# Route to serve the API status page
@app.route('/APIhealth', methods=['GET'])
def api_health_page():
    return render_template('APIstatus.html')  # Render the API status page (assumes 'APIstatus.html' exists)

# Route to check the health status of the API and model
@app.route('/check_status', methods=['GET'])
def check_status():
    if not model_loaded:
        return jsonify({"status": "API failed to load model or scaler", "error": load_error}), 500

    try:
        # Create dummy data to verify model functionality
        test_data = pd.DataFrame([{
            'Number_of_Customers': 100, 'Menu_Price': 20.0, 'Marketing_Spend': 500.0, 
            'Cuisine_Type': 1, 'Average_Customer_Spending': 30.0, 
            'Promotions': 1, 'Reviews': 10
        }])
        
        # Separate binary and non-binary columns to apply scaling
        binary_data = test_data[['Promotions']]
        non_binary_data = test_data.drop(['Promotions'], axis=1)
        
        # Scale the non-binary columns using the pre-trained scaler
        nb_scaled = scaler.transform(non_binary_data)
        scaled_df = pd.DataFrame(nb_scaled, columns=non_binary_data.columns)
        
        # Combine scaled non-binary and binary data, ensuring proper column order
        final_input = pd.concat([scaled_df, binary_data], axis=1)
        final_input = final_input[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                                   'Cuisine_Type', 'Average_Customer_Spending', 
                                   'Promotions', 'Reviews']]
        
        # Make a prediction to verify model functionality
        test_prediction = model.predict(final_input)
        
        return jsonify({
            "status": "API is running without any issues",
            "test_prediction": test_prediction[0][0]
        })
    
    except Exception as e:
        return jsonify({"status": "API has an error", "error": str(e)}), 500

# Route to handle sending messages from the contact form
@app.route('/send-message', methods=['POST'])
def send_message():
    if request.method == 'POST':
        # Get form data from the request
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        try:
            # Save the message data to a text file for future reference
            with open('messages.txt', 'a') as file:
                file.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n{'-'*40}\n")
            
            return "Message sent successfully!"  # Send success message

        except Exception as e:
            print(f"Failed to send email: {e}")
            return "Failed to send message. Please try again later."

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Run on all available IP addresses, port 8080
