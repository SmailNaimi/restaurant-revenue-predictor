from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pickle
import pandas as pd

app = Flask(__name__)
# Load the model and scaler
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# Serve index.html for GET requests and handle predictions for POST requests
@app.route('/')
def home():
    return render_template('home.html')  # Assurez-vous de nommer le fichier HTML 'home.html'

# Route pour servir la page HTML
@app.route('/Team', methods=['GET'])
def team():
    return render_template('team.html')  # Retourne la page avec le bouton

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')  # Page du formulaire à remplir
    elif request.method == 'POST':
        try:
            # Get JSON data from request
            data = request.get_json() or request.form

            # Define the required features
            features = ['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                        'Cuisine_Type', 'Average_Customer_Spending', 
                        'Promotions', 'Reviews']

            # Check if all required features are in the request
            if not all(key in data for key in features):
                return jsonify({"error": "Missing input data"}), 400

            # Convert data to DataFrame for scaling
            input_data = pd.DataFrame([data])

            # Separate binary and non-binary columns
            binary_data = input_data[['Promotions']]
            non_binary_data = input_data.drop(['Promotions'], axis=1)

            # Scale non-binary columns
            nb_scaled = scaler.transform(non_binary_data)
            scaled_df = pd.DataFrame(nb_scaled, columns=non_binary_data.columns)

            # Combine scaled and binary data and ensure column order
            final_input = pd.concat([scaled_df, binary_data], axis=1)
            final_input = final_input[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                                       'Cuisine_Type', 'Average_Customer_Spending', 
                                       'Promotions', 'Reviews']]

            # Make prediction
            prediction = model.predict(final_input)

            # Return prediction result
            return jsonify({"predicted_Monthly_Revenue": prediction[0][0]})

        except Exception as e:
            # Handle any errors and return a message
            return jsonify({"error": str(e)}), 500



# Route pour servir la page HTML
@app.route('/About', methods=['GET'])
def about():
    return render_template('about.html')  # Retourne la page avec le bouton
@app.route('/APIhealth', methods=['GET'])
def api_health_page():
    return render_template('APIstatus.html')  # Retourne la page avec le bouton

# Route pour vérifier l'état de l'API
@app.route('/check_status', methods=['GET'])
def check_status():
    if not model_loaded:
        return jsonify({"status": "API failed to load model or scaler", "error": load_error}), 500

    try:
        # Données fictives pour vérifier l'état
        test_data = pd.DataFrame([{
            'Number_of_Customers': 100, 'Menu_Price': 20.0, 'Marketing_Spend': 500.0, 
            'Cuisine_Type': 1, 'Average_Customer_Spending': 30.0, 
            'Promotions': 1, 'Reviews': 10
        }])
        
        # Séparation des colonnes binaires et non-binaires pour le scaling
        binary_data = test_data[['Promotions']]
        non_binary_data = test_data.drop(['Promotions'], axis=1)
        
        # Application du scaling
        nb_scaled = scaler.transform(non_binary_data)
        scaled_df = pd.DataFrame(nb_scaled, columns=non_binary_data.columns)
        
        # Combinaison des données et respect de l'ordre des colonnes
        final_input = pd.concat([scaled_df, binary_data], axis=1)
        final_input = final_input[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
                                   'Cuisine_Type', 'Average_Customer_Spending', 
                                   'Promotions', 'Reviews']]
        
        # Effectuer une prédiction pour vérifier le fonctionnement
        test_prediction = model.predict(final_input)
        
        return jsonify({
            "status": "API is running without any problems",
            "test_prediction": test_prediction[0][0]
        })
    
    except Exception as e:
        return jsonify({"status": "API has an error", "error": str(e)}), 500
    
@app.route('/send-message', methods=['POST'])
def send_message():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        try:
            # Enregistrer le message dans un fichier texte
            with open('messages.txt', 'a') as file:
                file.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n{'-'*40}\n")
            
            return "Message sent successfully!"  # Message simple sans utiliser de session

        except Exception as e:
            print(f"Failed to send email: {e}")
            return "Failed to send message. Please try again later."

if __name__ == '__main__':
    app.run(debug=True)
