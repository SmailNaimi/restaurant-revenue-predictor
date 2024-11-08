# 🌟 Restaurant Revenue Prediction Application

This application predicts monthly revenue for restaurants based on various inputs, such as customer numbers, menu pricing, marketing spend, and customer reviews. 🚀 The application is powered by a machine learning model, which is deployed as a web service using Flask. This README provides detailed instructions on setting up, running, and testing the application, as well as an explanation of its API endpoints.

## 📋 Table of Contents
- [📖 Overview](#-overview)
- [🔧 Prerequisites](#-prerequisites)
- [📁 Project Structure](#-project-structure)
- [⚙️ Setup and Installation](#%EF%B8%8F-setup-and-installation)
- [🚀 Usage](#-usage)
  - [1. Training the Model](#training-the-model)
  - [2. Running the Application](#running-the-application)
  - [3. Testing the Application](#testing-the-application)
  - [4. Containerized Deployment with Docker](#containerized-deployment-with-docker)
  - [5. Environment Variables](#environment-variables)
- [🔌 API Endpoints](#-api-endpoints)
- [🔄 Data Flow Explanation](#-data-flow-explanation)
- [🧠 Model Explanation](#-model-explanation)
- [🛠️ Troubleshooting](#%EF%B8%8F-troubleshooting)
- [📜 License](#-license)
- [📞 Contact](#-contact)

## 📖 Overview

The Restaurant Revenue Prediction Application is designed to assist restaurant owners in forecasting their monthly revenue by providing predictions based on several factors including the number of customers, menu pricing, marketing spend, and more. 📊 The machine learning model used in this application is a linear regression model, trained on sample restaurant data to understand the relationship between these factors and monthly revenue. This tool allows users to make data-driven decisions to optimize their restaurant's profitability.

## 🔧 Prerequisites

1. **Python 3.x**: Ensure Python is installed. You can verify this by running `python --version`.
2. **Docker**: For containerized deployment, Docker must be installed and running.
3. **Virtual environment** (optional but recommended): Helps to manage dependencies.

## 📁 Project Structure

- **`app.py`**: Main Flask application file that serves web pages and handles API requests.
- **`model.py`**: Script for training and saving the machine learning model.
- **`requirements.txt`**: Lists all dependencies required for the application.
- **`Dockerfile`**: Instructions to build a Docker image for the application.
- **HTML Files**:
  - **`home.html`**: 🏠 The homepage of the application that provides a start button.
  - **`about.html`**: ℹ️ About page explaining the purpose and benefits of the project.
  - **`predict.html`**: 📊 A page where users can input data for revenue prediction.
  - **`team.html`**: 👥 Information about the team members behind the project.
  - **`APIstatus.html`**: ✅ Displays the status of the API and model health.
- **`static/`**: Directory containing static files like images.
- **`templates/`**: Directory containing HTML templates for rendering web pages.
- **`model.pkl`**: Serialized version of the trained machine learning model.
- **`scaler.pkl`**: Serialized data scaler used for preprocessing inputs.

## ⚙️ Setup and Installation

1. **Clone the Repository**  
   Clone this repository to your local machine using the command:
   ```bash
   git clone https://github.com/SmailNaimi/restaurant-revenue-predictor
   cd RestaurantRevenuePrediction
   ```

2. **Create a Virtual Environment** (optional)  
   Creating a virtual environment is recommended to manage dependencies without polluting the global Python environment.
   ```bash
   python3 -m venv env
   source env/bin/activate  #On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**  
   Install the necessary Python packages using the following command:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### 1. Training the Model

To train the machine learning model, ensure that `Restaurant_revenue.csv` is available in the project folder. 📂 This CSV file contains the training data needed to fit the model. Run the following command:
```bash
python model.py
```
This script will train the model using a linear regression approach and save both the trained model (`model.pkl`) and the scaler (`scaler.pkl`) for later use. 💾

### 2. Running the Application

Once the model has been trained, you can start the Flask web application by running:
```bash
python app.py
```
By default, the application will be accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000). 🌐

### 3. Testing the Application

**1. Unit Tests**: 🧪 You can create and execute unit tests to validate the core functionalities in `app.py` and `model.py`. Consider using `unittest` or `pytest` frameworks.

**2. API Testing**: Use tools like `Postman` or `cURL` to test the API endpoints. You can send a POST request to `/predict` with the following JSON payload:
```json
{
    "Number_of_Customers": 400,
    "Menu_Price": 90.0,
    "Marketing_Spend": 5000.0,
    "Cuisine_Type": 1,
    "Average_Customer_Spending": 70.0,
    "Promotions": 1,
    "Reviews": 10
}
```
Expected response:
```json
{
    "predicted_Monthly_Revenue": "predicted_value"
}
```

### 4. Containerized Deployment with Docker

1. **Build the Docker Image** 🐳  
   To build a Docker image for the application, use the following command:
   ```bash
   docker build -t restaurant-revenue-app .
   ```

2. **Run the Docker Container**  
   After building the Docker image, run it with:
   ```bash
   docker run -p 5000:5000 restaurant-revenue-app
   ```
   The application will now be accessible at [http://localhost:5000](http://localhost:5000). 🌐

### 5. Environment Variables

If your application requires environment variables, you can specify them in a `.env` file or pass them during the Docker run command:
```bash
docker run --env-file .env -p 5000:5000 restaurant-revenue-app
```

## 🔌 API Endpoints

- **`GET /`**: Renders the home page (`home.html`).
- **`GET /About`**: Renders the about page (`about.html`).
- **`GET /Team`**: Renders the team page (`team.html`).
- **`GET /predict`**: Renders the prediction input form (`predict.html`).
- **`POST /predict`**: Accepts JSON or form data, preprocesses the input, and returns the predicted monthly revenue.
- **`GET /check_status`**: Checks and returns the health status of the model and API.
- **`POST /send-message`**: Handles user messages submitted via the contact form on the team page.

## 🔄 Data Flow Explanation

1. **Input Collection** ✍️: The user inputs data on the prediction page (`predict.html`).
2. **Data Preprocessing** 🔄: The inputs are preprocessed using the saved `scaler.pkl` to ensure that all numerical values are standardized.
3. **Model Prediction** 🤖: The preprocessed data is fed into the trained model (`model.pkl`) to generate a revenue prediction.
4. **Output** 📈: The predicted revenue is returned to the user in JSON format or displayed on the web page.

## 🧠 Model Explanation

The machine learning model used in this application is a **Linear Regression** model, which is ideal for predicting continuous numerical values. 📉 The model was trained on features such as:
- **Number of Customers**: The estimated number of customers visiting the restaurant.
- **Menu Price**: Average pricing of menu items.
- **Marketing Spend**: Monthly spending on marketing activities.
- **Cuisine Type**: Encoded categorical variable representing the type of cuisine offered.
- **Average Customer Spending**: The average spending per customer.
- **Promotions**: Binary feature indicating if promotions are being run.
- **Reviews**: Number of customer reviews.

During training, the data was preprocessed to handle categorical values, and numerical features were scaled for better performance. The model's performance was evaluated using metrics such as **R-squared** and **RMSE (Root Mean Square Error)**.

## 🛠️ Troubleshooting

- **Model or Scaler Not Loaded**: Ensure `model.pkl` and `scaler.pkl` are present in the root folder.
- **Environment Issues**: If dependencies are missing, verify the `requirements.txt` file and re-install the dependencies.
- **Docker Issues**: Ensure Docker is running and accessible.
- **Port Conflicts**: Ensure port 5000 is not being used by another service.
- **Data Format Errors**: Make sure all input fields are properly filled and are in the expected format.


## 📞 Contact

For further queries, please contact me at [Link](https://www.linkedin.com/in/ismail-naimi-3a0a082b4/). 🤝

