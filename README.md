# 🏡 Real Estate Price Prediction 💼

This project predicts real estate prices in **Bangalore** with a smooth and interactive experience. Just input details like location and square footage, and let the models do the magic! 💫

[🎉 Live Demo](https://real-estate-price-prediction-dldi.onrender.com) – Try it out now!

## 🚀 Features

- **Smart Data Prep**: Cleans and transforms the raw data for peak performance.
- **Powerful Models**: Combines the forces of **Random Forest**, **Gradient Boosting**, and **XGBoost** for super-accurate predictions.
- **Flask Interface**: Easy-to-use, sleek UI powered by Flask for seamless interactions.
- **Deployed on Render**: Accessible online for real-time predictions from anywhere!

## 🏗️ Project Structure

- **`Data/`**: Raw and cleaned data files.
- **`src/`**: Includes trained models and prediction scripts in modular coding format
- **`Templates/`**: HTML templates for the Flask application’s front end.
- **`App/`**: Flask app setup for serving the model and interface.
- **`static/`**: CSS file for styling 
- **`requirements/`**: All the necessary libraries

## 🔍 Data

The dataset used is the **Bangalore House Price Prediction dataset**, featuring:
- Location
- Size (square footage)
- Number of Bedrooms/Bathrooms
- Other relevant features

## 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/JaiDixit07/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction
```

Install Dependencies
Ensure you have Python 3.8+ installed, then install the required Python packages with:

```bash
pip install -r requirements.txt
```

##  Required Libraries

- setuptools
- numpy
- pandas
- matplotlib
- seaborn
- fuzzywuzzy
- scikit-learn
- xgboost
- scipy
- dill
- flask
- gunicorn


# 🖥️ Usage
To run the application locally:

Clone the repository and install dependencies.
Run the Flask app:
```bash
python app.py
```
The app will be available at http://localhost:5000. From there, you can input the necessary details (location, square footage, etc.) and get real-time price predictions.

## 🔧 Customization
You can tweak the model by experimenting with various features or applying different machine learning algorithms. Check the notebook files for exploratory data analysis (EDA) and modeling details.

## 📊 Visualizations
The project provides multiple visualizations for data analysis, including distribution plots and correlation heatmaps, to better understand feature relationships.

## 📁 Notebooks
- **EDA Notebook**: Analyzes the dataset for insights and visualizations.
- **Model Training Notebook**: Contains the scripts for model training, tuning, and evaluation.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository, submit issues, or create pull requests.

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.

## 🙌 Acknowledgements
- **Scikit-learn** and **XGBoost** for providing the libraries used in modeling.
- **Flask** for the web framework powering the user interface.
- **onrender** for the deployment 

-------------------------------------------------------------------------------------------------

Developed by Jai Dixit.