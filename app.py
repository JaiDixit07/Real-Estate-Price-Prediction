from flask import Flask, request, render_template, jsonify
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__, static_folder='static')
app = application

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
if request.method == 'POST':
        try:
            # Collect input data from the form
            location = request.form.get('location')
            total_sqft = float(request.form.get('total_sqft'))
            bath = float(request.form.get('bath'))
            bhk = int(request.form.get('bhk'))
            # Create an instance of CustomData
            data = CustomData(
                location=location,
                total_sqft=total_sqft,
                bath=bath,
                bhk=bhk
            )
            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            # Load prediction pipeline
            predict_pipeline = PredictPipeline()
            # Make predictions
            results = predict_pipeline.predict(pred_df)
            # Return the result as JSON
            return jsonify({'prediction_result': f" ₹ {results[0]:.2f} LAKH "})
        except Exception as e:
            # Catch and log any errors for debugging
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR: {error_details}")
            return jsonify({'error': str(e), 'details': error_details}), 500

    # If it's a GET request, return the template
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
