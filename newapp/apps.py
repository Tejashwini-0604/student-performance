from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping function using your GPA-based rules
def map_grade_class(prediction):
    if prediction >= 3.5:
        return 'A'
    elif 3.0 <= prediction < 3.5:
        return 'B'
    elif 2.5 <= prediction < 3.0:
        return 'C'
    elif 2.0 <= prediction < 2.5:
        return 'D'
    else:
        return 'F'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and convert form inputs to appropriate types
        features = [
            int(request.form['Age']),
            int(request.form['Gender']),
            int(request.form['Ethnicity']),
            int(request.form['ParentalEducation']),
            float(request.form['StudyTimeWeekly']),
            int(request.form['Absences']),
            int(request.form['Tutoring']),
            int(request.form['ParentalSupport']),
            int(request.form['Extracurricular']),
            int(request.form['Sports']),
            int(request.form['Music']),
            int(request.form['Volunteering']),
            float(request.form['GPA'])
        ]
    except ValueError:
        return render_template('index.html', prediction_text="Please enter valid values for all fields.")

    # Convert features into numpy array for the model
    features_array = np.array([features])

    # Make a numeric prediction using the model
    numeric_prediction = model.predict(features_array)[0]

    # Map numeric prediction to a letter grade based on GPA classification rules
    grade_prediction = map_grade_class(numeric_prediction)

    # Render the result on the webpage
    return render_template('index.html', prediction_text=f'Predicted Grade: {grade_prediction}')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
