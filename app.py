from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

###############################################

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        review = request.form.get('review')

        data_point = [review]
        
        model = joblib.load("model/naive_bayes.pkl")
        
        prediction = model.predict(data_point)

        if prediction[0] == 1:
            output = "Positive Review"
        else:
            output = "Negative Review"

        return render_template("prediction.html", review=review, prediction=output)

    return render_template('prediction.html', prediction="")

#####################################################

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)