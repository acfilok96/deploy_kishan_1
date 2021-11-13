from flask import Flask, render_template,request
import numpy as np
import pickle

app = Flask(__name__)

classifier_model = pickle.load(open("classifier_model.pkl","rb"))


def pred_function(array_ok):
    array_ok = np.array(array_ok)
    array_ok = array_ok.reshape(1,-1)
    pred_1 = classifier_model.predict(array_ok)
    if(pred_1 == 0):
        return "salary is less than 50,000"
    return  "salary is more than 50,000"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result_page", methods=["POST"])
def about():
    if request.method == "POST":
        array =[]

        workclass = int(request.form["workclass"])
        array.append(workclass)

        education = int(request.form["education"])
        array.append(education)

        marital_status = int(request.form["marital-status"])
        array.append(marital_status)

        occupation = int(request.form["occupation"])
        array.append(occupation)

        relationship = int(request.form["relationship"])
        array.append(relationship)
        
        race = int(request.form["race"])
        array.append(race)
        
        sex = int(request.form["sex"])
        array.append(sex)

        country = int(request.form["country"])
        array.append(country)

        age = int(request.form["age"])
        array.append(age)

        fnlwgt = int(request.form["fnlwgt"])
        array.append(fnlwgt)

        education_num = int(request.form["education-num"])
        array.append(education_num)

        capital_gain = int(request.form["capital-gain"])
        array.append(capital_gain)
        
        capital_loss = int(request.form["capital-loss"])
        array.append(capital_loss)
        
        hours_per_week = int(request.form["hours-per-week"])
        array.append(hours_per_week)

        predc = pred_function(array)

        return render_template("index.html", prediction = predc)



if __name__ == "__main__":
    app.run(debug=True)

