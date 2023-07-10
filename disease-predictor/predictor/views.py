from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from predictor.forms import BreastCancerForm, DiabetesForm, HeartDiseaseForm
from .models import *
from django.contrib.auth.models import User
from predictor.forms import *


def heart(request):

    df = pd.read_csv('static/Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]

    value = ''

    if request.method == 'POST':

        # age, sex, cp = map(float, request.POST.get())

        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])  # chest pain
        trestbps = float(request.POST['trestbps'])  # resting blood pressure
        chol = float(request.POST['chol'])  # serun cholestrol in mg/dl
        fbs = float(request.POST['fbs'])  # fasting blood sugar in > 120 mg/dl
        # resting electrocardiographic values(0,1,2)
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])  # maximum heart rate achieved
        exang = float(request.POST['exang'])  # exercise induced angina
        # ST depression induced by exercise
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])  # Slope of peak exercise
        # number of major vessels (0-3) colored by flourosopy
        ca = float(request.POST['ca'])
        # 0 = normal; 1 = fixed defect; 2 = reversable defect
        thal = float(request.POST['thal'])

        user = HeartDisease(age=age , sex = sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                            thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
        user.save()

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        print(user_data)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        calculated_prediction = int(predictions[0])
        value = 'have' if calculated_prediction == 1 else "don't have"

        # if int(predictions[0]) == 1:
        #     value = 'have'
        # elif int(predictions[0]) == 0:
        #     value = "don\'t have"

    return render(request,
                  'heart.html',
                  {
                      'result': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-black',
                      'heart': True,
                      'form': HeartDiseaseForm(),
                  })


def diabetes(request):

    dfx = pd.read_csv('static/Diabetes_XTrain.csv')
    dfy = pd.read_csv('static/Diabetes_YTrain.csv')
    X = dfx.values
    Y = dfy.values
    Y = Y.reshape((-1,))

    value = ''
    if request.method == 'POST':

        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        bloodpressure = float(request.POST['bloodpressure'])
        skinthickness = float(request.POST['skinthickness'])
        bmi = float(request.POST['bmi'])
        insulin = float(request.POST['insulin'])
        pedigree = float(request.POST['pedigree'])
        age = float(request.POST['age'])

        user = Diabetes(pregnancies=pregnancies, glucose=glucose, bloodpressure=bloodpressure,
                        skinthickness=skinthickness, bmi=bmi, insulin=insulin, pedigree=pedigree, age=age)
        user.save()

        user_data = np.array(
            (pregnancies,
             glucose,
             bloodpressure,
             skinthickness,
             bmi,
             insulin,
             pedigree,
             age)
        ).reshape(1, 8)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, Y)

        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'diabetes.html',
                  {
                      'result': value,
                      'title': 'Diabetes Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'diabetes': True,
                      'form': DiabetesForm(),
                  }
                  )


def breast(request):

    df = pd.read_csv('static/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)

    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        # user = User.objects.create_user(radius = radius, texture=texture, perimeter = perimeter ,area = area, smoothness = smoothness)
        user = BreastCancer(radius=radius, texture=texture,
                            perimeter=perimeter, area=area, smoothness=smoothness)
        user.save()

        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)

        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        print(user_data)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = "have"

        elif int(predictions[0]) == 0:
            value = "dont have"

    return render(request,
                  'breast.html',
                  {
                      'result': value,
                      'title': 'Breast Cancer Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'breast': True,
                      'form': BreastCancerForm(),
                  })


def home(request):

    name = BreastCancer.objects.all()

    return render(request,
                  'home.html', {'name': name})


def handler404(request):
    return render(request, '404.html', status=404)
