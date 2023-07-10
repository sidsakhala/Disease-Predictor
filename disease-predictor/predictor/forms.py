from django import forms


class BreastCancerForm(forms.Form):

    radius = forms.FloatField(label='Mean Radius', min_value=0, max_value=10000,
                              widget=forms.NumberInput(attrs={'class': 'form-control'}))
    texture = forms.FloatField(label='Mean Texture', min_value=0, max_value=10000,
                               widget=forms.NumberInput(attrs={'class': 'form-control'}))
    perimeter = forms.FloatField(label='Mean Perimeter', min_value=0, max_value=10000,
                                 widget=forms.NumberInput(attrs={'class': 'form-control'}))
    area = forms.FloatField(label='Mean Area', min_value=0, max_value=10000,
                            widget=forms.NumberInput(attrs={'class': 'form-control'}))
    smoothness = forms.FloatField(label='Mean Smoothness', min_value=0,
                                  max_value=10000, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class DiabetesForm(forms.Form):

    pregnancies = forms.FloatField(label='Pregnancies: To express the Number of pregnancies',
                                   min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    glucose = forms.FloatField(label='Glucose: To express the Glucose level in blood',
                               min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    bloodpressure = forms.FloatField(label='BloodPressure: To express the Blood pressure measurement',
                                     min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    skinthickness = forms.FloatField(label='SkinThickness: To express the thickness of the skin',
                                     min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    insulin = forms.FloatField(label='Insulin: To express the Insulin level in blood',
                               min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    bmi = forms.FloatField(label='BMI: To express the Body mass index', min_value=0,
                           max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    pedigree = forms.FloatField(label='DiabetesPedigreeFunction: To express the Diabetes percentage',
                                min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    age = forms.FloatField(label='Age: To express the age', min_value=0,
                           max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))


class HeartDiseaseForm(forms.Form):

    age = forms.FloatField(label='Age', min_value=0, max_value=150,
                           widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # sex = forms.FloatField(label='Sex', min_value=0, max_value=1000,
    #    widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHOICES = [('1', 'M'), ('0', 'F')]
    sex = forms.CharField(
        label='Sex', widget=forms.RadioSelect(choices=CHOICES))
    # cp = forms.FloatField(label='chest pain type (4 values)', min_value=0,
    #                       max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHOICES1 = [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')]
    cp = forms.CharField(
        label='chest pain type', widget=forms.RadioSelect(choices=CHOICES1))
    trestbps = forms.FloatField(label='resting blood pressure', min_value=0,
                                max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    chol = forms.FloatField(label='serum cholestoral in mg/dl ', min_value=0,
                            max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    fbs = forms.FloatField(label='fasting blood sugar > 120 mg/dl', min_value=120,
                           max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # restecg = forms.FloatField(label='resting electrocardiographic results (values 0,1,2)',
    #                            min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHOICES2 = [('0', '0'), ('1', '1'), ('2', '2')]
    restecg = forms.CharField(
        label='resting electrocardiographic results (values 0,1,2)', widget=forms.RadioSelect(choices=CHOICES2))
    thalach = forms.FloatField(label='maximum heart rate achieved ', min_value=0,
                               max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    exang = forms.FloatField(label='exercise induced angina', min_value=0,
                             max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    oldpeak = forms.FloatField(label='ST depression induced by exercise relative to rest ',
                               min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    slope = forms.FloatField(label='the slope of the peak exercise ST segment ', min_value=0,
                             max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    # ca = forms.FloatField(label='number of major vessels (0-3) colored by flourosopy',
    #                       min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHOICES3 = [('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')]
    ca = forms.CharField(
        label='number of major vessels (0-3) colored by flourosopy', widget=forms.RadioSelect(choices=CHOICES3))
    # thal = forms.FloatField(label='thal: 0 = normal; 1 = fixed defect; 2 = reversable defect',
    #                         min_value=0, max_value=1000, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    CHOICES4 = [('0', '0'), ('1', '1'), ('2', '2')]
    thal = forms.CharField(
        label='thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', widget=forms.RadioSelect(choices=CHOICES4))
