from django.db import models

# Create your models here.
# class Userdb(models.Model):
#     email = models.CharField(max_length=100,null = True)
#     password = models.CharField(max_length=100,null = True)

#     class Meta:
#         db_table = 'login'


class BreastCancer(models.Model):
    radius = models.CharField(max_length=100)
    texture = models.CharField(max_length=100)
    perimeter = models.CharField(max_length=100)
    area = models.CharField(max_length=100)
    smoothness = models.CharField(max_length=100)




class HeartDisease(models.Model):
    age = models.CharField(max_length=100)
    sex = models.CharField(max_length=100)
    cp = models.CharField(max_length=100)
    trestbps = models.CharField(max_length=100)
    chol = models.CharField(max_length=100)
    fbs = models.CharField(max_length=100)
    restecg = models.CharField(max_length=100)
    thalach = models.CharField(max_length=100)
    exang = models.CharField(max_length=100)
    oldpeak = models.CharField(max_length=100)
    slope = models.CharField(max_length=100)
    ca = models.CharField(max_length=100)
    thal = models.CharField(max_length=100)


class Diabetes(models.Model):

    pregnancies = models.CharField(max_length=100)
    glucose = models.CharField(max_length=100)
    bloodpressure = models.CharField(max_length=100)
    skinthickness = models.CharField(max_length=100)
    insulin = models.CharField(max_length=100)
    bmi = models.CharField(max_length=100)
    pedigree = models.CharField(max_length=100)
    age = models.CharField(max_length=100)
