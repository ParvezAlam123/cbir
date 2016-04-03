from django.db import models


# Create your models here.
class Image(models.Model):
    file = models.ImageField(upload_to='images/', null=True)
    bgrHist = models.TextField(blank=True, null=True)
    hsvHist = models.TextField(blank=True, null=True)
    lbpHist = models.BinaryField(blank=True, null=True)
    dissimilarity = models.TextField(blank=True, null=True)
    correlation = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
    updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)

    def __str__(self):
        return str(self.file)


# table to hold training set for machine learning algorithms
class Example(models.Model):
    file = models.ImageField(upload_to='images/training/', null=True)
    lbpHist = models.BinaryField(blank=True, null=True)
    label = models.CharField(max_length=45, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
    updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)

    def __str__(self):
        return str(self.file)


# machine learning models
class Classifier(models.Model):
    model = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
    updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)
