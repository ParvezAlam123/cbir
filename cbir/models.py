from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver


# Create your models here.
class Image(models.Model):
    file = models.ImageField(upload_to='images/', null=True)
    bgrHist = models.BinaryField(blank=True, null=True)
    hsvHist = models.BinaryField(blank=True, null=True)
    texture = models.BinaryField(blank=True, null=True)
    lbpHist = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
    updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)

    def __str__(self):
        return str(self.file)


# delete associated image file when image model is removed
@receiver(post_delete, sender=Image)
def image_post_delete_handler(sender, **kwargs):
    image = kwargs['instance']
    storage, path = image.file.storage, image.file.path
    storage.delete(path)


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
