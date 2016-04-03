from django import forms
from .models import Image
import re


class ImageForm(forms.ModelForm):

    class Meta:
        model = Image
        fields = ['file']

    # # example custom validation
    # def clean_url(self):
    #     url = self.cleaned_data.get('url')
    #     if re.match('(\s+)', url):    # check for whitespace
    #         raise forms.ValidationError("The url cannot contain whitespace")
    #     return url
