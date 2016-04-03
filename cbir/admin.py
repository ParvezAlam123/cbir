from django.contrib import admin

# Register your models here.
from .forms import ImageForm
from .models import Image


class ImageAdmin(admin.ModelAdmin):
    list_display = ["__str__", "created_at", "updated_at"]
    form = ImageForm
    # class Meta:
    #     model = Image

admin.site.register(Image, ImageAdmin)
