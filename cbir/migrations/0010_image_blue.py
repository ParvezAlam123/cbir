# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-02-04 22:54
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cbir', '0009_auto_20160203_1718'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='blue',
            field=models.TextField(blank=True, null=True),
        ),
    ]