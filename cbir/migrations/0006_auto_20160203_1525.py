# -*- coding: utf-8 -*-
# Generated by Django 1.9.2 on 2016-02-03 15:25
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cbir', '0005_auto_20160203_1340'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='blue',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='image',
            name='green',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='image',
            name='red',
            field=models.TextField(blank=True, null=True),
        ),
    ]