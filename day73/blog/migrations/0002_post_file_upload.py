# Generated by Django 5.1.3 on 2024-11-25 15:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("blog", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="post",
            name="file_upload",
            field=models.ImageField(
                blank=True, null=True, upload_to="blog/images/%Y/%m/%d"
            ),
        ),
    ]