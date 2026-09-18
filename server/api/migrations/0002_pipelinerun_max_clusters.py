from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='pipelinerun',
            name='max_clusters',
            field=models.PositiveIntegerField(default=5),
        ),
    ]