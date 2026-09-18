from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_pipelinerun_max_clusters'),
    ]

    operations = [
        migrations.AddField(
            model_name='pipelinerun',
            name='process_all_patches',
            field=models.BooleanField(default=False),
        ),
    ]