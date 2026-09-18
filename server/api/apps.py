from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        try:
            from .models import PipelineRun
            PipelineRun.objects.filter(status__in=['RUNNING', 'PENDING']).update(
                status='FAILED', error_message="Pipeline execution was interrupted."
            )
        except Exception:
            pass
