from django.apps import AppConfig


import sys
import threading

def reset_interrupted_runs():
    try:
        from .models import PipelineRun
        PipelineRun.objects.filter(status__in=['RUNNING', 'PENDING']).update(
            status='FAILED', error_message="Pipeline execution was interrupted."
        )
    except Exception:
        pass

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    def ready(self):
        if 'runserver' in sys.argv:
            # Delay execution slightly to avoid AppRegistryNotReady warning
            threading.Timer(1.0, reset_interrupted_runs).start()
