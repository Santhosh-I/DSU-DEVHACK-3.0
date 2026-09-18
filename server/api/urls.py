from django.urls import path
from .views import PipelineRunListCreateView, PipelineRunDetailView, BacktrackClusterView

urlpatterns = [
    path('pipeline/runs/', PipelineRunListCreateView.as_view(), name='pipeline-run-list-create'),
    path('pipeline/runs/<uuid:pk>/', PipelineRunDetailView.as_view(), name='pipeline-run-detail'),
    path('pipeline/runs/<uuid:pk>/backtrack-cluster/', BacktrackClusterView.as_view(), name='backtrack-cluster'),
]
