import os
import sys
import threading

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
from django.utils import timezone
from rest_framework import generics
from .models import PipelineRun
from .serializers import PipelineRunSerializer

# Add src/ to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from pipeline.run_pipeline import run_pipeline

def execute_pipeline(run_id):
    """Background task to run the ML pipeline."""
    try:
        run_instance = PipelineRun.objects.get(id=run_id)
        run_instance.status = 'RUNNING'
        run_instance.save()
        
        output_dir = PROJECT_ROOT / 'data' / 'runs' / str(run_instance.id)
        model_path = PROJECT_ROOT / 'models' / 'production' / 'best_model_SegFormer_v2.pth'
        config_path = SRC_DIR / 'config' / 'config.yaml'
        
        run_instance.output_dir = str(output_dir)
        run_instance.save()
        
        bbox = tuple(float(x) for x in run_instance.bbox.split(','))
        
        def update_stage(stage_name):
            run_instance.current_stage = stage_name
            run_instance.save(update_fields=['current_stage'])
            
        summary = run_pipeline(
            bbox=bbox,
            target_date=str(run_instance.target_date),
            output_dir=output_dir,
            model_path=model_path,
            cloud_cover=run_instance.cloud_cover,
            backtrack_days=run_instance.backtrack_days,
            max_clusters=run_instance.max_clusters,
            filter_by_bbox=not run_instance.process_all_patches,
            config_path=str(config_path),
            progress_callback=update_stage
        )
        
        has_failures = len(summary.get('stages_failed', [])) > 0
        run_instance.status = 'FAILED' if has_failures else 'COMPLETED'
        run_instance.summary = summary
        run_instance.completed_at = timezone.now()
        run_instance.save()
        
    except Exception as e:
        run_instance.status = 'FAILED'
        run_instance.error_message = str(e)
        run_instance.completed_at = timezone.now()
        run_instance.save()

class PipelineRunListCreateView(generics.ListCreateAPIView):
    queryset = PipelineRun.objects.all()
    serializer_class = PipelineRunSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        thread = threading.Thread(target=execute_pipeline, args=(instance.id,))
        thread.daemon = True
        thread.start()

class PipelineRunDetailView(generics.RetrieveAPIView):
    queryset = PipelineRun.objects.all()
    serializer_class = PipelineRunSerializer

from rest_framework.views import APIView
from rest_framework.response import Response

def execute_single_cluster_backtrack(run_id, cluster_id):
    try:
        run_instance = PipelineRun.objects.get(id=run_id)
        output_dir = PROJECT_ROOT / 'data' / 'runs' / str(run_instance.id)
        config_path = SRC_DIR / 'config' / 'config.yaml'
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        if 'backtracking' not in config:
            config['backtracking'] = {}
        config['backtracking']['days'] = run_instance.backtrack_days
            
        scene_id = None
        if run_instance.summary and 'outputs' in run_instance.summary:
            scenes = run_instance.summary['outputs'].get('raw_scenes', [])
            if scenes:
                scene_id = Path(scenes[0]).name
                
        if not scene_id:
            scene_dirs = list((output_dir / 'reports').glob('*'))
            if scene_dirs:
                scene_id = scene_dirs[0].name
                
        import importlib
        stage_05 = importlib.import_module("pipeline.05_backtrack")
        stage_06 = importlib.import_module("pipeline.06_attribute")
        
        sources = stage_05.run_single_cluster(
            scene_id=scene_id,
            cluster_id=cluster_id,
            output_dir=output_dir / 'attribution',
            config=config
        )
        
        stage_06.run(
            scene_id=scene_id,
            sources=sources,
            detections_path=str(output_dir / 'reports' / scene_id / 'debris_summary.csv'),
            output_dir=output_dir / 'attribution',
            config=config,
            detection_date=str(run_instance.target_date)
        )
    except Exception as e:
        print(f"Single cluster backtrack failed: {e}")

class BacktrackClusterView(APIView):
    def post(self, request, pk):
        cluster_id = request.data.get('cluster_id')
        if cluster_id is None:
            return Response({"error": "cluster_id required"}, status=400)
            
        thread = threading.Thread(target=execute_single_cluster_backtrack, args=(pk, int(cluster_id)))
        thread.daemon = True
        thread.start()
        
        return Response({"status": f"Started backtracking cluster {cluster_id}"})
