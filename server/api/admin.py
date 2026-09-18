from django.contrib import admin

from .models import PipelineRun


@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
	list_display = (
		'id',
		'status',
		'target_date',
		'cloud_cover',
		'backtrack_days',
		'max_clusters',
		'process_all_patches',
		'created_at',
		'completed_at',
	)
	list_filter = ('status', 'target_date', 'created_at')
	search_fields = ('id', 'bbox', 'output_dir', 'error_message')
	readonly_fields = (
		'id',
		'status',
		'output_dir',
		'created_at',
		'completed_at',
		'summary',
		'error_message',
	)
	ordering = ('-created_at',)
