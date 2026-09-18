from django.db import models
import uuid

class PipelineRun(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bbox = models.CharField(max_length=255, help_text="Comma separated: lon_min,lat_min,lon_max,lat_max")
    target_date = models.DateField(help_text="Target date for the run in YYYY-MM-DD")
    cloud_cover = models.IntegerField(default=20)
    backtrack_days = models.IntegerField(default=30)
    max_clusters = models.PositiveIntegerField(default=5)
    process_all_patches = models.BooleanField(default=False)
    run_name = models.CharField(max_length=255, blank=True, default='',
                                help_text="Optional human-readable name. Auto-generated if blank.")
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    output_dir = models.CharField(max_length=500, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    
    # Store summary stats as JSON and error messages if any
    summary = models.JSONField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.run_name:
            # Auto-generate from date + bbox region
            parts = self.bbox.split(',')
            if len(parts) == 4:
                lon_min, lat_min, lon_max, lat_max = [float(p) for p in parts]
                lon_c = (lon_min + lon_max) / 2
                lat_c = (lat_min + lat_max) / 2
                self.run_name = f"{self.target_date}_{lat_c:.2f}_{lon_c:.2f}"
            else:
                self.run_name = f"{self.target_date}_run"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.run_name} ({self.status})"
    
    class Meta:
        ordering = ['-created_at']
