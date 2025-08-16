# main/apps.py

from django.apps import AppConfig
import sys

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    def ready(self):
        # This code runs once when the server starts.
        # We check for 'runserver' to avoid running this during migrations.
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv:
            print("Initializing AI Model for the application...")
            from . import views  # Import views here
            try:
                from segment_anything import sam_model_registry, SamPredictor

                sam_checkpoint = "main/static/checkpoints/sam_vit_l_0b3195.pth"
                model_type = "vit_l"
                # Use "cuda" if you have a GPU, otherwise "cpu"
                device = "cpu"

                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)

                # Store the predictor in the views module
                views.PREDICTOR = SamPredictor(sam)
                print("AI Model loaded successfully into memory.")
            except Exception as e:
                views.PREDICTOR = None
                print(f"Error loading AI Model on startup: {e}")