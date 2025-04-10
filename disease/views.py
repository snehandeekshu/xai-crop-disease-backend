from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PlantImage
from .serializers import PlantImageSerializer

from django.conf import settings
import os

import numpy as np
from keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os

# Load model once (optional for speed)
try:
    from tensorflow.keras.models import load_model
    model_path = os.path.join(settings.BASE_DIR, 'plant_disease_model.h5')
    model = load_model(model_path)
  # Change if needed
except ImportError:
    model = None

class PredictView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, format=None):
        serializer = PlantImageSerializer(data=request.data)
        if serializer.is_valid():
            instance = serializer.save()
            img_path = instance.image.path

            # Load image and preprocess
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds[0])
            confidence = float(np.max(preds[0]))

            # LIME explanation
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=1000
            )
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
            )

            # Save explanation image
            explanation_path = f"media/explain_{instance.id}.png"
            plt.imsave(explanation_path, mark_boundaries(temp / 255.0, mask))

            # Return response
            return Response({
                "prediction": str(predicted_class),
                "confidence": confidence,
                "explanation_image_url": request.build_absolute_uri('/') + explanation_path
            })

        return Response(serializer.errors, status=400)
