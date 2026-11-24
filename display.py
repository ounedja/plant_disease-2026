"""Human-readable prediction output."""

from database import PlantClass, get_class
from model_loader import Prediction


def format_prediction(prediction: Prediction, plant_class: PlantClass | None = None) -> str:
    """Format a prediction with its plant, status, confidence, and advice."""
    if plant_class is None:
        plant_class = get_class(prediction.class_index)

    status = "Healthy" if plant_class.healthy else f"Disease: {plant_class.label}"
    confidence_status = "" if prediction.is_confident else " (low confidence)"
    return (
        f"Plant: {plant_class.plant}\n"
        f"Status: {status}\n"
        f"Confidence: {prediction.confidence:.1%}{confidence_status}\n"
        f"Advice: {plant_class.advice}"
    )


def display_prediction(prediction: Prediction) -> None:
    """Print a prediction to the console."""
    print(format_prediction(prediction))
