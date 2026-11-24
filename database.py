"""Class labels and care guidance for the plant disease model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantClass:
    plant: str
    label: str
    healthy: bool
    advice: str
    model_label: str


CLASSES: tuple[PlantClass, ...] = (
    PlantClass(
        "Potato",
        "Early blight",
        False,
        "Remove badly affected leaves, avoid overhead watering, rotate crops, "
        "and consult local guidance about registered fungicides.",
        "Potato__Early_blight",
    ),
    PlantClass(
        "Potato",
        "Healthy",
        True,
        "Continue regular monitoring, water at the soil level, and maintain "
        "good airflow around plants.",
        "Potato__healthy",
    ),
    PlantClass(
        "Potato",
        "Late blight",
        False,
        "Remove infected plants and debris promptly, keep foliage dry, and "
        "seek local agricultural advice because late blight can spread quickly.",
        "Potato__Late_blight",
    ),
    PlantClass(
        "Tomato",
        "Early blight",
        False,
        "Remove affected lower leaves, mulch to reduce soil splash, water at "
        "the base, and rotate crops.",
        "Tomato__Early_blight",
    ),
    PlantClass(
        "Tomato",
        "Healthy",
        True,
        "Continue regular monitoring, water at the soil level, and maintain "
        "good airflow around plants.",
        "Tomato__healthy",
    ),
    PlantClass(
        "Tomato",
        "Late blight",
        False,
        "Remove infected plants and debris promptly, avoid wet foliage, and "
        "seek local agricultural advice about appropriate control.",
        "Tomato__Late_blight",
    ),
    PlantClass(
        "Tomato",
        "Tomato mosaic virus",
        False,
        "Remove infected plants, disinfect tools, control weeds and insect "
        "vectors, and avoid handling healthy plants after infected ones.",
        "Tomato__Tomato_mosaic_virus",
    ),
    PlantClass(
        "Tomato",
        "Tomato yellow leaf curl virus",
        False,
        "Remove infected plants, control whiteflies, keep the growing area "
        "free of weeds, and use locally recommended resistant varieties.",
        "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    ),
)


def get_class(class_index: int) -> PlantClass:
    """Return class information for a model output index."""
    if not 0 <= class_index < len(CLASSES):
        raise ValueError(f"class_index must be between 0 and {len(CLASSES) - 1}")
    return CLASSES[class_index]
