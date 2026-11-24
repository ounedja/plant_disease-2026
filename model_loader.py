"""TFLite model loading and single-image inference."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from database import CLASSES
from preprocessing import INPUT_SIZE


@dataclass(frozen=True)
class Prediction:
    class_index: int
    confidence: float
    is_confident: bool


def _create_interpreter(model_path: Path):
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter(model_path=str(model_path))
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=str(model_path))
    except Exception as error:
        raise RuntimeError(
            "No TFLite interpreter is installed. Install tflite-runtime, ai-edge-litert, or tensorflow."
        ) from error

class ModelLoader:
    """Load a TFLite model and return its top class and confidence."""

    def __init__(
        self,
        model_path: str | Path = "plant_disease.tflite",
        *,
        confidence_threshold: float = 0.6,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.confidence_threshold = confidence_threshold
        self.interpreter = _create_interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        input_shape = tuple(int(dimension) for dimension in self.input_details["shape"])
        expected_shape = (1, INPUT_SIZE[0], INPUT_SIZE[1], 3)
        if input_shape != expected_shape:
            raise ValueError(
                f"Expected model input shape {expected_shape}, got {input_shape}"
            )

    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference on a preprocessed batch from ``preprocess_image``."""
        expected_shape = tuple(int(dimension) for dimension in self.input_details["shape"])
        if image.shape != expected_shape:
            raise ValueError(f"Expected preprocessed input {expected_shape}, got {image.shape}")

        input_dtype = self.input_details["dtype"]
        if np.issubdtype(input_dtype, np.integer):
            scale, zero_point = self.input_details["quantization"]
            if scale <= 0:
                raise ValueError("Quantized model input has an invalid scale")
            input_tensor = np.rint(image / scale + zero_point)
            limits = np.iinfo(input_dtype)
            input_tensor = np.clip(input_tensor, limits.min, limits.max).astype(input_dtype)
        else:
            input_tensor = image.astype(input_dtype, copy=False)

        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])
        output = np.asarray(output, dtype=np.float32).reshape(-1)

        if np.issubdtype(self.output_details["dtype"], np.integer):
            scale, zero_point = self.output_details["quantization"]
            if scale > 0:
                output = (output - zero_point) * scale

        if output.size != len(CLASSES):
            raise ValueError(
                f"Expected {len(CLASSES)} model scores, got {output.size}"
            )
        if not np.isfinite(output).all():
            raise ValueError("Model returned non-finite scores")

        if np.all((output >= 0.0) & (output <= 1.0)):
            scores = output
        else:
            shifted = output - np.max(output)
            exponentials = np.exp(shifted)
            scores = exponentials / exponentials.sum()

        class_index = int(np.argmax(scores))
        confidence = float(scores[class_index])
        return Prediction(
            class_index=class_index,
            confidence=confidence,
            is_confident=confidence >= self.confidence_threshold,
        )
