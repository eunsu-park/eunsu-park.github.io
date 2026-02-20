#!/usr/bin/env python3
"""
TensorFlow Lite Inference Example
TFLite 추론 예제

Performs image classification using a TFLite model.

Install:
    pip install tflite-runtime numpy pillow

Usage:
    python3 tflite_inference.py --model model.tflite --image image.jpg
    python3 tflite_inference.py --model model.tflite --image image.jpg --labels labels.txt
"""

import numpy as np
from PIL import Image
import time
import argparse
import os

# Try to import TFLite runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("Error: tflite-runtime or tensorflow not found")
        print("Install with: pip install tflite-runtime")
        exit(1)

class TFLiteClassifier:
    """TensorFlow Lite Image Classifier"""

    def __init__(self, model_path: str, labels_path: str = None):
        """Initialize classifier with model and optional labels"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        print(f"Loading model: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]['dtype']

        print(f"Input shape: {self.input_shape}")
        print(f"Input dtype: {self.input_dtype}")

        # Load labels
        self.labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.labels)} labels")

    def preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for inference"""
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.input_width, self.input_height))

        # Convert to numpy array
        input_data = np.array(image, dtype=np.float32)

        # Normalize (MobileNet style: -1 to 1)
        input_data = (input_data - 127.5) / 127.5

        # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)

        return input_data

    def classify(self, image_path: str, top_k: int = 5) -> dict:
        """Classify image and return top-k predictions"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess
        input_data = self.preprocess(image_path)

        # Inference
        start_time = time.perf_counter()

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        inference_time = (time.perf_counter() - start_time) * 1000

        # Get top-k predictions
        top_indices = output.argsort()[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            score = float(output[idx])
            predictions.append({
                "class_id": int(idx),
                "label": label,
                "score": score,
                "confidence": f"{score * 100:.1f}%"
            })

        return {
            "image": image_path,
            "predictions": predictions,
            "inference_time_ms": round(inference_time, 2),
            "model_input_size": f"{self.input_width}x{self.input_height}"
        }

    def benchmark(self, num_runs: int = 100) -> dict:
        """Benchmark inference speed"""
        print(f"\nBenchmarking ({num_runs} runs)...")

        # Create dummy input
        dummy_input = np.random.rand(*self.input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
            self.interpreter.invoke()
            _ = self.interpreter.get_tensor(self.output_details[0]['index'])
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000 / avg_time

        return {
            "runs": num_runs,
            "avg_time_ms": round(avg_time, 2),
            "std_time_ms": round(std_time, 2),
            "fps": round(fps, 1),
            "min_time_ms": round(min(times), 2),
            "max_time_ms": round(max(times), 2)
        }

def create_dummy_model():
    """Create a dummy TFLite model for testing"""
    try:
        import tensorflow as tf

        # Simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save
        with open('dummy_model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("Created dummy_model.tflite for testing")
        return 'dummy_model.tflite'

    except ImportError:
        print("TensorFlow not available. Cannot create dummy model.")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="TFLite Image Classifier")
    parser.add_argument("--model", required=True, help="Path to TFLite model")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--labels", help="Path to labels file")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    print("=== TFLite Inference ===\n")

    try:
        classifier = TFLiteClassifier(args.model, args.labels)

        if args.benchmark:
            # Run benchmark
            results = classifier.benchmark()
            print("\nBenchmark Results:")
            print(f"  Average time: {results['avg_time_ms']:.2f} ms (+/- {results['std_time_ms']:.2f})")
            print(f"  FPS: {results['fps']:.1f}")
            print(f"  Min/Max: {results['min_time_ms']:.2f} / {results['max_time_ms']:.2f} ms")

        elif args.image:
            # Classify image
            results = classifier.classify(args.image, args.top_k)

            print(f"\nImage: {results['image']}")
            print(f"Inference time: {results['inference_time_ms']} ms")
            print(f"\nTop-{args.top_k} Predictions:")

            for i, pred in enumerate(results['predictions'], 1):
                print(f"  {i}. {pred['label']}: {pred['confidence']}")

        else:
            print("Please specify --image or --benchmark")
            print("\nModel info:")
            print(f"  Input shape: {classifier.input_shape}")
            print(f"  Input dtype: {classifier.input_dtype}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
