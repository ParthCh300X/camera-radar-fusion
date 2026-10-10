
import torch
import cv2
import numpy
import pandas
import matplotlib
import ultralytics

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Ultralytics:", ultralytics.__version__)

assert torch.cuda.is_available(), "CUDA is unavailable"
assert ultralytics.__version__ == "8.3.0", "Unexpected YOLO package version"

print("SUCCESS: Core packages imported.")
