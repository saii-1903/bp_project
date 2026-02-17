# BP Project Scripts

This folder contains the training and testing scripts for the BP project, extracted from `modelsky` and surrounding files.

## Included Files

-   **`[Train.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/Train.py)`**: The main training script for BP classification and regression.
-   **`[Test.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/Test.py)`**: The main testing script for BP predictions.
-   **`[AL.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/AL.py)`**: An alternative testing script with advanced stability filtering.
-   **`[hbglucosetraining.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/hbglucosetraining.py)`**: Training script for Hemoglobin and Glucose models.
-   **`[hbglutest.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/hbglutest.py)`**: Testing script for Hemoglobin and Glucose.
-   **`[dangerr.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/bp%20project/dangerr.py)`**: An alternative prediction script that **is related** to the models in `modelsky`. It uses the same model format (e.g., `classifier.pkl`, `scaler_cls.pkl`) as `Test.py`.

## Excluded Files

-   **`[danger.py](file:///c:/Users/saish/OneDrive/Attachments/Documents/PlatformIO/Project-3/danger.py)`**: This file was excluded because it is an **alternative training script** that produces a different model structure (e.g., creates `global_feature_scaler.pkl`) which is not compatible with the current `modelsky` contents.
