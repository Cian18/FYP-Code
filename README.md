# FYP-Code
Code for my final year project

# 🏋️‍♂️ Exercise Technique Analysis

This project was developed across three successive prototypes:

- **Prototype 1**: Analyses bicep curl technique and provides basic feedback.
- **Prototype 2**: Improves the bicep curl analysis with refined classification and feedback logic.
- **Prototype 3**: Extends the approach to analyse barbell squat technique.

Each prototype involves the following core stages:

1. **Classification**
   - Detects the phase of the movement (e.g. concentric, eccentric).
2. **Technique Modelling**
   - Generates ideal joint trajectories for comparison.
3. **Feedback Generation**
   - Compares actual motion to the model and provides human-like qualitative feedback.

In **Prototype 3**, additional steps were introduced:

- **Camera Angle Detection**
  - Automatically identifies whether the video is a side or front view.
- **Angle-Specific Pipelines**
  - Classification, modelling, and feedback stages are handled separately for front and side views.

