# TODO: Improve Fake News Detection Model Accuracy

## Improvement Plan Steps
- [x] Update requirements.txt: Add xgboost and spacy for new model and advanced features.
- [x] Edit src/feature_engineering.py: Add advanced features like POS tags and named entities using spaCy.
- [x] Edit src/train_evaluate.py: Add hyperparameter tuning using GridSearchCV, implement cross-validation, and add XGBoost model.
- [x] Edit notebooks/fake_news_detection.py: Integrate new features, tuning, and cross-validation into the pipeline.
- [x] Run notebooks/fake_news_detection.py to retrain models with improvements and evaluate accuracy.
- [x] Update TODO.md with completions and any additional tasks.

## New Task: Continue Training with Pausing Function
- [x] Edit notebooks/fake_news_detection.py: Enhance pausing function to save and resume training state (e.g., completed models, current step in pipeline).
- [x] Add pausing checks in the training loop to periodically check for pause and save progress.
- [x] Update resume logic to load checkpoint and skip completed steps or continue from interruption.
- [x] Run notebooks/fake_news_detection.py to test the enhanced pausing and training continuation.
- [x] Update TODO.md with completions.

## Task Completed
- [x] All improvements implemented: XGBoost model, hyperparameter tuning, cross-validation, advanced features (POS tags, named entities).
- [x] Training pipeline with pausing/resume functionality added.
- [x] Flask web app running on http://127.0.0.1:5000 for testing predictions.
- [x] Model artifacts saved to app/ directory for deployment.
- [x] Code errors corrected: Fixed checkpoint to save/load 'results', removed plt.show() from plot functions, improved result handling for resuming.
- [x] Pipeline verified: Runs without errors, preprocessing completed successfully.
