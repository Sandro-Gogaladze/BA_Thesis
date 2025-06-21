# Makefile for Income Estimation project
# This Makefile simplifies running the entire workflow for each model type

# Define root directory paths
ROOT_DIR := .
INCOME_ESTIMATION_DIR := $(ROOT_DIR)/IncomeEstimation

# Define Python command
PYTHON := python

# Define model types
MODELS := xgboost segment_aware huber_threshold quantile

# Default target
.PHONY: help
help:
	@echo "Income Estimation Makefile"
	@echo "--------------------------"
	@echo "Available targets:"
	@echo "  make all                  - Run complete pipeline for ALL models (preprocessing + XGBoost + Quantile + both Pre-hoc models)"
	@echo "  make preprocessing        - Run only data preprocessing"
	@echo "  make baseline/xgboost     - Run complete XGBoost pipeline (cleanup, train, inference, evaluate)"
	@echo "  make prehoc/segment_aware - Run complete Segment-Aware pipeline"
	@echo "  make prehoc/huber_threshold - Run complete Huber Threshold pipeline"
	@echo "  make posthoc/quantile     - Run complete Quantile pipeline"


# Preprocessing data
.PHONY: preprocessing
preprocessing:
	@echo "Running data preprocessing..."
	$(PYTHON) -m IncomeEstimation.src.preprocessing.preprocessor
	@echo "Data preprocessing completed successfully."

# Define rules for each model type
.PHONY: baseline/xgboost
baseline/xgboost:
	@echo "Running complete XGBoost pipeline..."
	$(PYTHON) -m IncomeEstimation.cleanup --model xgboost --confirm
	$(PYTHON) -m IncomeEstimation.src.train.train --model xgboost
	$(PYTHON) -m IncomeEstimation.src.inference.predict --model xgboost
	$(PYTHON) -m IncomeEstimation.src.evaluation.evaluate --model xgboost
	@echo "XGBoost pipeline completed successfully."

.PHONY: posthoc/quantile
posthoc/quantile:
	@echo "Running complete Quantile pipeline..."
	$(PYTHON) -m IncomeEstimation.cleanup --model quantile --confirm
	$(PYTHON) -m IncomeEstimation.src.train.train --model quantile
	$(PYTHON) -m IncomeEstimation.src.inference.predict --model quantile
	$(PYTHON) -m IncomeEstimation.src.evaluation.evaluate --model quantile
	@echo "Quantile pipeline completed successfully."

.PHONY: prehoc/huber_threshold
prehoc/huber_threshold:
	@echo "Running complete Huber Threshold pipeline..."
	$(PYTHON) -m IncomeEstimation.cleanup --model huber_threshold --confirm
	$(PYTHON) -m IncomeEstimation.src.train.train --model huber_threshold
	$(PYTHON) -m IncomeEstimation.src.inference.predict --model huber_threshold
	$(PYTHON) -m IncomeEstimation.src.evaluation.evaluate --model huber_threshold
	@echo "Huber Threshold pipeline completed successfully."

.PHONY: prehoc/segment_aware
prehoc/segment_aware:
	@echo "Running complete Segment-Aware pipeline..."
	$(PYTHON) -m IncomeEstimation.cleanup --model segment_aware --confirm
	$(PYTHON) -m IncomeEstimation.src.train.train --model segment_aware
	$(PYTHON) -m IncomeEstimation.src.inference.predict --model segment_aware
	$(PYTHON) -m IncomeEstimation.src.evaluation.evaluate --model segment_aware
	@echo "Segment-Aware pipeline completed successfully."

# Run complete pipeline for all models
.PHONY: all
all:  preprocessing baseline/xgboost posthoc/quantile prehoc/huber_threshold prehoc/segment_aware
	@echo "========================================"
	@echo "COMPLETE PIPELINE FINISHED SUCCESSFULLY"
	@echo "========================================"
	@echo "All models have been trained, evaluated, and visualized:"
	@echo "  ✓ XGBoost baseline model"
	@echo "  ✓ Post-hoc Quantile model" 
	@echo "  ✓ Pre-hoc Huber Threshold model"
	@echo "  ✓ Pre-hoc Segment-Aware model"
	@echo ""
	@echo "Results are available in:"
	@echo "  - IncomeEstimation/baseline/xgboost/results/"
	@echo "  - IncomeEstimation/posthoc/quantile/results/"
	@echo "  - IncomeEstimation/prehoc/huber_threshold/results/"
	@echo "  - IncomeEstimation/prehoc/segment_aware/results/"
	@echo "========================================"
