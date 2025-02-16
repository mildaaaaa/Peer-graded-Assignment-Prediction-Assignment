# Load required libraries
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(readr)
library(rmarkdown)

# Ensure necessary directories exist
if (!dir.exists("results")) {
  dir.create("results", showWarnings = FALSE)
}

# Load the training and testing datasets
train_path <- "data/pml-training.csv"
test_path <- "data/pml-testing.csv"

if (file.exists(train_path) & file.exists(test_path)) {
  training <- read_csv(train_path)
  testing <- read_csv(test_path)
} else {
  stop("Training or testing dataset not found. Please check file paths.")
}

# Data preprocessing
na_threshold <- 0.9
na_counts <- colSums(is.na(training)) / nrow(training)
training <- training[, na_counts < na_threshold]

# Ensure testing dataset has the same features as training (excluding 'classe')
common_features <- setdiff(names(training), "classe")
testing <- testing[, common_features, drop = FALSE]

# Check if testing dataset is empty
debug_testing <- nrow(testing) == 0
if (debug_testing) {
  warning("Testing dataset is empty. Predictions will be skipped.")
}

# Remove irrelevant columns
training <- training[, -c(1:7)]
training$classe <- as.factor(training$classe)

# Split training data
set.seed(123)
trainIndex <- createDataPartition(training$classe, p = 0.7, list = FALSE)
trainData <- training[trainIndex, ]
validData <- training[-trainIndex, ]

# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(classe ~ ., data = trainData, ntree = 100)

# Validate model
rf_predictions <- predict(rf_model, validData)
conf_matrix <- confusionMatrix(rf_predictions, validData$classe)
print(conf_matrix)

# Feature importance
importance <- varImp(rf_model)
plot(importance, main = "Feature Importance")

# Apply model to test dataset
if (!debug_testing) {
  final_predictions <- predict(rf_model, testing)
  print(head(final_predictions))  # Debugging output
} else {
  final_predictions <- NULL
}

# Ensure 'results' directory exists before writing predictions
if (!dir.exists("results")) {
  dir.create("results", showWarnings = FALSE)
}

# Check if 'results' folder is writable
if (file.access("results", 2) != 0) {
  stop("Error: Cannot write to 'results' directory. Please check permissions.")
}

# Save predictions only if they exist
if (!is.null(final_predictions) && length(final_predictions) > 0) {
  prediction_files <- function(predictions) {
    for (i in seq_along(predictions)) {
      filename <- paste0("results/problem_id_", i, ".txt")
      print(paste("Attempting to write file:", filename))  # Debugging output
      tryCatch(
        {
          write.table(predictions[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
          message("Successfully written: ", filename)
        },
        error = function(e) {
          message("Error writing file: ", filename, " - ", e$message)
        }
      )
    }
  }
  prediction_files(final_predictions)
} else {
  warning("final_predictions is empty. Skipping file saving.")
}

# Save model if it exists
if (exists("rf_model")) {
  saveRDS(rf_model, "rf_model.rds")
} else {
  warning("Model 'rf_model' not found. Skipping save.")
}

# Generate R Markdown report
if (file.exists("prediction_report.Rmd")) {
  render("prediction_report.Rmd", output_format = "html_document")
} else {
  warning("File 'prediction_report.Rmd' not found in", getwd(), ". Listing available files:")
  print(list.files())
}
