# Langkah 1: Import Dataset & Preprocessing
library(readr)
library(dplyr)
library(caret)

# Baca dataset
training_data <- read_csv("pml-training.csv")
testing_data <- read_csv("pml-testing.csv")

# Data cleaning: Hilangkan kolom dengan banyak nilai kosong (NA)
training_data <- training_data %>% select_if(~sum(is.na(.)) < nrow(training_data) * 0.5)
testing_data <- testing_data %>% select_if(~sum(is.na(.)) < nrow(testing_data) * 0.5)

# Hilangkan fitur yang tidak relevan
irrelevant_features <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
training_data <- training_data %>% select(-one_of(irrelevant_features))
testing_data <- testing_data %>% select(-one_of(irrelevant_features)
# Langkah 2: Exploratory Data Analysis (EDA)
library(ggplot2)

# Visualisasi distribusi variabel target
ggplot(training_data, aes(x = classe)) + geom_bar() + theme_minimal()

# Periksa korelasi antar fitur dan target
cor_matrix <- cor(training_data %>% select_if(is.numeric))
corrplot::corrplot(cor_matrix, method = "circle")

# Langkah 3: Membangun Model Pembelajaran Mesin
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)

# Model Random Forest
model_rf <- train(classe ~ ., data = training_data, method = "rf", trControl = train_control)

# Model Gradient Boosting
model_gbm <- train(classe ~ ., data = training_data, method = "gbm", trControl = train_control, verbose = FALSE)

# Langkah 4: Evaluasi Model
# Akurasi dan confusion matrix untuk Random Forest
pred_rf <- predict(model_rf, training_data)
confusionMatrix(pred_rf, training_data$classe)

# Akurasi dan confusion matrix untuk Gradient Boosting
pred_gbm <- predict(model_gbm, training_data)
confusionMatrix(pred_gbm, training_data$classe)

# Langkah 5: Prediksi 20 Kasus Uji
best_model <- ifelse(model_rf$results$Accuracy > model_gbm$results$Accuracy, model_rf, model_gbm)
predictions <- predict(best_model, testing_data)

# Simpan hasil prediksi
write.csv(predictions, "predictions.csv", row.names = FALSE)

# Langkah 6: Dokumentasi & Pengiriman
# Buat laporan dalam Markdown R (.Rmd) dan render ke HTML
library(knitr)
library(rmarkdown)

rmarkdown::render("report.Rmd", output_format = "html_document")