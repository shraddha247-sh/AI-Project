# %%
# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(ggplot2)
library(gridExtra)

# Create charts directory if it doesn't exist
dir.create("charts", showWarnings = FALSE)

# %%
# Step 1: Data Collection and Initial Analysis
df <- read.csv("Social Meida Dataset.csv")
print("=== Initial Dataset Info ===")
str(df)
summary(df)

# %%
# Exploratory Data Analysis (EDA)
print("=== Basic Statistics ===")
summary(df)

# %%
# Check for missing values
print("=== Missing Values Analysis ===")
missing_data <- colSums(is.na(df))
missing_percentage <- (missing_data / nrow(df)) * 100
missing_info <- data.frame(
  Missing_Values = missing_data,
  Percentage = missing_percentage
)
print(missing_info)

# %%
# Distribution of numerical variables
numerical_cols <- c("Age", "Income..USD.", "Social.Media.Usage..Hours.Day.")
plots <- list()

for (col in numerical_cols) {
  p <- ggplot(df, aes(x = !!sym(col))) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_density(aes(y = after_stat(count) * 2), color = "red") +
    labs(title = paste("Distribution of", col))
  plots[[col]] <- p
}

# Save the combined plot
png("charts/numerical_distributions.png", width = 1200, height = 400)
grid.arrange(grobs = plots, ncol = 3)
dev.off()

# %%
# Box plots for numerical variables by gender
plots <- list()
for (col in numerical_cols) {
  p <- ggplot(df, aes(x = Gender, y = !!sym(col))) +
    geom_boxplot(fill = "steelblue", alpha = 0.7) +
    labs(title = paste(col, "by Gender"))
  plots[[col]] <- p
}

# Save the combined plot
png("charts/numerical_by_gender.png", width = 1200, height = 400)
grid.arrange(grobs = plots, ncol = 3)
dev.off()

# %%
# Categorical variables analysis
categorical_cols <- c("Gender", "Education.Level", "Influence.Level", "City", "Product.Category")
plots <- list()

for (col in categorical_cols) {
  if (col == "City") {
    # For cities, show top 10
    top_values <- df %>%
      count(!!sym(col)) %>%
      top_n(10, n) %>%
      pull(!!sym(col))
    
    data <- df %>%
      filter(!!sym(col) %in% top_values)
  } else {
    data <- df
  }
  
  p <- ggplot(data, aes(x = !!sym(col))) +
    geom_bar(fill = "steelblue", alpha = 0.7) +
    labs(title = paste("Distribution of", col)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  plots[[col]] <- p
}

# Save the combined plot
png("charts/categorical_distributions.png", width = 1200, height = 800)
grid.arrange(grobs = plots, ncol = 2)
dev.off()

# %%
# Correlation analysis
correlation_matrix <- cor(df[, numerical_cols], use = "complete.obs")
png("charts/correlation_matrix.png", width = 800, height = 800)
corrplot(correlation_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", tl.srt = 45)
dev.off()

# %%
# Income analysis by different categories
plots <- list()
for (col in categorical_cols) {
  if (col == "City") {
    # For cities, show top 10
    top_values <- df %>%
      count(!!sym(col)) %>%
      top_n(10, n) %>%
      pull(!!sym(col))
    
    data <- df %>%
      filter(!!sym(col) %in% top_values)
  } else {
    data <- df
  }
  
  p <- ggplot(data, aes(x = !!sym(col), y = Income..USD.)) +
    geom_boxplot(fill = "steelblue", alpha = 0.7) +
    labs(title = paste("Income Distribution by", col)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  plots[[col]] <- p
}

# Save the combined plot
png("charts/income_by_categories.png", width = 1200, height = 800)
grid.arrange(grobs = plots, ncol = 2)
dev.off()

# %%
# Social media usage patterns
platforms <- c("Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok", "Snapchat", "Pinterest")
platform_usage <- data.frame(matrix(nrow = nrow(df), ncol = length(platforms)))
colnames(platform_usage) <- platforms

for (platform in platforms) {
  platform_usage[[platform]] <- grepl(platform, df$Social.Media.Platforms, ignore.case = TRUE)
}

# Platform usage by gender
platform_usage_by_gender <- platform_usage %>%
  mutate(Gender = df$Gender) %>%
  group_by(Gender) %>%
  summarise(across(everything(), mean))

# Reshape data for plotting
platform_usage_long <- platform_usage_by_gender %>%
  pivot_longer(-Gender, names_to = "Platform", values_to = "Usage")

# Create and save the plot
png("charts/platform_usage_by_gender.png", width = 800, height = 600)
ggplot(platform_usage_long, aes(x = Gender, y = Usage, fill = Platform)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Platform Usage by Gender",
       x = "Gender",
       y = "Usage Percentage") +
  theme(legend.position = "right")
dev.off()

# %%
# Income vs Social Media Usage
png("charts/income_vs_usage.png", width = 800, height = 600)
ggplot(df, aes(x = Social.Media.Usage..Hours.Day., y = Income..USD., color = Gender)) +
  geom_point(alpha = 0.6) +
  labs(title = "Income vs Social Media Usage",
       x = "Social Media Usage (Hours/Day)",
       y = "Income (USD)")
dev.off()

# %%
# Age vs Income by Education Level
png("charts/age_vs_income.png", width = 800, height = 600)
ggplot(df, aes(x = Age, y = Income..USD., color = Education.Level, shape = Gender)) +
  geom_point(alpha = 0.6) +
  labs(title = "Age vs Income by Education Level",
       x = "Age",
       y = "Income (USD)")
dev.off()

# %%
# Key Insights
print("=== Key Insights ===")
print(paste("Average Age:", mean(df$Age, na.rm = TRUE), "years"))
print(paste("Average Income: $", mean(df$Income..USD., na.rm = TRUE)))
print(paste("Average Social Media Usage:", mean(df$Social.Media.Usage..Hours.Day., na.rm = TRUE), "hours/day"))
print(paste("Most Common Education Level:", names(which.max(table(df$Education.Level)))))
print(paste("Most Common Product Category:", names(which.max(table(df$Product.Category)))))
print(paste("Most Common City:", names(which.max(table(df$City)))))

# %%
# Step 2: Data Cleaning
print("=== Missing Values ===")
print(colSums(is.na(df)))

# Remove rows with missing values
df_clean <- na.omit(df)

# %%
# Step 3: Feature Engineering
# Create platform count
df_clean$Platform_Count <- sapply(strsplit(df_clean$Social.Media.Platforms, ","), length)

# Create binary features for each platform
platform_features <- data.frame(matrix(nrow = nrow(df_clean), ncol = length(platforms)))
colnames(platform_features) <- paste0("Uses_", platforms)

for (platform in platforms) {
  platform_features[[paste0("Uses_", platform)]] <- as.integer(grepl(platform, df_clean$Social.Media.Platforms, ignore.case = TRUE))
}

# Combine with main dataframe
df_clean <- cbind(df_clean, platform_features)

# Create age groups
df_clean$Age_Group <- cut(df_clean$Age, 
                         breaks = c(0, 25, 35, 45, 60, Inf),
                         labels = c("18-25", "26-35", "36-45", "46-60", "60+"))

# Create usage intensity
df_clean$Usage_Intensity <- cut(df_clean$Social.Media.Usage..Hours.Day.,
                              breaks = c(0, 1, 2, 3, 4, Inf),
                              labels = c("Very Low", "Low", "Medium", "High", "Very High"))

# Create additional features
df_clean$Uses_Multiple_Platforms <- as.integer(df_clean$Platform_Count > 2)
df_clean$Uses_Instagram_Facebook <- as.integer(df_clean$Uses_Instagram & df_clean$Uses_Facebook)
df_clean$Professional_Platforms <- pmin(df_clean$Uses_LinkedIn + df_clean$Uses_Twitter, 1)
df_clean$Personal_Platforms <- pmin(df_clean$Uses_Instagram + df_clean$Uses_Facebook + df_clean$Uses_Snapchat, 1)

# %%
# Step 4: Data Preprocessing
# Encode categorical variables
categorical_columns <- c("Gender", "Education.Level", "Influence.Level", "City", "Product.Category", "Age_Group", "Usage_Intensity")
df_clean_encoded <- df_clean

# Create factor levels for each categorical column
factor_levels <- list()
for (col in categorical_columns) {
  factor_levels[[col]] <- levels(factor(df_clean[[col]]))
}

# Encode categorical variables
for (col in categorical_columns) {
  df_clean_encoded[[paste0(col, "_encoded")]] <- as.integer(factor(df_clean[[col]], levels = factor_levels[[col]]))
}

# Select features for the model (excluding zero-variance features)
features <- c("Age", paste0(categorical_columns, "_encoded"), 
              "Social.Media.Usage..Hours.Day.",
              "Uses_Instagram_Facebook",
              "Professional_Platforms", "Personal_Platforms",
              paste0("Uses_", platforms))

X <- df_clean_encoded[, features]
y <- df_clean_encoded$Income..USD.

# Split the data
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Scale the features
preprocess_params <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preprocess_params, X_train)
X_test_scaled <- predict(preprocess_params, X_test)

# %%
# Step 5: Model Training
# Initialize the model
rf_regressor <- randomForest(x = X_train_scaled, y = y_train, ntree = 100, importance = TRUE)

# %%
# Step 6: Model Evaluation
# Make predictions
y_pred <- predict(rf_regressor, X_test_scaled)

# Print evaluation metrics
print("=== Model Evaluation ===")
print(paste("R2 Score:", R2(y_pred, y_test)))
print(paste("Mean Absolute Error: $", MAE(y_pred, y_test)))
print(paste("Root Mean Squared Error: $", RMSE(y_pred, y_test)))

# Plot actual vs predicted values
png("charts/income_prediction.png", width = 800, height = 600)
plot(y_test, y_pred, 
     xlab = "Actual Income (USD)", 
     ylab = "Predicted Income (USD)",
     main = "Actual vs Predicted Income")
abline(0, 1, col = "red", lty = 2)
dev.off()

# %%
# Step 7: Feature Importance Analysis
# Get feature importances
importance_df <- data.frame(
  Feature = features,
  Importance = importance(rf_regressor)[, 1]
) %>%
  arrange(desc(Importance))

# Plot feature importance
png("charts/feature_importance.png", width = 800, height = 600)
ggplot(importance_df, aes(x = Importance, y = reorder(Feature, Importance))) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  labs(title = "Feature Importance for Income Prediction",
       x = "Importance",
       y = "Feature")
dev.off()

# %%
# Save the model, preprocessing objects, and factor levels
saveRDS(rf_regressor, "income_regressor_model.rds")
saveRDS(preprocess_params, "preprocess_params.rds")
saveRDS(factor_levels, "factor_levels.rds")

# %%
# Example of making predictions on new data
predict_income <- function(new_data) {
  # Load the saved model, preprocessing objects, and factor levels
  model <- readRDS("income_regressor_model.rds")
  preprocess_params <- readRDS("preprocess_params.rds")
  factor_levels <- readRDS("factor_levels.rds")
  
  # Create a copy of the input data
  new_data <- new_data
  
  # Create platform count
  new_data$Platform_Count <- sapply(strsplit(new_data$Social.Media.Platforms, ","), length)
  
  # Create platform features
  platform_features <- data.frame(matrix(nrow = nrow(new_data), ncol = length(platforms)))
  colnames(platform_features) <- paste0("Uses_", platforms)
  
  for (platform in platforms) {
    platform_features[[paste0("Uses_", platform)]] <- as.integer(grepl(platform, new_data$Social.Media.Platforms, ignore.case = TRUE))
  }
  
  # Combine with main dataframe
  new_data <- cbind(new_data, platform_features)
  
  # Create additional features
  new_data$Uses_Multiple_Platforms <- as.integer(new_data$Platform_Count > 2)
  new_data$Uses_Instagram_Facebook <- as.integer(new_data$Uses_Instagram & new_data$Uses_Facebook)
  new_data$Professional_Platforms <- pmin(new_data$Uses_LinkedIn + new_data$Uses_Twitter, 1)
  new_data$Personal_Platforms <- pmin(new_data$Uses_Instagram + new_data$Uses_Facebook + new_data$Uses_Snapchat, 1)
  
  # Create age groups and usage intensity
  new_data$Age_Group <- cut(new_data$Age, 
                           breaks = c(0, 25, 35, 45, 60, Inf),
                           labels = c("18-25", "26-35", "36-45", "46-60", "60+"))
  
  new_data$Usage_Intensity <- cut(new_data$Social.Media.Usage..Hours.Day.,
                                breaks = c(0, 1, 2, 3, 4, Inf),
                                labels = c("Very Low", "Low", "Medium", "High", "Very High"))
  
  # Encode categorical variables using saved factor levels
  for (col in categorical_columns) {
    new_data[[paste0(col, "_encoded")]] <- as.integer(factor(new_data[[col]], levels = factor_levels[[col]]))
  }
  
  # Select features
  X_new <- new_data[, features]
  
  # Scale features
  X_new_scaled <- predict(preprocess_params, X_new)
  
  # Make prediction
  prediction <- predict(model, X_new_scaled)
  
  return(prediction)
}

# %%
# Example usage:
new_data <- data.frame(
  Age = 30,
  Gender = "Male",
  Education.Level = "Bachelor's",
  Social.Media.Usage..Hours.Day. = 2.5,
  Social.Media.Platforms = "Instagram, Facebook, Twitter",
  Influence.Level = "Somewhat Influential",
  City = "New York",
  Product.Category = "Electronics"
)

# %%
# Make prediction
prediction <- predict_income(new_data)
print(paste("Predicted Income: $", round(prediction, 2))) 





