setwd("C:\\Users\\Dell\\Desktop\\Analysis\\sadc")

library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)

library(plm)
library(lmtest)

library(randomForest)
library(xgboost)
library(neuralnet)
library(e1071)

data <- read_xlsx("SADC E Data.xlsx")

panel_data <- pdata.frame(data, index = c("Country", "Year"))

# print(colnames(panel_data))

summary(panel_data)

colSums(is.na(panel_data))

correlation_matrix <- cor(panel_data[, c("GDP", "Inflation", "Imports", "Exports", "EDebt", "ERate", "IReserves", "LForce", "Unemployment", "FDI", "IRate")])
corrplot(correlation_matrix, method = "color")

#simple lr
lr = lm(GDP ~ Inflation + Imports + Exports + EDebt + ERate + IReserves + LForce + Unemployment + FDI + IRate + factor(Year) + factor(Country),
        data = data)
summary(lr)

#simple plm
#plm_lr = plm(GDP ~ Inflation + Imports + Exports + EDebt + ERate + IReserves + LForce + Unemployment + FDI + IRate + factor(Year),
             #index = "Country",
             #model = "within",
             #data = data)
#summary(plm_lr)

fixed_effects_model <- plm(GDP ~ Inflation + Imports + Exports + EDebt + ERate + IReserves + LForce + Unemployment + FDI + IRate,
                           data = panel_data,
                           model = "within")

random_effects_model <- plm(GDP ~ Inflation + Imports + Exports + EDebt + ERate + IReserves + LForce + Unemployment + FDI + IRate,
                            data = panel_data,
                            model = "random")

summary(fixed_effects_model)

bptest(fixed_effects_model)

phtest(fixed_effects_model,random_effects_model)

######################################################################

## RANDOM FOREST
set.seed(123)
train_indices <- createDataPartition(panel_data$GDP, p = 0.7, list = FALSE)
train_data <- panel_data[train_indices, ]
test_data <- panel_data[-train_indices, ]

rf_model <- randomForest(GDP ~ ., data = train_data, ntree = 100)

rf_predictions <- predict(rf_model, newdata = test_data)

rf_rmse <- sqrt(mean((rf_predictions - test_data$GDP)^2))
rf_r2 <- cor(rf_predictions, test_data$GDP)^2

######################################################################

# XGBoost model

numeric_vars <- c("GDP", "Inflation", "Imports", "Exports", "EDebt", "ERate", "IReserves", "LForce", "Unemployment", "FDI", "IRate")
train_data_numeric <- train_data[, numeric_vars]

train_matrix <- xgb.DMatrix(as.matrix(train_data_numeric[, -1]), label = train_data_numeric$GDP)
test_matrix <- xgb.DMatrix(as.matrix(test_data[, numeric_vars[-1]]), label = test_data$GDP)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

watchlist <- list(train = train_matrix, test = test_matrix)

xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  early_stopping_rounds = 10,
  watchlist = watchlist,
  verbose = 0
)

xgb_predictions <- predict(xgb_model, test_matrix)

xgb_rmse <- sqrt(mean((xgb_predictions - test_data$GDP)^2))
print(paste("RMSE:", xgb_rmse))

actual <- test_data$GDP
xgb_rsquared <- 1 - sum((actual - xgb_predictions)^2) / sum((actual - mean(actual))^2)
xgb_accuracy <- xgb_rsquared * 100

print(paste("Accuracy (R-squared):", xgb_accuracy, "%"))


##################################################################

## RANDOM FOREST
numeric_vars <- c("GDP", "Inflation", "Imports", "Exports", "EDebt", "ERate", "IReserves", "LForce", "Unemployment", "FDI", "IRate")
train_data_numeric <- train_data[, numeric_vars]

train_data_normalized <- as.data.frame(lapply(train_data_numeric, function(x) (x - min(x)) / (max(x) - min(x))))

formula <- as.formula(paste("GDP ~", paste(numeric_vars[-1], collapse = " + ")))

neural_model <- neuralnet(
  formula,
  data = train_data_normalized,
  hidden = c(5, 3),
  linear.output = TRUE
)

test_data_normalized <- as.data.frame(lapply(test_data[, numeric_vars], function(x) (x - min(x)) / (max(x) - min(x))))
neural_model_predictions <- predict(neural_model, test_data_normalized)

neural_model_rsquared <- 1 - sum((actual - neural_model_predictions)^2) / sum((actual - mean(actual))^2)
neural_model_accuracy <- neural_model_rsquared * 100

print(paste("Accuracy (R-squared):", neural_model_accuracy, "%"))


########################################################################

# SVR model
numeric_vars <- c("GDP", "Inflation", "Imports", "Exports", "EDebt", "ERate", "IReserves", "LForce", "Unemployment", "FDI", "IRate")
train_data_numeric <- train_data[, numeric_vars]

svr_model <- svm(GDP ~ ., data = train_data_numeric, kernel = "radial")

svr_predictions <- predict(svr_model, newdata = test_data[, numeric_vars])

svr_rsquared <- 1 - sum((actual - svr_predictions)^2) / sum((actual - mean(actual))^2)
svr_accuracy <- svr_rsquared * 100

print(paste("Accuracy (R-squared):", svr_accuracy, "%"))

