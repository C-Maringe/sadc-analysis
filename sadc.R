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

# Summary of fixed effects model
summary(fixed_effects_model)

#testing for heteroscedasticity
bptest(fixed_effects_model)

#testing for random effects vs fixed effects
phtest(fixed_effects_model,random_effects_model)

set.seed(123)
train_indices <- createDataPartition(panel_data$GDP, p = 0.7, list = FALSE)
train_data <- panel_data[train_indices, ]
test_data <- panel_data[-train_indices, ]

rf_model <- randomForest(GDP ~ ., data = train_data, ntree = 100)

rf_predictions <- predict(rf_model, newdata = test_data)

rf_rmse <- sqrt(mean((rf_predictions - test_data$GDP)^2))
rf_r2 <- cor(rf_predictions, test_data$GDP)^2