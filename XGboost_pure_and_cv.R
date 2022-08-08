library(tidyverse)
library(ggplot2)
library(tidyr)
library(dplyr)
library(corrplot)
library(xgboost)
library(mltools)
library(data.table)
library(caTools)

## Importing data -----
setwd("")

set.seed(123)

# EDA -----
str(data)

# Plotting
xyplot(y ~ x1 | x2, data = data, layout = c(6, 3),
       xlab = "desctiption of independent variables",
       ylab = "description of target", aspect = "xy")

xyplot(target ~ x1 | x2, data, groups = Subject,
       type = "a",
       auto.key =
         list(space = "right", lines = TRUE))


# Splitting data -----
X <- subset(data, select = -c(target))
y <- data$target

smp_size <- floor(0.80 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]


# Create model -----
xgb <- xgboost(params = list(),
               data = data.matrix(X_train),
               eta = 0.1,
               max_depth = 8, 
               nrounds = 25,
               label = y_train
)

# Predict 
y_pred <- predict(xgb, data.matrix(X_test))

plot(x = y_pred, y= y_test,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values',
     col = c("blue", "red"))

rmse(y_test, y_pred)
mean(abs((y_test-y_pred)/y_test)) * 100 
Gini(y_pred, y_test)
R2(pred, y_test)


########################## with grid search and CV ###########################

# Transformations

data <- data %>% 
  mutate_if(is.integer, as.numeric)

data <- data %>% 
  mutate_if(is.character, as.factor)

data <- data %>% 
  mutate_if(is.factor, as.numeric)

# Splitting 

smp_size <- floor(0.80 * nrow(data))
train_ind_gr <- sample(seq_len(nrow(data)), size = smp_size)

train_gr <- data[train_ind_gr, ]
test_gr <- data[-train_ind_gr, ]
  
# Grid search 
xgb_grid = expand.grid(
  nrounds = c(120,287),
  eta = c(0.03, 0.01),
  max_depth = c(2, 8, 12),
  gamma = c(0,1),
  colsample_bytree = c(0.7,0.8),
  min_child_weight = c(0,1),
  subsample = c(0.75,0.9)
)

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  allowParallel = TRUE
)

xgb_model <- train(target ~ . , data = train_gr, method = "xgbTree", trControl = xgb_trcontrol, tuneGrid = xgb_grid) # our case was with specific formula, implemented in mixed models

# Selecting tuning parameters
# Fitting nrounds = 287, max_depth = 12, eta = 0.03, gamma = 0, colsample_bytree = 0.7, min_child_weight = 0, subsample = 0.75 on full training set
print(xgb_model)
saveRDS(xgb_model, "./xgb_model.rds")

results <- xgb_model$results


pred <- predict(xgb_model, test_gr)

y_test <- test_gr$quantity_log

mean((y_test - pred)^2)
RMSE(y_test, pred) 
# Feature importance -----
xgb_imp <- varImp(xgb_model)
xgb_fe_imp <- xgb_imp$importance

write.csv(xgb_fe_imp, "xgb_fe_imp.csv", row.names = FALSE)

