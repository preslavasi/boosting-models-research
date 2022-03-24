library(tidyverse)
library(ggplot2)
library(tidyr)
library(dplyr)
library(catboost)
library(mltools)
library(data.table)
library(caTools)
library(MLmetrics)
library(caret)

## Importing data -----
setwd("")
data <- data_sleepstudy

# EDA -----
str(data) 
summary(data)

# Check unique values for each column
apply(data, 2, function(x) length(unique(x)))

# Reaction     Days  Subject 
# 180       10       18 

# Check for missing values
sum(is.na(data))

# Plot data
xyplot(Reaction ~ Days | Subject, data = sleepstudy, layout = c(6, 3))

# Modelling -----
set.seed(123)

# Splitting data -----
X <- data[,c(2,3)]
y <- data[,1]

X$Subject <- as.factor(X$Subject)

smp_size <- floor(0.80 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]

# Modeling -----

fit_params <- list(iterations = 250,
                   depth = 8,
                   learning_rate = 0.01,
                   loss_function = 'RMSE',
                   eval_metric = 'RMSE',
                   random_seed = 55,
                   use_best_model = TRUE,
                   od_wait = 12)

train_pool <- catboost.load_pool(data = X_train, label = y_train, cat_features = c(2))
train_pool
 
model <- catboost.train(train_pool, params = fit_params)

# Feature importances -----
model$feature_importances

fe_imp <- as_tibble(catboost.get_feature_importance(model, train_pool))
fe_inter <- catboost.get_feature_importance(model, train_pool, type = 'Interaction')

feat_imp <- fe_imp %>%
  rownames_to_column() %>%
  select(Feature = rowname, Importance = V1 ) %>%
  arrange(desc(Importance))

ggplot(feat_imp, aes(x = Feature, y = Importance)) +
  geom_bar(stat='identity') +
  theme(axis.text.x= element_text(angle = 45)) +
  scale_x_discrete(limits = feat_imp$Feature)

# Prediction ----- 
test_pool <- catboost.load_pool(data = X_test, label = y_test, cat_features = c(2))
test_pool

pred <- catboost.predict(model, test_pool)
pred

# Evaluation -----
postResample(pred, y_test)

object_importance <- catboost.get_object_importance(model,
                                                    test_pool,
                                                    train_pool)
object_importance$scores

################## CV and grid search ################## 

ctb_trcontrol <- trainControl(method = "cv",
                            number = 4)

grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = c(0.1,0.03,0.01),
                    iterations = c(100, 124, 216),
                    l2_leaf_reg = 1e-3,
                    rsm = c(0.75,0.8,0.95),
                    border_count = 20)

ctb_train <- train(X_train, y_train,
                method = catboost.caret,
                logging_level = 'Verbose',
                tuneGrid = grid, trControl = ctb_trcontrol)
ctb_train

importance <- varImp(ctb_train, scale = FALSE)
print(importance)

# Prediction -----
ctb_pred <- predict(ctb_train, X_test)
ctb_pred

# Evaluation -----
postResample(ctb_pred, y_test)


