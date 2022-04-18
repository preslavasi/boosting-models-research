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
# setwd("")
data

# EDA -----
str(data) 
summary(data)

# Check unique values for each column
apply(data, 2, function(x) length(unique(x)))

# Check for missing values
sum(is.na(data))

set.seed(123)

char <- X[, sapply(X, class) == 'character']
char <- char %>% 
  mutate_if(is.character, as.factor)
cat <- colnames(char)

# Splitting data -----
X <- subset(data, select = -c(target))
y <- data$target

smp_size <- floor(0.80 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]

# Transformations -----
X_train <- X_train %>% 
  mutate_if(is.character, as.factor)
X_test <- X_test %>% 
  mutate_if(is.character, as.factor)

# # Modeling -----
# ctb_trcontrol <- trainControl(method = "cv",
#                               number = 4)
# 
# grid <- expand.grid(depth = c(4, 6, 8),
#                     learning_rate = c(0.1,0.03,0.01),
#                     iterations = c(100, 124, 216),
#                     l2_leaf_reg = 1e-3,
#                     rsm = c(0.75,0.8,0.95),
#                     border_count = 20)
# 

# ################## CV and grid search ################## 
fit_params <- list(iterations = 216,
                   depth = 6,
                   learning_rate = 0.1,
                   rsm = 0.75,
                   border_count = 20,
                   l2_leaf_reg = 0.001,
                   loss_function = 'RMSE',
                   eval_metric = 'RMSE',
                   random_seed = 55,
                   use_best_model = TRUE,
                   od_wait = 12)

Rprof(memory.profiling = TRUE)
train_pool <- catboost.load_pool(data = X_train, label = y_train, cat_features = c())
start_time <- proc.time()
ctb_cv <- catboost.cv(train_pool, params = fit_params, fold_count = 4)
(run_time <- proc.time() - start_time)

summary(ctb_cv)


fit_params_af_cv <- list(iterations = 215,
                   depth = 6,
                   learning_rate = 0.1,
                   rsm = 0.75,
                   border_count = 20,
                   l2_leaf_reg = 0.001,
                   loss_function = 'RMSE',
                   eval_metric = 'RMSE',
                   random_seed = 55,
                   use_best_model = TRUE,
                   od_wait = 12)

train_pool <- catboost.load_pool(data = X_train, label = y_train, cat_features = c())
train_pool

# Training -----
start_time <- proc.time()
model_af_cv <- catboost.train(train_pool, params = fit_params_af_cv)
(run_time <- proc.time() - start_time)

path_to_intermed <- ""
Rprof(NULL)
summaryRprof(memory = "both")
p<-summaryRprof(memory = "both")
profRes <- data.table(
  mem.total.byself = sum(p$by.self$mem.total),
  mem.total.bytotal = sum(p$by.total$mem.total),
  time.total.byself = sum(p$by.self$total.time),
  time.total.bytotal = sum(p$by.total$total.time)
)
fwrite(profRes,
       paste0(path_to_intermed,"_profiling",".csv"))

model_af_cv $feature_importances

fe_imp_af_cv <- as_tibble(catboost.get_feature_importance(model_af_cv , train_pool))
fe_inter_af_cv <- catboost.get_feature_importance(model_af_cv, train_pool, type = 'Interaction')

# Prediction -----
test_pool <- catboost.load_pool(data = X_test, label = y_test, cat_features = c())
test_pool

ctb_pred_2 <- catboost.predict(model_af_cv, test_pool)
ctb_pred_2

results <- cbind(pred,y_test)
results <- cbind(X_test,results)
write.csv(results, "test_results_catbcvintegr.csv", row.names = FALSE)

# Evaluation -----
postResample(ctb_pred_2, y_test)

