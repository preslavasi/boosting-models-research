library(lightgbm)
library(mltools)
library(MLmetrics)
library(caret)
library(dplyr)
library(zoo)
library(data.table)
library(tidyr)

## Importing data -----
setwd("/Users/preslava_ivanova/Box Sync/Preslava/Mixed Models Investigation/Cat11")

set.seed(123)

# EDA -----
str(cat11_results_init)

sum(is.na(cat11_results_init))
colnames(cat11_results_init)[colSums(is.na(cat11_results_init)) > 0]

# Splitting data -----
X <- subset(cat11_results_init, select = -c(quantity_log))
y <- cat11_results_init$quantity_log

char <- X[, sapply(X, class) == 'character']
char <- char %>% 
  mutate_if(is.character, as.factor)
cat <- colnames(char)

######################### NO CV ########################
smp_size <- floor(0.80 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]


# Transformations for grid search -----
X_train <- X_train %>% 
  mutate_if(is.character, as.factor)
X_test <- X_test %>% 
  mutate_if(is.character, as.factor)

dtrain <- lgb.Dataset(data  = as.matrix(X_train),
                      label = as.matrix(y_train),
                      categorical_feature = cat, params = list(use_missing = TRUE))
dvalid <- lgb.Dataset(data  = as.matrix(X_test),
                      label = as.matrix(y_test),
                      categorical_feature = cat,  params = list(use_missing = TRUE))

# Setting parameters -----
params <- list(num_leaves     =   10.0,
               max_depth      =    3.0,
               subsample      =    0.7,
               colsample_bytree =  0.7,
               min_child_weight =  0.0,
               scale_pos_weight = 100.0,
               learning_rate   =   0.1,
               nthread          =  2.0,
               objective = "regression"
               , metric = "rmse"
) 

# Creating model -----
start_time <- proc.time()
lgbm_cv <- lgb.cv(
  params = params
  , data = dtrain
  , nrounds = 270
  , nfold = 5,
  verbose = 1,
  categorical_feature = cat,
  early_stopping_rounds = 5
)
(run_time <- proc.time() - start_time)

lgbm_cv$best_iter
lgbm_cv$best_score

# Prediction -----
lgb_model <- lgb.train(
  
)
pred <- predict(lgbm_cv, as.matrix(X_test))

# Evaluation ----- 
rmse(y_test, pred) # 0.05297805
mean(abs((y_test-pred)/y_test)) * 100 
Gini(pred, y_test) 
R2(pred, y_test)  

# too good to be true

# Save LGBM model -----
lgb.save(lgbm_model, tempfile(fileext = ".txt"))

saveRDS(lgbm_model, "lgbm_model_grid.rds")
# Importance -----
lgbm_fe_imp <- lgb.importance(lgbm_model)

plot(x = pred, y = y_test,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values',
     col = c("blue", "red"))
