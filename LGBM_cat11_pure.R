library(lightgbm)
library(mltools)
library(MLmetrics)
library(caret)
library(dplyr)
library(zoo)
library(data.table)
library(tidyr)

## Importing data -----
setwd("")

set.seed(123)

# EDA -----
str(data)

sum(is.na(data))
colnames(data)[colSums(is.na(data)) > 0]

# Splitting data -----
X <- subset(data, select = -c(target))
y <- data$target

char <- X[, sapply(X, class) == 'character']
char <- char %>% 
  mutate_if(is.character, as.factor)
cat <- colnames(char)

# Splitting data -----
smp_size <- floor(0.80 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]


# Transformations for grid search -----
X_train <- X_train %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.character, as.factor)

dtrain <- lgb.Dataset(data  = as.matrix(X_train),
                      label = as.matrix(y_train),
                      categorical_feature = cat, params = list(use_missing = TRUE))
dvalid <- lgb.Dataset(data  = as.matrix(X_test),
                      label = as.matrix(y_test),
                      categorical_feature = cat,  params = list(use_missing = TRUE))

# Grid search ----
grid_search <- expand.grid(
  num_leaves        = c(5,10,12),
  max_depth         = c(2,3),
  subsample         = c(0.7,0.9),
  colsample_bytree  = c(0.7,0.9),
  min_child_weight  = c(0,0.01),
  scale_pos_weight  = c(100,300),
  learning_rate = c(0.1,0.2),
  nthread = c(2,3),
  nrounds = c(80,110,120)
)
cat("Total number of models with REDUCED configuration: ", nrow(grid_search) , "\n")

model <- list()
perf <- numeric(nrow(grid_search))

start_time <- proc.time()
for (i in 1:nrow(grid_search)) {
  cat("Model ***", i , "*** of ", nrow(grid_search), "\n")
  model[[i]] <- lgb.train(
    list(objective         = "regression",
         metric            = c("rmse","mape"),
         learning_rate     = grid_search[i, "learning_rate"],
         # min_child_samples = 100,
         # max_bin           = 100,
         # subsample_freq    = 1,
         num_leaves        = grid_search[i, "num_leaves"],
         max_depth         = grid_search[i, "max_depth"],
         subsample         = grid_search[i, "subsample"]
         # colsample_bytree  = grid_search[i, "colsample_bytree"],
         # min_child_weight  = grid_search[i, "min_child_weight"],
         # scale_pos_weight  = grid_search[i, "scale_pos_weight"]
    ),
    data = dtrain,
    valids = list(validation = dvalid),
    nthread = grid_search[i, "nthread"], 
    nrounds = grid_search[i, "nrounds"],
    verbose= 1,
    categorical_feature = cat
    # ,early_stopping_rounds = 2,
    
  )
  perf[i] <- min(unlist(model[[i]]$record_evals[["validation"]][["rmse"]][["eval"]])) # problem with adding mape as well
  invisible(gc()) # free up memory after each model run
}
(run_time <- proc.time() - start_time)


# Grid search result
cat("Model ", which.min(perf), " is with min RMSE: ", min(perf), sep = "","\n")
best_params <- grid_search[which.min(perf), ]
# fwrite(best_params,"best_params_for_sample_data.txt")

cat("Best params within chosen grid search: ", "\n")
t(best_params)

# Setting parameters (based on best parameters from grid search) ----- 
params <- list(num_leaves     =   10.0,
               max_depth      =    3.0,
               subsample      =    0.7,
               colsample_bytree =  0.7,
               min_child_weight =  0.0,
               scale_pos_weight =100.0,
               learning_rate   =   0.1,
               nthread          =  2.0,
               nrounds        =  120.0,
               objective = "regression"
               , metric = "rmse"
) 

# Creating model -----
start_time <- proc.time()
lgbm_model <- lightgbm(
  data = dtrain
  , params = params
  , nrounds = 120
)
(run_time <- proc.time() - start_time)

# Prediction -----
pred <- predict(lgbm_model, as.matrix(X_test))

# Evaluation ----- 
rmse(y_test, pred)
mean(abs((y_test-pred)/y_test)) * 100 
Gini(pred, y_test) 
R2(pred, y_test)  

# too good to be true

# Save LGBM model -----
saveRDS(lgbm_model, "lgbm_model_grid.rds")

# Importance -----
lgbm_fe_imp <- lgb.importance(lgbm_model)

plot(x = pred, y = y_test,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values',
     col = c("blue", "red"))
