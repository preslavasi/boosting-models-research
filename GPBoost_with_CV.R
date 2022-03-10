library(caret)
library(dplyr)
library(gpboost)
library(MLmetrics)
library(mltools)
library(stats)
library(utils)
library(zoo)

# Importing data -----
setwd("")

set.seed(123)

# EDA -----
str(data)

# Splitting data -----
X <- subset(data, select = -c(target))
y <- cdata$target

char <- X[, sapply(X, class) == 'character']
char <- char %>% 
  mutate_if(is.character, as.factor)
cat <- colnames(char)

# Splitting data NO CV -----
smp_size <- floor(0.80 * nrow(X))
train_ind <- sample(seq_len(nrow(X)), size = smp_size)

X <- X %>% 
  mutate_if(is.character, as.factor)
X <- X %>% 
  mutate_if(is.factor, as.numeric)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

y_train <- y[train_ind]
y_test <- y[-train_ind]

# Setting random effects -----
re <- c()

re_data <- X_train[, colnames(X_train) %in% re]

############################# NO CV ############################

# Transformations -----
dtrain <- gpb.Dataset(data = as.matrix(X_train), label = as.matrix(y_train))
dtrain <- gpb.Dataset.set.categorical(dtrain, categorical_feature = cat)

dvalid <- gpb.Dataset(data = as.matrix(X_test), label = as.matrix(y_test))
dvalid <- gpb.Dataset.set.categorical(dvalid, categorical_feature = cat)

# Creating covariates for features
covX <- cov(X[!names(X) %in% re])

# Modelling -----
gp_model <- GPModel(group_data = re_data, gp_rand_coef_data = covX)
                    
########################## CV with GPBoost ############################

# Grid search -----
params <- list(objective = "regression")
param_grid = list("learning_rate" = c(0.03,0.02,0.01), "min_data_in_leaf" = c(3,12,20),
                  "max_depth" = c(6,8,10), "max_bin" = c(87,164,255))

opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid,
                                              params = params,
                                              num_try_random = NULL,
                                              nfold = 8,
                                              data = dtrain,
                                              gp_model = gp_model,
                                              verbose_eval = 2,
                                              categorical_feature = cat,
                                              nrounds = 200,
                                              early_stopping_rounds = 5,
                                              eval = "rmse")
print(opt_params)

# best_params
# $best_params$learning_rate
# [1] 0.01
# 
# $best_params$min_data_in_leaf
# [1] 20
# 
# $best_params$max_depth
# [1] 6
# 
# $best_params$max_bin
# [1] 87
# 
# 
# $best_iter
# [1] 155
# 
# $best_score
# [1] 0.1523698

# Modelling -----

# Create model
params_m <- list(learning_rate = 0.01,
               max_depth = 6,
               min_data_in_leaf = 20,
               objective = "regression",
               max_bin = 87,
               verbose = 1
)

cvbst <- gpb.cv(params = params_m,
              data = dtrain,
              gp_model = gp_model,
              use_gp_model_for_validation = TRUE,
              nrounds = 155,
              nfold = 6,
              eval = "rmse",
              early_stopping_rounds = 5)

print(paste0("Optimal number of iterations: ", cvbst$best_iter,
             ", best test error: ", cvbst$best_score))

# "Optimal number of iterations: 155, best test error: 0.153448093535704" used on dtrain with covX

