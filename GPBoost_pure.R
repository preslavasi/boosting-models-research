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
y <- data$target

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

# Transformations -----
dtrain <- gpb.Dataset(data = as.matrix(X_train), label = as.matrix(y_train))
dtrain <- gpb.Dataset.set.categorical(dtrain, categorical_feature = cat)

dvalid <- gpb.Dataset(data = as.matrix(X_test), label = as.matrix(y_test))
dvalid <- gpb.Dataset.set.categorical(dvalid, categorical_feature = cat)

# Creating covariates for features
covX <- cov(X[!names(X) %in% re])

# Modelling -----
gp_model <- GPModel(group_data = re_data, gp_rand_coef_data = covX)

# Train model
bst <- gpboost(dtrain,
               gp_model = gp_model,
               nrounds = 200,
               learning_rate = 0.02,
               max_depth = 8,
               min_data_in_leaf = 3,
               objective = "regression",
               verbose = 0)

summary(gp_model)

filename <- tempfile(fileext = "gpbst.txt")
gpb.save(bst,filename = filename)

# Prediction -----
re_data_test <- X_test[, colnames(X_test) %in% re]

pred <- predict(bst, data = as.matrix(X_test), group_data_pred = re_data_test)
summary(pred)

pred$fixed_effect
pred$random_effect_mean

y_pred <- pred$fixed_effect + pred$random_effect_mean

# Evaluation ----- 
rmse(y_test, y_pred) # 0.4056306 => with covX  0.406733
mean(abs((y_test-y_pred)/y_test)) * 100 # 2.971379 => with covX 3.008614
Gini(y_pred, y_test) # 0. 0.9858373 => with covX 0.9856584
R2(y_pred, y_test) # 0.9522957 => with covX 0.9522556
RMSE <- sqrt(mean((y_pred - y_test)^2))

# Feature importance -----
fe_imp <- gpb.importance(bst, percentage = TRUE)

gpb.plot.importance(fe_imp, top_n = 20, measure = "Gain") 

# write.csv(fe_imp, "fe_imp_pure_GPBoost.csv", row.names = FALSE)
