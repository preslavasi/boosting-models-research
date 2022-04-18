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
char <- X[, sapply(X, class) == 'character']
char <- char %>% 
  mutate_if(is.character, as.factor)
cat <- colnames(char)

X_train <- X_train %>% 
  mutate_if(is.character, as.factor) 
X_test <- X_test %>% 
  mutate_if(is.character, as.factor) 

# Modeling -----
fit_params <- list(iterations = 250,
                   depth = 8,
                   learning_rate = 0.01,
                   loss_function = 'RMSE',
                   eval_metric = 'RMSE',
                   random_seed = 55,
                   use_best_model = TRUE,
                   od_wait = 12)

Rprof(memory.profiling = TRUE)
train_pool <- catboost.load_pool(data = X_train, label = y_train, cat_features = c())
train_pool

start_time <- proc.time()
model <- catboost.train(train_pool, params = fit_params)
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
# Feature importances -----
model$feature_importances

fe_imp <- as_tibble(catboost.get_feature_importance(model, train_pool))
fe_inter <- catboost.get_feature_importance(model, train_pool, type = 'Interaction')

# feat_imp <- fe_imp %>%
#   rownames_to_column() %>%
#   select(Feature = rowname, Importance = V1 ) %>%
#   arrange(desc(Importance))

# ggplot(feat_imp, aes(x = Feature, y = Importance)) +
#   geom_bar(stat='identity') +
#   theme(axis.text.x= element_text(angle = 45)) +
#   scale_x_discrete(limits = feat_imp$Feature)

# Prediction ----- 
test_pool <- catboost.load_pool(data = X_test, label = y_test, cat_features = c())
test_pool

pred <- catboost.predict(model, test_pool)
pred

results <- cbind(pred,y_test)
results <- cbind(X_test,results)
write.csv(results, "test_results_catbpure.csv", row.names = FALSE)

# Evaluation -----
postResample(pred, y_test)

object_importance <- catboost.get_object_importance(model,
                                                    test_pool,
                                                    train_pool)
object_importance$scores
