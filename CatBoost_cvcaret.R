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
Rprof(memory.profiling = TRUE)
ctb_trcontrol <- trainControl(method = "cv",
                              number = 4)

grid <- as.data.frame(expand.grid(depth = c(4, 6, 8),
                    learning_rate = c(0.1,0.03,0.01),
                    iterations = c(100, 124, 216),
                    l2_leaf_reg = 1e-3,
                    rsm = c(0.75,0.8,0.95),
                    border_count = 20))


data_t <- data %>%
  mutate_if(is.character, as.factor)%>% 
  mutate_if(is.factor, as.numeric)%>%
  mutate_if(is.logical, as.numeric)

# train_gr <- cat11_results_init_t[train_ind,]
# test_gr <- cat11_results_init_t[-train_ind, ]

start_time <- proc.time()
ctb_train <- train( target ~ formula,
                 X_train,y_train,
                   method = catboost.caret,trControl = ctb_trcontrol, tuneGrid=grid
                   )
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

ctb_train$bestTune

importance <- varImp(ctb_train, scale = FALSE)
print(importance)

# Prediction -----
ctb_pred <- predict(ctb_train, X_test)
ctb_pred

# pred_ext <- extractPrediction(
#   ctb_train,
#   testX = X_test,
#   testY = y_test_df)

results <- cbind(ctb_pred,y_test)
results <- cbind(X_test,results)
write.csv(results, "test_results_catbcvcaret.csv", row.names = FALSE)

# Evaluation -----
postResample(ctb_pred, y_test)
