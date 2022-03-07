library(mltools)
library(data.table)
library(caTools)
library(gpboost)
library(lme4)
library(lattice)

##### TEST 1 #####

# Importing data -----
setwd("/Users/pivanova/Documents/Investigations/GPBoost")
data("sleepstudy")
data <- sleepstudy

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

# Splitting data 
X <- data[,c(2,3)]
y <- data[,1]

# Splitting data
smp_size <- floor(0.80 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

X_train <- X[train_ind, ]
X_test <- X[-train_ind, ]

# OHE 
X_train_ohe <- one_hot(data.table(X_train))
X_test_ohe <- one_hot(data.table(X_test))

y_train <- y[train_ind]
y_test <- y[-train_ind]

# Grouping data 
group_train <- rep(1,nrow(X_train)) # grouping variable
for(i in 1:ncol(X_train)) group_train[((i-1)*nrow(X_train)/ncol(X_train)+1):(i*nrow(X_train)/ncol(X_train))]<- i
group_test <- 1:nrow(X_test)

# Create model
gp_model <- GPModel(group_data = group_train)

# Train model
X_train_t <- as.matrix(X_train)

bst <- gpboost(X_train_t,
               label = y_train,
               gp_model = gp_model,
               nrounds = 500,
               learning_rate = 0.06,
               max_depth = 10,
               min_data_in_leaf = 3,
               objective = "regression",
               verbose = 0)

summary(gp_model)

# Covariance parameters:
#   Error_term    Group_1 
# 2958.54000    0.65488 

# Prediction -----
X_test_t <- as.matrix(X_test)

y_pred <- predict(bst, data = X_test_t, group_data_pred = group_test)
summary(pred)

pred$fixed_effect
pred$random_effect_mean
pred$random_effect_cov
pred$response_mean
pred$response_var


##### TEST 2 #####

# Modelling -----

# Create model
y_train_t <- as.matrix(y_train)
group_train_test <- X_train_ohe[,2:19]
X_train_ohe <- as.matrix(X_train_ohe)

gp_model_2 <- fitGPModel(group_data=group_train, likelihood="gaussian",                               
                       y=y_train_t, X=X_train_t, params = list(std_dev = TRUE))
summary(gp_model_2)

# Covariance parameters:
#             Error_term  Group_1
# Param.      2063.950  1.84747
# Std. dev.    244.946 30.70250
# Linear regression coefficients:
#           Days   Subject
# Param.    10.51750 0.7303850
# Std. dev.  1.32761 0.0206084

coefs <- gp_model_2$get_coef()
z_values <- coefs[1,] / coefs[2,]
p_values <- 2 * exp(pnorm(-abs(z_values), log.p=TRUE))
coefs_summary <- rbind(coefs, z_values, p_values)
print(signif(coefs_summary, digits=4)) # show p-values

#           Days    Subject
# Param.    1.052e+01  7.304e-01
# Std. dev. 1.328e+00  2.061e-02
# z_values  7.922e+00  3.544e+01
# p_values  2.334e-15 3.968e-275

# Prediction -----
X_test_t <- as.matrix(X_test)

pred <- predict(gp_model_2, X_pred=X_test_t,
                group_data_pred=group_test,
                predict_var=TRUE, predict_response=FALSE)

pred$mu # predicted latent mean

# [1] 235.4844 246.0252 319.6162 267.7590 299.3116 300.0420 310.5595 262.0620 283.8275 325.8976 295.0754 243.2181 285.2882 295.8058
# [15] 337.8759 286.0186 255.1964 318.3016 288.2098 319.7624 340.7974 265.4218 349.5620 255.6347 287.1873 256.3651 266.8826 298.4352
# [29] 340.5053 278.1305 288.6480 311.5821 322.0996 364.1697 333.3475 364.9001

pred$var # predicted latent variance

# [1] 2065.684 2065.684 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796
# [15] 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796
# [29] 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796 2065.796


##### TEST 3 #####

# Modelling -----

# Create model
y_train_t <- as.matrix(y_train)
group_train_ohe <- X_train_ohe[,2:19]
X_train_ohe <- as.matrix(X_train_ohe)

gp_model_2_ohe <- fitGPModel(group_data=group_train_ohe, likelihood="gaussian",                               
                         y=y_train_t, X=X_train_ohe, params = list(std_dev = TRUE))
summary(gp_model_2_ohe)

coefs <- gp_model_2_ohe$get_coef()
z_values <- coefs[1,] / coefs[2,]
p_values <- 2 * exp(pnorm(-abs(z_values), log.p=TRUE))
coefs_summary <- rbind(coefs, z_values, p_values)
print(signif(coefs_summary, digits=4)) # show p-values

# Prediction -----
group_test_ohe <- X_test_ohe[,2:19]
X_test_ohe <- as.matrix(X_test_ohe)

pred_ohe <- predict(gp_model_2_ohe, X_pred=X_test_ohe,
                group_data_pred=group_test_ohe,
                predict_var=TRUE, predict_response=FALSE)

pred_ohe$mu 
pred_ohe$var

# Calculate mean for y_test 


# Plot data 
plot(x = pred$mu , y= y_test,
     xlab='Predicted Values',
     ylab='Actual Values',
     main='Predicted vs. Actual Values')
abline(a=0, b=1)