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
cat11_results_init_nona <- cat11_results_init %>%
  group_by(priceline_id, promo_zone_id) %>%
  na.locf(is.na(cat11_results_init), na.rm = FALSE)

cat11_results_init_nona <- cat11_results_init_nona %>% 
  mutate_if(is.integer, as.numeric)

cat11_results_init_nona <- cat11_results_init_nona %>% 
  mutate_if(is.character, as.factor)

cat11_results_init_nona <- cat11_results_init_nona %>% 
  mutate_if(is.factor, as.numeric)

# Splitting 

smp_size <- floor(0.80 * nrow(cat11_results_init))
train_ind_gr <- sample(seq_len(nrow(cat11_results_init_nona)), size = smp_size)

train_gr <- cat11_results_init_nona[train_ind_gr, ]
test_gr <- cat11_results_init_nona[-train_ind_gr, ]
  
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

xgb_model <- train(quantity_log ~ 0 + loess_4:model_category_id + master_adj_spline:promo_zone_id + 
                     price_log:priceline_id + promo_tact_min_purch_bagb_pct:promo_zone_id + 
                     promo_feat_ad_2:promo_zone_id + cpi_index_log:promo_zone_id + 
                     promo_overlay_2_val:promo_zone_id + hol_newyear:promo_zone_id + 
                     hol_postnewyear:promo_zone_id + hol_superbowl:promo_zone_id + 
                     hol_week_prexms:promo_zone_id + hol_pre_thanksgiving:promo_zone_id + 
                     hol_pos_flag:promo_zone_id + hol_co01_r:promo_zone_id + hol_co02_r:promo_zone_id + 
                     hol_co03_r:promo_zone_id + hol_co04_r:promo_zone_id + hol_co05_r:promo_zone_id + 
                     hol_co06_r:promo_zone_id + hol_co07_r:promo_zone_id + hol_co08_r:promo_zone_id + 
                     hol_co09_r:promo_zone_id + hol_co10_r:promo_zone_id + promo_coupon_2_pct:model_cpi_group + 
                     hol_week_xms:model_cpi_group + hol_first_shop_day:model_cpi_group + 
                     hol_labor_day:model_cpi_group + hol_valentines:model_cpi_group + 
                     hol_cinco_mayo:model_cpi_group + hol_co01:model_cpi_group + 
                     hol_co02:model_cpi_group + hol_co03:model_cpi_group + hol_co04:model_cpi_group + 
                     hol_co05:model_cpi_group + hol_co06:model_cpi_group + hol_co07:model_cpi_group + 
                     hol_co08:model_cpi_group + hol_co09:model_cpi_group + hol_co10:model_cpi_group + 
                     promo_tact_promo_price_pct:priceline_id + promo_overlay_1_val:priceline_id + 
                     promo_feat_ad_1:priceline_id + promo_tact_bagb_pct:priceline_id + 
                     promo_feat_ad_3:is_private_label + covid_price_log:hol_covid + 
                     promo_tact_bqgq_pct:promo_zone_id + promo_tact_dol_off_pct:promo_zone_id + 
                     promo_feat_ad_front_pct:promo_zone_id + promo_feat_ad_back_pct:promo_zone_id + 
                     promo_feat_ad_wrap_pct:promo_zone_id + promo_feat_ad_inside_pct:promo_zone_id + 
                     hol_veterens:promo_zone_id + promo_tact_pct_off_pct + promo_log_glbl_cpn_rdm + 
                     promo_overlay_3_pct + promo_overlay_4_pct + promo_overlay_5_pct + 
                     promo_coupon_1_pct + promo_tact_min_purch_pct + promo_tact_mlt_buy_pct + 
                     hol_stock_out + (0 + promo_tact_bqgq_pct | key) + (0 + promo_tact_dol_off_pct | 
                                                                          key) + (0 + promo_feat_ad_front_pct | key) + (0 + promo_feat_ad_back_pct | 
                                                                                                                          key) + (0 + promo_feat_ad_wrap_pct | key) + (0 + promo_feat_ad_inside_pct | 
                                                                                                                                                                         key) + (0 + hol_veterens | key) + (0 + price_log | key) + 
                     (1 | key), data = train_gr, method = "xgbTree", trControl = xgb_trcontrol, tuneGrid = xgb_grid)

# Selecting tuning parameters
# Fitting nrounds = 287, max_depth = 12, eta = 0.03, gamma = 0, colsample_bytree = 0.7, min_child_weight = 0, subsample = 0.75 on full training set
print(xgb_model)
saveRDS(xgb_model, "./xgb_model.rds")

results <- xgb_model$results

# RMSE        R2           MAE          RMSESD         R2SD          MAESD
# 0.3756644   0.9538713    0.2612461    0.010969988    0.002924889   0.005602590

pred <- predict(xgb_model, test_gr)

y_test <- test_gr$quantity_log

mean((y_test - pred)^2) # 0.1309525
RMSE(y_test, pred) # 0.3618736

# Feature importance -----
xgb_imp <- varImp(xgb_model)
xgb_fe_imp <- xgb_imp$importance

write.csv(xgb_fe_imp, "xgb_fe_imp.csv", row.names = FALSE)

