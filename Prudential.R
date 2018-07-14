
#install.packages('xgboost')
#.61973 - Private Score

library(xgboost)

train <- read.csv("C:/Users/bwozn/Documents/DataScience/Prudential Kaggle/train.csv")
test <- read.csv("C:/Users/bwozn/Documents/DataScience/Prudential Kaggle/test.csv")

train_resp <- train$Response

train$Product_Info_2_char = as.factor(substring(train$Product_Info_2, 1, 1))
train$Product_Info_2_num = as.factor(substring(train$Product_Info_2, 2, 2))

drop <- "Response"

train <- train[ , !(names(train) %in% drop)]

test$Product_Info_2_char = as.factor(substring(test$Product_Info_2, 1, 1))
test$Product_Info_2_num = as.factor(substring(test$Product_Info_2, 2, 2))

# train_matrix <- data.matrix(train)
xgb.data <- xgb.DMatrix(data.matrix(train[,2:NCOL(train)]), label = train_resp)

# head(xgb.data)
ntrees = 100

head(train)

# searchGridSubCol <- expand.grid(subsample = c(.8, 1),
#                                 colsample_bytree = .8,
#                                 max_depth = c(8,20),
#                                 eta = .1
# )
# 
# ErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
# 
#   #Extract Parameters to test
#   currentSubsampleRate <- parameterList[["subsample"]]
#   currentColsampleRate <- parameterList[["colsample_bytree"]]
#   currentDepth <- parameterList[["max_depth"]]
#   currentEta <- parameterList[["eta"]]
#   xgboostModelCV <- xgb.cv(data =  xgb.data, nrounds = 10, nfold = 10, "max.depth" = currentDepth, "eta" = currentEta,
#                            "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate, early_stopping_rounds = 10)
#   xvalidationScores <- as.data.frame(xgboostModelCV$evaluation_log)
#   rmse <- tail(xvalidationScores$test_rmse_mean, 1)
#   trmse <- tail(xvalidationScores$train_rmse_mean,1)
#   output <- return(c(rmse, trmse, currentSubsampleRate, currentColsampleRate, currentDepth, currentEta))
# })
# 
# 
# output <- as.data.frame(t(ErrorsHyperparameters))
# head(output)

xgb <- xgboost(xgb.data, nrounds = ntrees, eta = .1, max_depth = 8, colsample_bytree = .8, subsample = .8)

xgb.test <- xgb.DMatrix(data.matrix(test[,2:NCOL(test)]))

pred <- predict(xgb, xgb.test)

pred <- round(pred)
pred <- ifelse(pred < 1, 1, pred)

submission <- cbind(test$Id, pred)
colnames(submission) <- c("Id", "Response")

write.csv(submission, "sub_xgboost.csv", row.names = FALSE)

#https://www.kaggle.com/silverstone1903/xgboost-grid-search-r used for grid search
#https://www.kaggle.com/aelvangunduz/analysis-of-data-using-xgboost used for improving score to .66
# https://www.kaggle.com/casalicchio/use-the-mlr-package-scores-0-649 used for improving score

#Medical Keyword Variables need to somehow be represented; they add info