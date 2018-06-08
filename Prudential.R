
install.packages('xgboost')
library(xgboost)

train <- read.csv("C:/Users/Brian Wozniak/Documents/DataScience/Prudential/train.csv")
test <- read.csv("C:/Users/Brian Wozniak/Documents/DataScience/Prudential/test.csv")

xgb.data <- xgb.DMatrix(data.matrix(train[,2:127]), label = train[,128])

ntrees = 100

searchGridSubCol <- expand.grid(subsample = c(.5, 1), 
                                colsample_bytree = c(.4, .6, .8, 1),
                                max_depth = c(4,6,8,10),
                                eta = c(2/ntrees, 4/ntrees, 6/ntrees, 8/ntrees, 10/ntrees)
)

ErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  currentSubsampleRate <- parameterList[["subsample"]]
  currentColsampleRate <- parameterList[["colsample_bytree"]]
  currentDepth <- parameterList[["max_depth"]]
  currentEta <- parameterList[["eta"]]
  xgboostModelCV <- xgb.cv(data =  xgb.data, nrounds = ntrees, nfold = 10, "max.depth" = currentDepth, "eta" = currentEta,                               
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate, early_stopping_rounds = 10)
})





xgb <- xgboost(xgb.data, nrounds = ntrees, eta = .1, max_depth = 5)

xgb.test <- xgb.DMatrix(data.matrix(test[,2:127]))
pred <- predict(xgb, xgb.test)

pred <- round(pred)
pred <- ifelse(pred < 1, 1, pred)

submission <- cbind(test$Id, pred)
colnames(submission) <- c("Id", "Response")
write.csv(submission, "C:/Users/Brian Wozniak/Documents/DataScience/Prudential/sub_xgboost.csv", row.names = FALSE)
