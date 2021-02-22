rm(list= ls()) # Clear environment variables
library(tidyverse) # Data Wrangling
library(caret) # ML Toolbox, Cross-validation
library(pROC) # to plot ROC 
############ Decision Tree Supporting Packages ###########
library(rpart) # Decision tree model
library(rpart.plot) # plot Decision tree model
library(gbm) # Boosting
library(e1071) # SVM
library(kernlab)
library(kknn)
library(NeuralNetTools) 
#source("ml_diagnostics1.R")
# Function to build learning curve data
# Input:
#   tdata: train data
#   nparts: number of train data subsets to train on
#   modform: model formula. e.g. y ~ x
#   modtype: name of model in Caret. e.g. "lm" for linear regression
#   wts: weights of observations in the train data
#   fitCtr: fit control object for tuning. Needed because Caret sometimes changes the values of the tuning parameter as it has a default range.
#   tg: tuning grid.

lcurve <- function(tdata = NULL,
                   nparts = NULL,
                   modform = NULL,
                   modtype = NULL,
                   wts = NULL,
                   fitCtr = NULL,
                   tg = NULL,
                   #tunel = NULL,
                   metric = 'mse',
                   funName = NULL,
                   y = NULL) {
  res <- data.frame(
    trainSize = integer(nparts),
    trainMSE = integer(nparts),
    testMSE = integer(nparts),
    fit_time = numeric(nparts),
    trainAccuracy = numeric(nparts),
    testAccuracy = numeric(nparts)
  )
  print('hj')
  # partition the data incrementally
  partsize = floor(nrow(tdata) / nparts)
  trnset = NULL
  for(p in 1:nparts)
  { 
    set.seed(2 * p) # randomize the seed to get different samples
    trnids = sample(nrow(tdata), size = partsize * p, replace = F)
    trnset = tdata[trnids,]
    tstset = tdata[-trnids,]
    trnids %>% length() %>% print()
    if (is.null(funName)){print('hello')
      start = Sys.time()
      fit = train(
        form = modform,
        data = trnset,
        method = modtype,
        tuneGrid = tg,
        tuneLength = 1,
        trControl = trainControl(method = 'none', #no cv
                                 verboseIter = T)
      )
      end = Sys.time()
    } else{ print(funName)
      start = Sys.time()
      fit = funName(modform,
                    data = trnset,
                    kernel = "polynomial",
                    type = 'C-classification',
                    #class.weights = c('0' = 1, '1' = 2.4),
                    cost =0.01,
                    d =3)
      
      end = Sys.time()
      print('fit complete')
    }
    fitted = predict(fit, newdata = trnset)
    predicted = predict(fit, newdata = tstset)
    res[p, 'trainSize'] = length(trnids)
    
    if(metric == 'mse'){
      res[p, 'trainMSE'] = Metrics::mse(actual = pull(trnset, y), predicted = fitted)
      res[p, 'testMSE'] = Metrics::mse(actual = pull(tstset, y), predicted = predicted)
    } else if (metric == 'accuracy'){
      res[p, 'trainAccuracy'] = Metrics::accuracy(actual = pull(trnset, y), predicted = fitted)
      res[p, 'testAccuracy'] = Metrics::accuracy(actual = pull(tstset, y), predicted = predicted)
    }
    res[p, 'fit_time'] = end - start
  }
  return(res)
}

lcplot <- function(lcurvedf, metric, modname, nparts,  ylim=c(.5,1)) {
  lcurvedf %>%
    gather(key = 'Measure', value = metric,grep(x= names(.), pattern = metric, value = T, ignore.case = T)) %>%
    ggplot(aes(x = trainSize, y = metric, color = Measure)) +
    geom_line(lwd = .0001) +
    geom_point() +
    scale_x_continuous(n.breaks = nparts) +
    scale_y_continuous(n.breaks = nparts, limits = ylim) +
    geom_smooth(se = F, lwd = 2) +
    theme_bw() +
    labs(
      title = paste0(modname, ' - ', 'Learning Curve'),
      x = 'Sample Size',
      y = metric
    )
}

lcTimePlot <- function(lcurvedf, metric, modname, nparts, ylim=c(0,1)) {
  lcurvedf %>%
    ggplot(aes(x = trainSize, y = fit_time)) +
    geom_line(lwd = .0001) +
    geom_point() +
    scale_x_continuous(n.breaks = nparts) +
    scale_y_continuous(n.breaks = nparts, limits = ylim) +
    geom_smooth(se = F, lwd = 2) +
    theme_bw() +
    labs(
      title = paste0(modname, ' - ', 'Fit Time Relative to Training set size'),
      x = 'Sample Size',
      y = 'Training Time (Secs)'
    )
}


# Read-in the data
df <-
  read.delim("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", header = F, sep = ' ')

df %>% head()
df %>% summary()
df %>% glimpse()
df %>% dim()

# Convert Qualitative Variables (as described on the UCI website) type factor
df[,c(1,3,4,6,7,9,10,12,14,15,17,19,20)] <- lapply(df[,c(1,3,4,6,7,9,10,12,14,15,17,19,20)], factor) 

df %>% glimpse()

df %>% is.na() %>% sum() # No missing values

# Re-code the dependent variable to have a meaningful name (credit) and better factor codes whereby: 0 is good and 1 is bad
df %>% rename('credit' = 'V21') -> df
(df$credit -1) %>% factor() ->  df$credit

# Split between train and test
set.seed(24)
trainID = sample(x = nrow(df),
                 size = ceiling(nrow(df) *.7),
                 replace = F)
trainset <- df[trainID,]
testset <- df[-trainID,]

# Class imbalance 
table(trainset$credit)
(1/table(trainset$credit)[1]) * 0.5 
(1/table(trainset$credit)[2]) * 0.5

mod_wts <- ifelse(trainset$credit == '0',
                  (1/table(trainset$credit)[1]) * 0.5,
                  (1/table(trainset$credit)[2]) * 0.5 )

trainset %>% glimpse()

## Standardize the numeric variables ##
transformation = preProcess(trainset, method = c("center", 
                                                       "scale"))
trainset_transformed = predict(transformation, trainset) # Standardize Train set
testset_transformed = predict(transformation, testset) # Standardize Test set


# Cross validation 5 folds (stratified on dependent variable for 
# class balancing)
set.seed(42)
folds_5 <- createFolds(trainset$credit,k = 5) # Create 5 fold indicies

# Create trainControl object for modeling
trainctr <- trainControl(
  method = 'cv',
  index = folds_5,
  seeds = list(11:20,
               21:30,
               31:40,
               41:50,
               51:60,
               60),
  verboseIter = T,
  allowParallel = TRUE#, 
  #classProbs = T # Return class probabilities
  ) 

########## Fit a Decision Tree Model ################
#########

##### Steps 1: Fit base model: No CV, No Complexity Parameter Tuning ####
set.seed(232)
# Fit a base (no cross validation) decision tree model
dt_fit_base <- rpart::rpart(formula = credit ~ .,
                            data = trainset_transformed, 
                            method = 'class', # Classification
                            control = rpart.control(
                              xval = 0, # No cross validation
                              cp = 0.001
                            ),
                            weights = mod_wts)
printcp(dt_fit_base)
dt_fit_base %>% summary()
# Plot the fitted base tree model
dt_fit_base %>% rpart.plot(main = "Classification Tree Model (Base)",
                           extra = 101, # n & % of observations
                           under = F, type = 1 
)

###### Base Tree Model Error Rate on Train Set #########
dt_base_fit <- predict(dt_fit_base, newdata = trainset_transformed, type ='class')

dt_base_fit_accuracy = mean(trainset$credit == dt_base_fit)

caret::confusionMatrix(data = dt_base_fit,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"))


###### Base Tree Model Error Rate on Test Set #########
dt_base_test_predictions <- predict(dt_fit_base, newdata = testset_transformed, type ='class')

dt_base_test_accuracy = mean(testset$credit == dt_base_test_predictions)

caret::confusionMatrix(data = dt_base_test_predictions,
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"))

################################################################################
################################################################################
##### Steps 2: Re-Fit base model with tuning: CV, Complexity Parameter Tuning
##### and tree pruning 
set.seed(234)
dt_ctr <- rpart.control(
  #minsplit = 2,
  minbucket = round(0.01 * (nrow(trainset))),# min number of obs in a leaf
  cp = 0.0001, # Complexity Parameter for minimum Classification Improvement required at each split. This controls for overfitting.
  xval = 10#, # 10 fold Cross-Validation
 # maxdepth = 8
)
set.seed(232)
dt_fit_tuned <- rpart::rpart(formula = credit ~ .,
                            data = trainset_transformed, 
                            method = 'class', # Classification
                            control = dt_ctr,
                            weights = mod_wts)

# Plot the fitted tuned tree model
dt_fit_tuned %>% rpart.plot(main = "Classification Tree Model (Tuned)",
                           extra = 101, # n & % of observations
                           under = F, type = 1 
)

rpart::plotcp(dt_fit_tuned)
rpart::printcp(dt_fit_tuned)

######### Prune the tuned model ########
# Prune the model using the complexity parameter with lowest cross validation error
set.seed(232)
dt_fit_tuned_pruned <- prune(dt_fit_tuned, cp= dt_fit_tuned$cptable[which.min(dt_fit_tuned$cptable[,"xerror"]),"CP"])

#rpart::plotcp(dt_fit_tuned_pruned)
#rpart::printcp(dt_fit_tuned_pruned)

# Plot the fitted Pruned tree model
dt_fit_tuned_pruned %>% rpart.plot(main = "Classification Tree Model (Pruned)",
                                   extra = 101, # n & % of observations
                                   under = F, type = 1 
)

######## Train Error Rate ############
dt_Tuned_pruned_fit <- predict(dt_fit_tuned_pruned, newdata = trainset_transformed, type ='class')


dt_pruned_fit_accuracy = mean(trainset$credit == dt_Tuned_pruned_fit)
caret::confusionMatrix(data = dt_Tuned_pruned_fit,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## Test Error Rate ###########
dt_Tuned_pruned_test <- predict(dt_fit_tuned_pruned, newdata = testset_transformed, type ='class')


dt_pruned_fit_accuracy = mean(testset$credit == dt_Tuned_pruned_test)
caret::confusionMatrix(data = dt_Tuned_pruned_test,
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

# Plot Learning Curve
lcurve(tdata = trainset_transformed, nparts = 10, 
       modform = credit ~ .,
       modtype =  "rpart",
       wts = mod_wts,
       y = 'credit',
       metric = "accuracy",
       tg = data.frame(cp =dt_fit_tuned$cptable[which.min(dt_fit_tuned$cptable[,"xerror"]),"CP"]) #complexity parameter: cp
) -> learning_curve_dt
learning_curve_dt %>% 
gather(key = 'Measure', value = metric,grep(x=names(.), pattern = "accuracy",ignore.case = T, value = T))

lcplot(
  lcurvedf = learning_curve_dt,
  metric = 'Accuracy',
  modname = 'DT',
  nparts = 12
)

plot(x = learning_curve_dt$trainSize, 
     y = learning_curve_dt$fit_time,
     main = 'DT Fit Time Relative to Training set size',
     xlab = '# of Fitted Observations',
     ylab = 'Fit Time',
     type="b", 
     lwd= 2,
     lty=2, col='blue', pch=1)
dt_basePredict <- predict(dt_fit_base, newdata = testset_transformed, type ='class')

dt_base_accuracy = mean(testset$credit == dt_basePredict)

caret::confusionMatrix(data = dt_basePredict,
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"))

########################################################################
############ Model 2: Neural Nets #########
############ 
############ 
############  Step 1 fit a base model ###################################
# Base model has 1 node and no decay, max iterations is 100
set.seed(2043)
nnet_base_fit <- nnet::nnet(credit ~.,
                        data = trainset_transformed,
                        wts = mod_wts,
                        size = 1,
                        decay = 0,
                        linout = F)

nnet_base_fit %>% summary()
library(NeuralNetTools)
NeuralNetTools::plotnet(nnet_base_fit)
######## NNET Base Train Error Rate ############
nnet_base_fitted <- predict(nnet_base_fit, newdata = trainset_transformed, type ='class')

nnet_base_fit_accuracy = mean(trainset$credit == nnet_base_fitted)
caret::confusionMatrix(data = factor(nnet_base_fitted),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## NNET Base Test Error Rate ###########
nnet_base_predicted <- predict(nnet_base_fit, newdata = testset_transformed, type ='class')

nnet_base_fit_accuracy = mean(testset$credit == nnet_base_predicted)
caret::confusionMatrix(data = factor(nnet_base_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)
#############################################################
################## Fit a Tuned NNET Model ##############

# Construct the tuning grid
size = c(1,2,3,4,5) # number of nodes in the hidden layer
decay = seq(from = 0.001, to = 0.1, by = 0.001) # regularization of weights
nnGrid = expand.grid(size = size, decay = decay)    

set.seed(242) 
trainctr2 <- trainControl(
  method = 'cv',
  index = folds_5,
  seeds = list(1:500,
               801:1300,
               601:1100,
               201:700,
               101:600,
               60),
  verboseIter = T,
  allowParallel = TRUE#, 
  #classProbs = T # Return class probabilities
)     
set.seed(242) 
nnet_tuned <- train(
  credit ~ .,
  data = trainset_transformed,
  method = "nnet",
  weights = mod_wts, 
  tuneGrid = nnGrid,
  trControl = trainctr2
)
summary(nnet_tuned)
print(nnet_tuned)
plot(nnet_tuned)
nnet_tuned$results
nnet_tuned$bestTune

############# NNET Tuned Train Error ##########################
nnet_tuned_fitted <- predict(nnet_tuned, newdata = trainset_transformed, type = "raw")

caret::confusionMatrix(data = nnet_tuned_fitted,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"))

############# NNET Tuned Test Error ##########################
nnet_tuned_predictions <- predict(nnet_tuned, newdata = testset_transformed, type = "raw")

nn_tuned_predict_accuracy = mean(testset$credit == nnet_tuned_predictions)

caret::confusionMatrix(data = nnet_tuned_predictions,
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"))

### Plot NNET Learning Curve
lcurve(tdata = trainset_transformed, nparts = 10, 
       modform = credit ~ .,
       modtype =  "nnet",
       wts = mod_wts,
       y = 'credit',
       metric = "accuracy",
       tg = data.frame(size = 1, decay = nnet_tuned$bestTune$decay)
) -> learning_curve_nnet
learning_curve_nnet %>% 
  gather(key = 'Measure', value = metric,grep(x=names(.), pattern = "accuracy",ignore.case = T, value = T))

lcTimePlot(
  lcurvedf = learning_curve_nnet,
  metric = 'Accuracy',
  modname = 'Neural Net',
  nparts = 12,
  ylim = c(0.2,0.5)
)

########################## W######################################
##################BOOSTING ##############
##################

# Model control object
set.seed(232)

boostedT_base_fit <- train(
  credit ~ .,
  data = trainset_transformed,
  method = 'gbm',
  tuneGrid = data.frame(
    n.trees = 250, # number of trees
    interaction.depth = 1, # number of split per tree
    shrinkage = 0.01,# learning rate fo boosting
    n.minobsinnode = 20), 
  trControl = trainControl(
    method = "none" # no Cross validation
  ),
  weights = mod_wts
)
boostedT_base_fit
######## Boosting Base Train Error Rate ############
boosting_base_fitted <- predict(boostedT_base_fit, newdata = trainset_transformed, type ='raw')

boosting_base_fit_accuracy = mean(trainset$credit == boosting_base_fitted)
caret::confusionMatrix(data = boosting_base_fitted,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## Boosting Base Test Error Rate ###########
boosting_base_predicted <- predict(boostedT_base_fit, newdata = testset_transformed, type ='raw')

boosting_test_fit_accuracy = mean(testset$credit == boosting_base_predicted)
caret::confusionMatrix(data = (boosting_base_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)
#############################################################
############# Tune the Base Boosting Model ##############
boosted_tgrid <-
  expand.grid(
    n.trees = 10:500,
    interaction.depth = 1:3, # splits per tree controls boosting complexity 
    shrinkage = seq(from = 0.001, to = 0.01, by = 0.004),
   # n.minobsinnode = round(0.5 * nrow(trainset))
   n.minobsinnode = 10
  )
trainctr3 <- trainControl(
  method = 'cv',
  index = folds_5,
  seeds = list(1:300,
               801:1100,
               601:921,
               201:521,
               101:421,
               60),
  verboseIter = T,
  allowParallel = TRUE#, 
  #classProbs = T # Return class probabilities
)     
set.seed(232)
boostedT_tune_fit <- train(
  credit ~ .,
  data = trainset_transformed,
  method = "gbm",
  weights = mod_wts, 
  tuneGrid = boosted_tgrid,
  trControl = trainctr3
)
 
boostedT_tune_fit$bestTune$n.trees
boostedT_tune_fit$bestTune$interaction.depth
boostedT_tune_fit$bestTune$shrinkage
plot(boostedT_tune_fit)
boostedT_tune_fit$bestTune

######## Boosting Base Train Error Rate ############
boosting_tuned_fitted <- predict(boostedT_tune_fit, newdata = trainset_transformed, type ='raw')

boosting_tuned_fit_accuracy = mean(trainset$credit == boosting_tuned_fitted)
caret::confusionMatrix(data = (boosting_tuned_fitted),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## Boosting Base Test Error Rate ###########
boosting_tuned_predicted <- predict(boostedT_tune_fit, newdata = testset_transformed, type ='raw')

boosting_tuned_test_accuracy = mean(testset$credit == boosting_tuned_predicted)
caret::confusionMatrix(data = (boosting_tuned_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

##################################### 
# Plot Learning Curve
lcurve(tdata = trainset_transformed, nparts = 10, 
       modform = credit ~ .,
       modtype =  "gbm",
       wts = mod_wts,
       y = 'credit',
       metric = "accuracy",
       tg = data.frame(n.trees = boostedT_tune_fit$bestTune$n.trees, 
                       interaction.depth = boostedT_tune_fit$bestTune$interaction.depth,
                       shrinkage = boostedT_tune_fit$bestTune$shrinkage,
                       n.minobsinnode =boostedT_tune_fit$bestTune$n.minobsinnode
                      )
) -> learning_curve_boostedT
learning_curve_boostedT %>% 
  gather(key = 'Measure', value = metric,grep(x=names(.), pattern = "accuracy",ignore.case = T, value = T))

lcplot(
  lcurvedf = learning_curve_boostedT,
  metric = 'Accuracy',
  modname = 'Boosted Trees',
  nparts = 12
)

lcTimePlot(
  lcurvedf = learning_curve_boostedT,
  metric = 'Accuracy',
  modname = 'Boosted Trees',
  nparts = 12,
  ylim = c(0.1,2.5)
)

################ Support Vector Machine #############
################ 
###################SVM ##############
##################

# 
set.seed(232)
SVM_base_fit <- e1071::svm(credit ~., 
                    data = trainset_transformed,
                    scale = F, #data is scaled already
                    kernel = 'linear', 
                    type = 'C-classification',
                    cost = 0, # cost of constraint violation
                    class.weights = c('0' = 1, '1' = 4.4),
)

SVM_base_fit
######## SVM Base Train Error Rate ############
svm_base_fitted <- predict(SVM_base_fit, newdata = trainset_transformed, type ='raw')

svm_base_fit_accuracy = mean(trainset$credit == svm_base_fitted)
caret::confusionMatrix(data = svm_base_fitted,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_base_Fit_confM
print(svm_base_Fit_confM)
svm_base_Fit_confM <- tibble(
  metric = names(svm_base_Fit_confM$byClass),
  value = svm_base_Fit_confM$byClass
)

########## SVM Base Test Error Rate ###########
svm_base_predicted <- predict(SVM_base_fit, newdata = testset_transformed, type ='raw')

svm_test_fit_accuracy = mean(testset$credit == svm_base_predicted)
caret::confusionMatrix(data = (svm_base_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_base_Test_confM
print(svm_base_Test_confM)
svm_base_Test_confM <- tibble(
  metric = names(svm_base_Test_confM$byClass),
  value = svm_base_Test_confM$byClass
)

#############################################################
############# Tune the Base SVM Linear Model ##############
set.seed(232)
# Tune with different cost values over 10 fold cv
svm_tune_fit = tune(svm,
                    credit~.,
                    data = trainset_transformed,
                    kernel = "linear",
                    type = 'C-classification',
                    class.weights = c('0' = 1, '1' = 4.4),
                    ranges = list(cost = c(0.001, 0.01, 0.1, 1, .5, 2, 3, 5, 10, 50, 100)))
svm_tune_fit$best.performance 
svm_tune_fit$best.model
svm_tune_fit %>% summary()
######## SVM Tuned Train Error Rate ############
svm_tuned_fitted <- predict(svm_tune_fit$best.model, newdata = trainset_transformed, type ='raw')

svm_tuned_fit_accuracy = mean(trainset$credit == svm_tuned_fitted)
caret::confusionMatrix(data = (svm_tuned_fitted),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_Tuned_Fit_confM
print(svm_Tuned_Fit_confM)
svm_Tuned_Fit_confM <- tibble(
  metric = names(svm_Tuned_Fit_confM$byClass),
  value = svm_Tuned_Fit_confM$byClass
)

########## SVM Tuned Test Error Rate ###########
svm_tuned_predicted <- predict(svm_tune_fit$best.model, newdata = testset_transformed, type ='raw')

svm_tuned_test_accuracy = mean(testset$credit == svm_tuned_predicted)
caret::confusionMatrix(data = (svm_tuned_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_Tuned_Test_confM
print(svm_Tuned_Test_confM)
svm_Tuned_Test_confM <- tibble(
  metric = names(svm_Tuned_Test_confM$byClass),
  value = svm_Tuned_Test_confM$byClass
)
#################################
set.seed(232)
svm_tune_fit_radial <- tune(svm,
                            credit~.,
                            data = trainset_transformed,
                            kernel = "radial",
                           # type = 'C-classification',
                            #class.weights = c('0' = 1, '1' = 2.4),
                           class.weights = c('0' = 1, '1' = 4.4),
                           ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10 )))
svm_tune_fit_radial$best.performance
svm_tune_fit_radial$best.parameters
######## SVM Tuned (Radial) Train Error Rate ############
svm_tuned_fitted_radial <- predict(svm_tune_fit_radial$best.model, newdata = trainset_transformed, type ='raw')

svm_tuned_fit_accuracy_radial = mean(trainset$credit == svm_tuned_fitted_radial)
caret::confusionMatrix(data = (svm_tuned_fitted_radial),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_Tuned_Fit_Radial_confM
print(svm_Tuned_Fit_Radial_confM)
svm_Tuned_Fit_Radial_confM <- tibble(
  metric = names(svm_Tuned_Fit_Radial_confM$byClass),
  value = svm_Tuned_Fit_Radial_confM$byClass
)

########## SVM Tuned (Radial) Test Error Rate ###########
svm_tuned_predicted_radial <- predict(svm_tune_fit_radial$best.model, newdata = testset_transformed, type ='raw')

svm_tuned_test_accuracy_radial = mean(testset$credit == svm_tuned_predicted_radial)
caret::confusionMatrix(data = (svm_tuned_predicted_radial),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_Tuned_Test_Radial_confM
print(svm_Tuned_Test_Radial_confM)
svm_Tuned_Test_Radial_confM <- tibble(
  metric = names(svm_Tuned_Test_Radial_confM$byClass),
  value = svm_Tuned_Test_Radial_confM$byClass
)

##################################### 
##################################### 
#################################
set.seed(232)
svm_tune_fit_poly <- tune(svm,
                            credit~.,
                            data = trainset_transformed,
                            kernel = "polynomial",
                             type = 'C-classification',
                          class.weights = c('0' = 1, '1' = 4.4),
                          ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10,15,20,25,30 )),
                              d =c(2, 3, 4)
                          )
svm_tune_fit_poly %>% summary()
svm_tune_fit_poly$best.parameters
svm_tune_fit_poly$best.model %>% summary()
######## SVM Tuned (poly) Train Error Rate ############
svm_tuned_fitted_poly <- predict(svm_tune_fit_poly$best.model, newdata = trainset_transformed, type ='raw')
plot(svm_tune_fit_poly)
svm_tuned_fit_accuracy_poly = mean(trainset$credit == svm_tuned_fitted_poly)
caret::confusionMatrix(data = (svm_tuned_fitted_poly),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
) -> svm_Tuned_Fit_Poly_confM
print(svm_Tuned_Fit_Poly_confM)
svm_Tuned_Fit_Poly_confM <- tibble(
  metric = names(svm_Tuned_Fit_Poly_confM$byClass),
  value = svm_Tuned_Fit_Poly_confM$byClass
)

########## SVM Tuned (poly) Test Error Rate ###########
svm_tuned_predicted_poly <- predict(svm_tune_fit_poly$best.model, newdata = testset_transformed, type ='class')

svm_tuned_test_accuracy_poly = mean(testset$credit == svm_tuned_predicted_poly)
caret::confusionMatrix(data = (svm_tuned_predicted_poly),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)-> svm_Tuned_Test_Poly_confM
print(svm_Tuned_Test_Poly_confM)
svm_Tuned_Test_Poly_confM <- tibble(
  metric = names(svm_Tuned_Test_Poly_confM$byClass),
  value = svm_Tuned_Test_Poly_confM$byClass
)

plot(svm_tune_fit_poly)

svm_results <- data.frame(
  Model_type = c('SVM Linear','SVM Linear',
                 'SVM Linear Cross-Validated', 'SVM Linear Cross-Validated',
                 'SVM Radial Cross-Validated','SVM Radial Cross-Validated',
                 'SVM Polynomial Cross-Validated','SVM Polynomial Cross-Validated'),
  Result_Set = rep(c('Train', 'Test'),4),
  Accuracy = c(svm_base_fit_accuracy,
               svm_test_fit_accuracy,
               svm_tuned_fit_accuracy,
               svm_tuned_test_accuracy,
               svm_tuned_fit_accuracy_radial,
               svm_tuned_test_accuracy_radial,
               svm_tuned_fit_accuracy_poly,
               svm_tuned_test_accuracy_poly
               ),
  Sensitivity = c(
    as.numeric(svm_base_Fit_confM[svm_base_Fit_confM$metric == "Sensitivity", 'value']$value),
    as.numeric(svm_base_Test_confM[svm_base_Test_confM$metric == "Sensitivity", 'value']$value),
    
    as.numeric(svm_Tuned_Fit_confM[svm_Tuned_Fit_confM$metric == "Sensitivity", 'value']$value),   
    as.numeric(svm_Tuned_Test_confM[svm_Tuned_Test_confM$metric == "Sensitivity", 'value']$value),
    
    as.numeric(svm_Tuned_Fit_Radial_confM[svm_Tuned_Fit_Radial_confM$metric == "Sensitivity", 'value']$value),
   as.numeric(svm_Tuned_Test_Radial_confM[svm_Tuned_Test_Radial_confM$metric == "Sensitivity", 'value']$value),
   
   as.numeric(svm_Tuned_Fit_Poly_confM[svm_Tuned_Fit_Poly_confM$metric == "Sensitivity", 'value']$value),
  as.numeric(svm_Tuned_Test_Poly_confM[svm_Tuned_Test_Poly_confM$metric == "Sensitivity", 'value']$value)
  ), 
  Specificity = c(
    as.numeric(svm_base_Fit_confM[svm_base_Fit_confM$metric == "Specificity", 'value']$value),
    as.numeric(svm_base_Test_confM[svm_base_Test_confM$metric == "Specificity", 'value']$value),
    
    as.numeric(svm_Tuned_Fit_confM[svm_Tuned_Fit_confM$metric == "Specificity", 'value']$value),   
    as.numeric(svm_Tuned_Test_confM[svm_Tuned_Test_confM$metric == "Specificity", 'value']$value),
    
    as.numeric(svm_Tuned_Fit_Radial_confM[svm_Tuned_Fit_Radial_confM$metric == "Specificity", 'value']$value),
    as.numeric(svm_Tuned_Test_Radial_confM[svm_Tuned_Test_Radial_confM$metric == "Specificity", 'value']$value),
    
    as.numeric(svm_Tuned_Fit_Poly_confM[svm_Tuned_Fit_Poly_confM$metric == "Specificity", 'value']$value),
    as.numeric(svm_Tuned_Test_Poly_confM[svm_Tuned_Test_Poly_confM$metric == "Specificity", 'value']$value)
  )
  )
svm_results %>% 
  arrange(Result_Set, Accuracy, Sensitivity)
##################################### 

# Plot Learning Curve
lcurve(tdata = trainset_transformed, nparts = 10, 
       modform = credit ~ .,
       funName =  svm,
       #wts = mod_wts,
       y = 'credit',
       metric = "accuracy",
       tg = data.frame(cost = 30)
) -> learning_curve_svmLinear
learning_curve_svmLinear %>% 
  gather(key = 'Measure', value = metric,grep(x=names(.), pattern = "accuracy",ignore.case = T, value = T))

lcplot(
  lcurvedf = learning_curve_svmLinear,
  metric = 'Accuracy',
  modname = 'SVM - Linear',
  nparts = 12
)

lcTimePlot(
  lcurvedf = learning_curve_svmLinear,
  metric = 'Accuracy',
  modname = 'SVM - Linear',
  nparts = 12,
 ylim = c(0.001,0.5)
)

####################### KNN ################################
####################### 
################## ##############
##################


#
set.seed(232)
knn_base_fit <- train(
  credit ~ .,
  data = trainset_transformed,
  method = 'kknn', 
  trControl = trainControl(method ='none'),
  weights = mod_wts, tuneLength = 1 ,
  tuneGrid = data.frame(
    kmax = 1,
    distance = 1,
    kernel = 'optimal' # number of split per tree
  ),
 # twoClassSummary = T,
  #trControl = trainControl(method = "none") # no Cross validation)
  )
knn_base_fit
######## knn Base Train Error Rate ############
knn_base_fitted <-
  predict(knn_base_fit, newdata = trainset_transformed, type = 'raw')

knn_base_fit_accuracy = mean(trainset$credit == knn_base_fitted)
caret::confusionMatrix(data = knn_base_fitted,
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## knn Base Test Error Rate ###########
knn_base_predicted <- predict(knn_base_fit, newdata = testset_transformed, type ='raw')

knn_test_fit_accuracy = mean(testset$credit == knn_base_predicted)
caret::confusionMatrix(data = (knn_base_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)
#############################################################
############# Tune the Base KNN Model ##############
knn_tgrid <-
  expand.grid(
    kmax = 1:30,
    distance = 2, # 
    kernel = 'optimal'
  )
 
set.seed(232)
knn_tune_fit <- train(
  credit ~ .,
  data = trainset_transformed,
  method = "kknn",
  weights = mod_wts, 
  tuneGrid = knn_tgrid,
  trControl = trainctr2
)

knn_tune_fit$bestTune

plot(knn_tune_fit)
 
######## KNN Tuned Train Error Rate ############
knn_tuned_fitted <- predict(knn_tune_fit, newdata = trainset_transformed, type ='raw')

knn_tuned_fit_accuracy = mean(trainset$credit == knn_tuned_fitted)
caret::confusionMatrix(data = (knn_tuned_fitted),
                       reference= trainset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

########## KNN Tuned Test Error Rate ###########
knn_tuned_predicted <- predict(knn_tune_fit, newdata = testset_transformed, type ='raw')

knn_tuned_test_accuracy = mean(testset$credit == knn_tuned_predicted)
caret::confusionMatrix(data = (knn_tuned_predicted),
                       reference= testset$credit,
                       positive = "1",
                       dnn =c("Predictions", "Actuals"),
)

##################################### 
# Plot Learning Curve
lcurve(tdata = trainset_transformed, 
       nparts = 9, 
       modform = credit ~ .,
       modtype =  "kknn",
       wts = mod_wts,
       y = 'credit',
       metric = "accuracy",
       tg = data.frame(kmax = 15,
                       distance = 2,
                       kernel = 'optimal')
) -> learning_curve_knn

learning_curve_knn %>% 
  gather(key = 'Measure', value = metric,grep(x=names(.), pattern = "accuracy",ignore.case = T, value = T))

lcplot(
  lcurvedf = learning_curve_knn,
  metric = 'Accuracy',
  modname = 'KKNN',
  nparts = 9,
  ylim =  c(0.3,1)
)

lcTimePlot(
  lcurvedf = learning_curve_knn,
  metric = 'Accuracy',
  modname = 'KNN',
  nparts = 9,
  ylim = c(0.1,2)
)
