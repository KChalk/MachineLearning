library(tidyverse)
library(MASS)

housing <- Boston


LR_solve <- function(X,y){
  model<- lm(y~X[,1]+
               X[,2]+
               X[,3]+
               X[,4]+
               X[,5]+
               X[,6]+
               X[,7]+
               X[,8]+
               X[,9]+
               X[,10]+
               X[,11]+
               X[,12]+
               X[,13]
  )  
  #    w <- ginv(t(X)%*%t(t(X)))# %*%
  #    t(X) *y
  w <- summary(model)$coefficients[,'Estimate']
  return(w)
}


LR_predict <- function(X,w){
  X <- as.matrix(add_column(X, intercept=1, .before=1))
  
  num_obs <-length(X[,1])
  num_feature <- length(X[1,])
  predicts <- integer()
  
  for (i in 1:num_obs){
    temp <- 0
    for (j in 1:num_feature){
      temp <- temp+(X[i,j]* w[j])
    }
    
    predicts[i] <- temp
  }
  return(predicts)
}


MSE_calc <- function(predictions, reality){
  mse <- 0
  totalerror <- 0
  
  num_predictions <-length(predictions) 
  for( i in 1:num_predictions){
    dif <- predictions[i]-reality[i]
    sqdif <- dif^2
    totalerror <- totalerror+sqdif
  }
  
  mse <- totalerror/num_predictions
  return(mse)
}

normalize <- function(data){
  min <- min(data)
  range <- max(data)-min
  n <- length(data)
  
  normaldata <- integer(n)
  
  for(i in 1:n){
    point <- data[i]
    if(point==0){
      normaldata[i] <- 0
    }else{
      normaldata[i] <- (point-min)/range  
    }
  }
  
  return(normaldata)
}  

gd_online <- function(X,y,stepnum){
  
  XwithInt <- add_column(X, intercept=1, before=1)
  d <- length(XwithInt[1,])
  n <- length(y)
  
  #initialize weights
  gd_weights <-integer(d)
  
  #initialize empty df
  errors <- data_frame(iter=1:stepnum,trainerror=0,testerror=0)
  
  for (t in 1:stepnum){
    #select a datapoint      
    i <- t%%n
    if(i==0){i <- n}
    #set a learning rate
    a <- .1 #2/t
    
    #Update Weight vector
    #helpful precalculation of error for this point y-f(x_i,w)
    point_error <-y[i]-LR_predict(X[i,],gd_weights)
    
    updateval <- integer()
    for (j in 1:d){
      updateval[j] <- XwithInt[i,j]* (a*point_error)
    }
    
    gd_weights <- gd_weights+updateval
    
  train_predictions <- LR_predict(housing_train[,1:13],gd_weights)
  test_predictions <- LR_predict(housing_test[,1:13],gd_weights)
  
  trainerror <- MSE_calc(train_predictions,housing_train[,14])
  testerror <- MSE_calc(test_predictions,housing_test[,14])
  
  errors[t,'trainerror'] <- trainerror
  errors[t,'testerror'] <- testerror
  }
  returnvals <- list(errors, gd_weights)
  return(returnvals)
}


dot_prod_mbyv <- function(A,b){
  acoln <- length(A[1,])
  arown <- length(A[,1])

  results <- integer()
  for (i in 1:arown){
    temp <- 0
    for (j in 1:acoln){
      temp <- temp+(A[i,j]* b[j,i])
    }
    
    results[i] <- temp
  }
  return(results)
  
}
#######################################################################

housing_train <- housing[1:433,]
housing_test <- housing[434:506,]

lr_weights <- LR_solve(data.frame(housing_train[,1:13]),housing_train[,14])

lr_train_predictions <- LR_predict(housing_train[,1:13],lr_weights)
lr_test_predictions <- LR_predict(housing_test[,1:13],lr_weights)

trainerror <- MSE_calc(lr_train_predictions,housing_train[,14])
testerror <- MSE_calc(lr_test_predictions,housing_test[,14])

X <- housing_train[,1:13]
y <- housing_train[,14]

d <- length(X[1,])
n <- length(y)

#normalize data
normalX <-  data.frame(matrix(ncol = d, nrow = n))
for(col in 1:d){
  normalX[,col] <- normalize(X[,col])
}


stepnum <- 500

XwithInt <- add_column(normalX, intercept=1, .before=1)

#initialize weights
  gd_weights <-integer(d+1)
  
#initialize empty df
  errors <- data_frame(iter=1:stepnum,trainerror=0,testerror=0)
  i=2
  for (t in 1:stepnum){
    #select a datapoint      
    i <- t%%n
    if(i==0){i <- n}
    #set a learning rate
    a <- .1 #2/t
    
    #Update Weight vector
    #helpful precalculation of error for this point y-f(x_i,w)
  
    
    point_error <-y[i]-LR_predict(normalX[i,],gd_weights)
    
    updateval <- integer()
    for (j in 1:(d+1)){
      updateval[j] <- XwithInt[i,j]* a*point_error
    }
    
    gd_weights <- gd_weights+updateval
    
    #old mistaktes
    #update intercept
    #    gd_weights[1] <- gd_weights[1]+(a*point_error)
    #    
    #update other weights
    #    for (j in 2:d+1){
    #        gd_weights[j] <- gd_weights[j]+(a*point_error*X[i,j-1])
  }
  



gd_output <- gd_online(normalX,y,500)

gd_errors <- gd_output[[1]] %>% 
  gather(set, error, trainerror:testerror) %>% 
  filter(iter<10|iter%%50==0)
gd_weight <- gd_output[[2]]

errorplot <- ggplot(gd_errors, aes(x=iter, y=error, color=set))+
  geom_point()

gd_test_predictions <- LR_predict(housing_test[,1:13],gd_weight)

gd_train_predictions <- LR_predict(housing_train[,1:13],gd_weight)




###############################################2
class_train <- read_csv('hw2Data/classification_train.txt', col_names = FALSE, col_types = 'ddd')
class_test <- read_csv('hw2Data/classification_test.txt', col_names = FALSE, col_types = 'ddc')

logr_predict<- function(X,logr_weights){
  z <- X %*% t(logr_weights)
  
  prob <- 1/(1+exp(-t(logr_weights)*X))
  
  for(i in 1:length(prob)){
    if(prob[i] > .5){
      class[i] <- 1
    }
    else{class[i] <- 0}
  }
  return(class)
}


GLR <- function(X,y,stepnum){
  d <- length(X[1,])
  n <- length(y)
  
  XwithInt <- add_column(X, intercept=1, before=1)
  
  X <- as.matrix(X)
  y <- as.matrix(y)
  #initialize weights
  logr_weights <-rep(1,d+1)
  
  #initialize empty df
  errors <- data_frame(iter=1:stepnum,trainerror=0,testerror=0)
  
  for (k in 1:stepnum){
    
    #select a datapoint      
    i <- k%%n
    
    if(i==0){i <- n}
    
    guessvect <- logr_predict(X[i],logr_weights)
    guess <- guessvect[1]
    y[i]
    incorrect <- y[i]-guess
    
    if (incorrect!=0){
      #set a learning rate
      a <- 2/k
      #Update Weight vector
      logr_weights <- logr_weights+(a*XwithInt[i])
    }
  }
  return(logr_weights)
}

trainX<- class_train[,1:2]
trainy <- class_train[,3]
testX <- class_test[,1:2]
testy <- class_test[,3]

X <- trainX
y <- trainy
stepnum <- 500

GLR()