##Loading package
getAllPackages = function() {
  install.packages("pacman")
  pacman::p_load(data.table, dplyr, magrittr, rrcovNA,glmnet,MASS, randomForest,rpart)
}

##Read-in data with specified directory and filename
readData = function(filename) {
  getAllPackages()
  data = data.frame(fread(filename, na.strings = c(NA, "NA", NULL), stringsAsFactors = TRUE))
  return(data)
}

##Drop columns if the number of missing values exceeds a percentage threshold
##Utilize method with data.table, which works well with large data sets
dropNaColumn = function(data, threshold = .5) {
  n = nrow(data)
  col.to.drop = which(sapply(data, function(x)
    (sum(is.na(
      x
    )) / n > threshold) |
      (sum(x == "", na.rm = TRUE) / n > threshold)))
  if (length(col.to.drop) == 0)
    return(data)
  return(data.frame(data[, -col.to.drop]))  ##convert to data frame in case only 1 or 0 column left
}

##Simple imputation for missing values: replace missing values with
##column means (numerical variables) or column modes (categorical variables)
getMode = function(col) {
  names(sort(-table(col)))[1]
}

simpleImpute = function(data) {
  for (i in 1:ncol(data)) {
    if (class(data[, i]) %in% c("numeric", "integer")) {
      data[is.na(data[, i]), i] = mean(data[, i], na.rm = TRUE)
    } else {
      data[data[,i]=="", i] = getMode(data[, i])
      data[,i] = factor(data[,i])
    }
  }
  return(data)
}

##Multivariate imputation approach
covImpute = function(data) {
  return(data.frame(impSeqRob(data)[1]))
}

##Decision Tree imputation
treeImpute = function(data,print=TRUE) {
  for (i in 1:ncol(data)){
    mod = rpart(data[!is.na(data[,i]),i] ~.,data=data[!is.na(data[,i]),], method = "anova")
    data[is.na(data[,i]),i] = predict(mod, data[is.na(data[,i]),]) 
    if (print==TRUE)
      cat("Finished tree imputing ",i,"variables.\n")
  }
  return(data)
}

imputeData = function(data,impute.method){
  if (impute.method=="simple")
    return(simpleImpute(data))
  
  if (impute.method=="cov")
    return(covImpute(data))
  
  if (impute.method=="tree")
    return(treeImpute(data))
}

##Splitting training data into training and validation set
##Ratio is percentage of training set of all training data
splitTrain = function(data,
                      type = "single",
                      ratio = .8,
                      kfold = 5) {
  if (!type %in% c("single", "cv"))
    print("Error: method of splitting data does not exist.")
  if (ratio[1] >= 1 || length(ratio) != 1)
    print("Error: train ratio has to be a number less than 1.")
  n = nrow(data)
  set.seed(1)
  if (type == "single") {
    splitId = sample(c(rep(1, floor(ratio * n)), rep(2, (
      n - floor(ratio * n)
    ))))
  } else if (type == "cv") {
    splitId = sample(rep(1:kfold, length = nrow(data)))
  }
  return(splitId)
}

##Fit a model based on name
fitModel = function(train,method) {
  
  ##Using backward stepwise selection
  if (method == "stepwise") {
    full.mod = lm(train[,1] ~.,data = train)
    step.mod = step(full.mod, direction = "backward", trace=1)
    return(step.mod)
  }
  
  if (method == "lasso"){
    set.seed(1)
    train.dat = model.matrix(~. -1,train)
    cv.out=cv.glmnet(train.dat[,-1],train.dat[,1],alpha=1) 
    lasso.1se = glmnet(train.dat[,-1],train.dat[,1],alpha=1,lambda = cv.out$lambda.1se)  
    return(lasso.1se)
  }
  
  if (method =="ridge"){
    set.seed(1)
    train.dat = model.matrix(~. -1,train)
    cv.out=cv.glmnet(train.dat[,-1],train.dat[,1],alpha=0) 
    ridge.1se = glmnet(train.dat[,-1],train.dat[,1],alpha=0,lambda = cv.out$lambda.1se)  
    return(ridge.1se)
  }
  
  if (method =="rf"){
    set.seed(1)
    rf.mod = randomForest(train[,1] ~.,data=train[,-1], importance=TRUE,ntree=ntree, do.trace=TRUE)
    return(rf.mod)
  }
}

predictModel = function(val,fit,method) {
  if (method %in% c("stepwise","rf")){
    predicted = predict(fit, newdata=val)
    return(predicted)
  }
    
  if (method %in% c("ridge","lasso")){
    val.data = model.matrix(~. -1,val)
    predicted = predict(fit,newx=val.data)
    return(predicted)
  }

}


##Select best models with training and validation set
validateModels = function(data,methods = c("lasso", "ridge", "rf"),type="single",ratio = .8,
                          kfold = 5) {
  if (type=="single"){
    splitid = splitTrain(data,ratio = ratio)
    train = data[which(splitid==1),]
    val = data[which(splitid==2),]
    for (i in 1:ncol(data)){
      if(class(data[,i])=="factor"){
        levels(val[,i]) = levels(train[,i])
      }
    }
    err.list = numeric(length(methods))
    ct = 1
    for (i in methods){
      fit = fitModel(train,i)
      predicted = predictModel(val[,-1],fit,i)
      mse = mean((predicted - val[,1])^2)
      err.list[ct] = mse
      ct=ct+1
    }
  }
  
  if (type=="cv"){
    set.seed(1)
    splitid = splitTrain(data,type="cv",kfold=kfold)
    err.list = numeric(length(methods))
    for(i in 1:kfold){
      train = data[which(splitid!=i),]
      val = data[which(splitid==i),]
      
      ##Make levels of categorical variables between train and test set to be same
      ##Important for randomForest to work
      for (j in 1:ncol(data)){
        if(class(data[,j])=="factor"){
          levels(val[,j]) = levels(train[,j])
        }
      }
      
      ct = 1
      for (j in methods){
        fit = fitModel(train,j)
        predicted = predictModel(val[,-1],fit,j)
        mse = mean((predicted - val[,1])^2)
        err.list[ct] = err.list[ct] + mse
        ct=ct+1
      }
    }
    err.list = err.list / kfold
  }
  names(err.list) = methods
  return(err.list)
}

predictTest = function(train.data,test.data, impute.method, model){
  test.data = dropNaColumn(test.data)
  test.data = imputeData(test.data,impute.method = impute.method)
  mod = fitModel(train.data,method = model)
  predicted = predictModel(test.data, mod,method=model)
  write.csv(predicted,"prediction.txt",col.names=FALSE,row.names=FALSE)
  return(predicted)
}

do_all = function(train.filename,test.filename,type="cv",ratio=.8,kfold=5,methods=c("lasso","ridge","rf")){
  print("Read in data.")
  train.data = readData(filename=train.filename)
  test.data = readData(filename=test.filename)
  
  ##Pre-processing training data
  ##Drop columns with high percentage of NA's
  print("Dropping columns with NAs above threshold")
  train.data = dropNaColumn(train.data)
  
  ##Imputing missing values (3 different methods)
  print("Imputing missing values.")
  simple.dat = imputeData(train.data,impute.method="simple")
  cov.dat = imputeData(train.data,impute.method="cov")
  tree.dat = imputeData(train.data,impute.method="tree")
  
  ##Validate models (can use type="single" for split (eg. 80/20) or type="cv" for kfold CV)
  print("Comparing models")
  simple.result=validateModels(simple.dat,methods=methods,type="cv")
  cov.result=validateModels(cov.dat,methods=methods,type="cv")
  tree.result=validateModels(tree.dat,methods=methods,type="cv")
  final.result=rbind(simple.result,cov.result,tree.result)
  rownames(final.result)=c("simple","cov","tree")
  
  ##Picking best methods to predict test set
  print("Selecting best model based on result")
  ind.best=which(final.result==min(final.result),arr.ind=TRUE)
  best.impute=rownames(final.result)[ind.best[1]]
  best.model=colnames(final.result)[ind.best[2]]
  best.train.data=eval(parse(text=paste0(best.impute,".dat")))
  
  ##Getting prediction from best methods
  print("Predicting and output prediction in csv.")
  test.prediction = predictTest(best.train.data,test.data,impute.method = best.impute,model = best.model)
  return(list(best.impute,best.model,final.result,test.prediction))
}

###########################################################################################################################################
##Using above functions to train and predict 
##Read-in data
path="your-directory"
setwd(path)
train.filename="train.txt"
test.filename="test.txt"

##Set global variable number of trees
ntree=500
result=do_all(train.filename,test.filename,type="cv",kfold=5)
Tri Nguyen
Tri Nguyen
"# impute_validate_predict_script" 
