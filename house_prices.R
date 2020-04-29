library(dplyr)
library(mice)
library(glmnet)

train <- read.csv("C:/Users/mhais/Desktop/MMA/867/Individual Assignment 1/House Prices/train.csv")
test <- read.csv("C:/Users/mhais/Desktop/MMA/867/Individual Assignment 1/House Prices/test.csv")

SalePrice <- drop(train$SalePrice) # assigning a variable to add back later

train$SalePrice <- NULL # dropping SalePrice

data <- rbind(train,test)
#Removing genuine NAs as per the data_description file
data$Alley <- ifelse(is.na(data$Alley),"None found",data$Alley)
data$BsmtQual <-ifelse(is.na(data$BsmtQual),"None found",data$BsmtQual) 
data$BsmtCond <-ifelse(is.na(data$BsmtCond),"None found",data$BsmtCond) 
data$BsmtExposure <-ifelse(is.na(data$BsmtExposure),"None found",data$BsmtExposure) 
data$BsmtFinType1 <-ifelse(is.na(data$BsmtFinType1),"None found",data$BsmtFinType1)
data$BsmtFinType2 <-ifelse(is.na(data$BsmtFinType2),"None found",data$BsmtFinType2)
data$FireplaceQu <-ifelse(is.na(data$FireplaceQu),"None found",data$FireplaceQu)
data$GarageType <-ifelse(is.na(data$GarageType),"None found",data$GarageType)
data$GarageFinish <-ifelse(is.na(data$GarageFinish),"None found",data$GarageFinish)
data$GarageQual <-ifelse(is.na(data$GarageQual),"None found",data$GarageQual)
data$GarageCond <-ifelse(is.na(data$GarageCond),"None found",data$GarageCond)
data$PoolQC <-ifelse(is.na(data$PoolQC),"None found",data$PoolQC)
data$Fence <-ifelse(is.na(data$Fence),"None found",data$Fence)
data$MiscFeature <-ifelse(is.na(data$MiscFeature),"None found",data$MiscFeature)

#Assessing missing values
is.na(data)
colSums(is.na(data))
md.pattern(data)
imputed_data <- mice(data, m=5, maxit=30, method='cart')
completed_data <- complete(imputed_data, 1)
md.pattern(completed_data)

train<- subset(completed_data, Id <=1460)
test <- subset (completed_data, Id >=1461)

train$SalePrice <- SalePrice
test[,"SalePrice"] <- NA

data <- rbind(train,test)

#LASSO Regression model
house.prices.data <- data
plot(SalePrice ~ GrLivArea, data = house.prices.data)
#removing outlier
#GrLivArea mean is 1510
mean(house.prices.training$GrLivArea)
house.prices.data$GrLivArea <- ifelse(house.prices.data$GrLivArea > 4000, 1510 ,house.prices.data$GrLivArea)
plot(SalePrice ~ GrLivArea, data = house.prices.data)
house.prices.training <- subset(house.prices.data, Id<=1460)
house.prices.prediction <- subset(house.prices.data, Id>=1461)

house.prices.testing<-subset(house.prices.data, (Id>=1001 & Id<=1460)) #withold 1000 datapoints into a "testing" data
house.prices.training<-subset(house.prices.data, Id<=1000) #redefine the training data

y <- log(house.prices.training$SalePrice)
x <- model.matrix(Id~MSSubClass +	MSZoning +	log(LotFrontage) +	 #build model on train
                    log(LotArea) +	Street + Alley +LotShape + LandContour +	
                    Utilities +	LotConfig +	LandSlope +	Neighborhood + Condition1 +	
                    Condition2 +	BldgType + HouseStyle +	OverallQual +	OverallCond +	
                    YearBuilt +	YearRemodAdd +	RoofStyle +	RoofMatl +	Exterior1st +	
                    Exterior2nd +	MasVnrType +	MasVnrArea +	ExterQual +	ExterCond +	
                    Foundation +	BsmtQual +	BsmtCond +	BsmtExposure +	BsmtFinType1 +	
                    BsmtFinSF1 +	BsmtFinType2 +	BsmtFinSF2 +	BsmtUnfSF +	TotalBsmtSF +	
                    Heating +	HeatingQC +	CentralAir +	Electrical +	X1stFlrSF +	X2ndFlrSF +	
                    LowQualFinSF +	log(GrLivArea)*SaleType*SaleCondition*BedroomAbvGr*FullBath*YrSold*MoSold +	
                    BsmtFullBath +	BsmtHalfBath +	FullBath +	
                    sqrt(GrLivArea)*BldgType*OverallCond *HouseStyle*BedroomAbvGr*FullBath*YrSold*MoSold +
                    HalfBath +	BedroomAbvGr +	KitchenAbvGr +	KitchenQual +	TotRmsAbvGrd + 
                    log(GrLivArea)*BldgType*OverallCond *HouseStyle*BedroomAbvGr*FullBath*YrSold*MoSold +
                    Functional +	Fireplaces +	FireplaceQu +	GarageType +	GarageYrBlt +	
                    GarageFinish +	GarageCars +	GarageArea +	GarageQual +	GarageCond +	
                    PavedDrive +	WoodDeckSF +	OpenPorchSF +	EnclosedPorch +	X3SsnPorch +	
                    ScreenPorch +	PoolArea +	PoolQC +	Fence +	MiscFeature +	MiscVal +	log(GrLivArea)*YearBuilt*Heating +
                    MoSold +	YrSold +	SaleType +	SaleCondition, house.prices.data)[,-1]
x<-cbind(house.prices.data$Id,x)

# split X into testing, trainig/holdout and prediction as before
x.training<-subset(x,x[,1]<=1000)
x.testing<-subset(x, (x[,1]>=1001 & x[,1]<=1460))
x.prediction<-subset(x,x[,1]>=1461)

#LASSO (alpha=1)

lasso.fit<-glmnet(x = x.training, y = y, alpha = 1)
plot(lasso.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = x.training, y = y, alpha = 1) #create cross-validation data
#cross validation will split the data set in to train and test and do it multiple time, 10 times, 10 slightly different testing sets and generate errors.
#the dot is the average error and the bars are the confidence intervals, average plus minus one standard deviation

plot(crossval)
penalty.lasso <- crossval$lambda.min #determine optimal penalty parameter, lambda
log(penalty.lasso) #see where it was on the graph
plot(crossval,xlim=c(-6,-4),ylim=c(0.00,0.005)) # lets zoom-in
lasso.opt.fit <-glmnet(x = x.training, y = y, alpha = 1, lambda = penalty.lasso) #estimate the model with the optimal penalty
coef(lasso.opt.fit) #resultant model coefficients

# predicting the performance on the testing set
lasso.testing <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =x.testing))
mean(abs(lasso.testing-house.prices.testing$SalePrice)/house.prices.testing$SalePrice*100) #calculate and display MAPE

lasso.testing

#ridge (alpha=0)
ridge.fit<-glmnet(x = x.training, y = y, alpha = 0)
plot(ridge.fit, xvar = "lambda")

#selecting the best penalty lambda
crossval <-  cv.glmnet(x = x.training, y = y, alpha = 0)
plot(crossval)
penalty.ridge <- crossval$lambda.min 
log(penalty.ridge) 
ridge.opt.fit <-glmnet(x = x.training, y = y, alpha = 0, lambda = penalty.ridge) #estimate the model with that
coef(ridge.opt.fit)

ridge.testing <- exp(predict(ridge.opt.fit, s = penalty.ridge, newx =x.testing))
mean(abs(ridge.testing-house.prices.testing$SalePrice)/house.prices.testing$SalePrice*100) 

# comparing the performance on the testing set, LASSO is better, so use it for prediction
predicted.prices.log.i.lasso <- exp(predict(lasso.opt.fit, s = penalty.lasso, newx =x.prediction))
predicted.prices.log.i.lasso <- data.frame("SalePrice" = predicted.prices.log.i.lasso)
names(predicted.prices.log.i.lasso)[1] <- "SalePrice" 
head(predicted.prices.log.i.lasso)
predicted.prices.log.i.lasso <- data.frame("Id" = house.prices.prediction$Id, "SalePrice" = predicted.prices.log.i.lasso$SalePrice) 
write.csv(predicted.prices.log.i.lasso, file = "C:/Users/mhais/Desktop/MMA/867/Individual Assignment 1/House Prices/submission_lasso5.csv", row.names=FALSE) # export the predicted prices into a CSV file





