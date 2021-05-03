#Student Name: Antonio Caruso
#Student Number: 19203608
#Project Title: "A Comparative Analysis of five Machine Learning Algorithms for predicting Supply Chain profitability"
#Submission due: 12/12/2020
#Lecturer: John Bohan


#################################Supply chain Dataset Analysis##############################


#FIRST TASK: Data Pre-processing and Exploratory Analysis

#Step 1: Imported Dataset and first view of the file

Finalproject <- read.csv(file = "DataCoSupplyChainDataset.csv", header = TRUE, sep = ",")

install.packages('DataExplorer')
library(DataExplorer)
plot_str(Finalproject) #visualisation of the 53 variables and 180,519 observations


#Step 2:Factual analysis on variables 
install.packages("Amelia", dependencies = TRUE)
library(Amelia)

lapply(Finalproject, class) #classes checked

missmap(Finalproject, main = "Missing values vs observed") #visualised presenc of 4% of missing values

Finalproject[!complete.cases(Finalproject),]
#NA Variables only in product description and order zip code, therefore removed



#Step 3: Removed irrelevant columns and characters columns for PCA analysis.
library(dplyr)

Finalproject_cleaned <- dplyr::select (Finalproject,-c(Category.Id,Customer.Email,Customer.Fname, Customer.Id,Customer.Lname,
                                                       Customer.Password, Customer.State,  Customer.Street,Customer.Zipcode,
                                                       Department.Id, Latitude, Longitude, Order.Customer.Id,order.date..DateOrders.,
                                                       Order.Id, Order.Item.Cardprod.Id,  Order.State, Order.Zipcode , Product.Card.Id,
                                                       Product.Category.Id, Product.Description,  Product.Image, shipping.date..DateOrders.,
                                                       Benefit.per.order,Product.Price, Delivery.Status, Type, Category.Name, Customer.City,
                                                       Customer.Country, Customer.Segment, Department.Name,   Market,   Order.City, 
                                                       Order.Country, Order.Region, Order.Status,Product.Name, Shipping.Mode))

ncol(Finalproject_cleaned) #14 columns remained 


#Step 4: created a csv file to visualise data in tableau
write.csv(Finalproject_cleaned, 'Finalproject_cleaned.csv')



#SECOND TASK: Feature Engineering with Principal Component Analysis 

install_github("vqv/ggbiplot")

library(ggplot2)
library(devtools)
library(usethis)
library(ggbiplot)
library(plyr)
library(grid)
library(scales)

#the following code removes the variables with zero variance so with no variability
#PCA tried to group things by maximising variance and there is no point in retaining this variable.

df_f <- Finalproject_cleaned[,apply(Finalproject_cleaned, 2, var, na.rm = TRUE) != 0] 
pca = prcomp(df_f,center = T, scale. = T)
summary(pca) # PC1 had the highest proportion of variance with 31% followed by PC2 with 13%

ggbiplot(pca) #Visualisation of the obtained 14 principal components,


####THIRD TASK: Application of the models 

#Step 1 - Conversion of target variable into a factor 
#and into binary classification 0-1 for non profitable and profitable orders

Finalproject_cleaned$Order.Profit.Per.Order <- ifelse(Finalproject_cleaned$Order.Profit.Per.Order < 0,0,1)

Finalproject_cleaned$Order.Profit.Per.Order

Finalproject_cleaned$Order.Profit.Per.Order <- as.factor(Finalproject_cleaned$Order.Profit.Per.Order)

summary(Finalproject_cleaned$Order.Profit.Per.Order) #33,784 orders were not profitable vs 146,735 profitable


#Step 2: Creation of dataframe used for applying the models
LateDelivery <- Finalproject_cleaned$Late_delivery_risk
itemquantity <- Finalproject_cleaned$Order.Item.Quantity
Discount <- Finalproject_cleaned$Order.Item.Discount.Rate
Shipment_Scheduled <- Finalproject_cleaned$Days.for.shipment..scheduled.
Shipment_Real <- Finalproject_cleaned$Days.for.shipping..real.

Profit <- Finalproject_cleaned$Order.Profit.Per.Order #target variable

Models_Data <- data.frame( Profit = Profit, LateDelivery = LateDelivery, itemquantity = itemquantity, 
                           Discount = Discount, Shipment_Scheduled = Shipment_Scheduled,
                           Shipment_Real = Shipment_Real)

head(Models_Data)

###################################


#1ST MODEL:CLASSIFICATION TREE

#Step 1: split data into training 70%, and test 30% for the 6 predictor variables identified with PCA
# profit ratio  was excluded as it biased the results
install.packages("caret")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("lattice")
install.packages("rattle")
install.packages("tibble")
install.packages("bitops")
install.packages("RColorBrewer")

library(caret)
library(rpart)
library(rpart.plot)
library(lattice)
library(rattle) 
library(tibble)
library(bitops)
library(RColorBrewer)


set.seed(457)
Classification_tree_data <- sort(sample(nrow(Models_Data), nrow(Models_Data)*.7))

train <- Models_Data[Classification_tree_data, ]
#
test <- Models_Data[-Classification_tree_data, ]

nrow (train) # the train dataset contains 126363 observations
nrow(test)# the test dataset contains 54156 observations


#Step 2: built the decision tree model on the dependent variable orders profit with the Rpart algorithm

Orders_tree <- rpart(Profit~LateDelivery + itemquantity + Discount + Shipment_Scheduled + Shipment_Real,data = train,method="class",control =rpart.control(minsplit =20, minbucket = 7,cp=0),
                     parms = list(split = "gini"))


#Step 3: displayed the decision tree,20 min split for each value chosen as default value
print(Orders_tree)
summary(Orders_tree)
fancyRpartPlot(Orders_tree)


#Step 4: Built prediction against the test dataset
Pred<- predict(object=Orders_tree,test,type="class")

#checked the accuracy through Confusion Matrix
confusionMatrix(table(test$Profit,Pred))


#FINAL RESULTS 

#True Negative 1
#False Positive: 9,967
#False Negative: 3
#True positive: 44,185

#Accuracy 0.8159
#Kappa 0.000001

############################


#2ND MODEL: RANDOM FOREST 

#Step 1: Set Seed and load required libraries

install.packages("cluster")
install.packages("gclus")
install.packages("adabag")
install.packages("randomForest")
install.packages("partykit")

library(cluster)
library(gclus)
library(adabag)
library(randomForest)
library(partykit)

#Step 2:Split data into test and training

set.seed(8894)

N <- nrow(Models_Data)
indtrain<-sample(1:N,size=0.70*N)
indtrain<-sort(indtrain) 
indtest<-setdiff(1:N,indtrain)


#Step 3: Built Random Forest Classifier

fit.rf <- randomForest(Profit~LateDelivery + itemquantity + Discount + Shipment_Scheduled + Shipment_Real,data=Models_Data,subset=indtrain)
pred.rf <- predict(fit.rf,newdata=Models_Data, subset=indtest, type="class") 

fit.rf #OOB 18.64%

varImpPlot(fit.rf) #Discount is the variable with the highest gini value

#Step 4: Checked Accuracy
confusionMatrix(table(Models_Data$Profit[indtest],pred.rf[indtest])) 

#FINAL RESULTS 

#True Negative 0
#False Positive: 10,225
#False Negative: 0
#True positive: 43,931

#accuracy 0.8112
#kappa 0 


############################

#3RD MODEL- KNN

install.packages("arules")
install.packages("class", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("FNN", dependencies = TRUE)
install.packages("gmodels", dependencies = TRUE)
install.packages("psych", dependencies = TRUE)
install.packages("data.table", dependencies=TRUE)
install.packages("BiocManager", dependecies=TRUE)

library("arules")
library("class")
library("caret")
library("dplyr")
library("e1071")
library("gmodels")
library("psych") 
library("data.table")
library("BiocManager")


#Step 1: Data normalistion 

library(class) # class package carries KNN function
normalize <- function(x) {
  return((x - min(x))/(max(x) - min(x)))
} # creating a normalize function for easy convertion.

Models_Data_convert <- as.data.frame(lapply(Models_Data[, 2:6], normalize))

head(Models_Data_convert)

#Step 2: created the training and test datasets
set.seed(123) 
dat.d <- sample(1:nrow(Models_Data_convert), size = nrow(Models_Data_convert) * 0.7, replace = FALSE)

train.subset <- Models_Data[dat.d, ] # 70% training data 
test.subset <- Models_Data[-dat.d, ] # remaining 30% test data

train.subset_labels <- Models_Data[dat.d, 1] 
test.subset_labels <- Models_Data[-dat.d, 1]

NROW(train.subset_labels) #123,363 observations
NROW(test.subset_labels) # 54156 observations

#Step 3: Trained a model on data by identifying optimum value for K
#squared root of total no of observations (123,363) which was 351.23
#Too many ties in KNN determined the decision of halving the K number in 175 

knn.175 <- knn(train = train.subset, test = test.subset, cl = train.subset_labels, k = 175) 

#Step 4: Evaluated the model performance. Then the accuracy at various levels of K was calculated
ACC.175 <- 100 * sum(test.subset_labels == knn.175)/NROW(test.subset_labels) # For knn = 175
ACC.175 #accuracy for k is 99.9723

#step 5 created a confusion matrix

confusionMatrix(table(knn.175, test.subset_labels)) 


#FINAL RESULTS 

#True negative: 10,126
#False Positive: 0
#False Negative: 15
#True positive: 44,015

#ACCURACY 0.9997
#KAPPA 0.9991

#######################


#4TH MODEL: LOGISTIC REGRESSION

#Step 1 Data Preparation

library(Amelia)

#Step 2: Model Fitting into training and test dataset split at 70% and 30% like the previous models

train_logistic <- Models_Data [1:126363,]
test_logistic <- Models_Data[126364:180519,]

#step 3: used the glm function to fit the regression model of type Logit with the parameter family= binomial

Logistic_model_fit <- glm(Profit~., family = binomial, control=glm.control(maxit=50),data = train_logistic)

summary(Logistic_model_fit) # shipment scheduled has the lowest value


#Step 4: the anova function was run to analyse the table of deviance using Anova and Chisq test

anova(Logistic_model_fit, test_logistic = "Chisq")
#the wide gap between the null deviance and residual deviance showed the model is performing well against a null model with only the intercept.


#Step 5: Now the accuracy and confusion matrix are evaluated 

Prediction <- predict(Logistic_model_fit,newdata=subset(test_logistic,select=c(1,2,3,4,5,6)),type='response')

Final_prediction <- ifelse(Prediction > 0.5, 1, 0) 

u <- union(Final_prediction,test_logistic$Profit)

t <- table( factor(Final_prediction,u),factor(test_logistic$Profit,u))

confusionMatrix(t)

#FINAL RESULTS 

#True Positive 44,086
#False Positive: 10,070
#True Negative: 0
#False Negative: 0

#ACCURACY 0.8141
#KAPPA 0
#########################


#5TH MODEL: NAIVE BAYES
install.packages("rsample")
install.packages("caret")
install.packages("dplyr")
install.packages("klaR")
install.packages("naivebayes")
install.packages("bnclassify")

library(rsample)
library(caret)
library(dplyr)

#Step 2: Convert random variables to factors

NBdata <- Models_Data %>%
  mutate(
    Discount = factor( Discount),
    Shipment_Scheduled = factor(Shipment_Scheduled),
    itemquantity= factor(itemquantity)
  )

#Step 3: Created training (70%) and test (30%) sets for the NaiveBdata data.

set.seed(1456)

split <- initial_split(NBdata, prop = .7, strata = "Profit")
train <- training(split) 

#Step 4: Tabulated the distribution of profitsacross train & test set

table(train$Profit) %>% prop.table() # non profitable 18% and profitable order 81%

#Step 5 : created response and feature data

library(klaR)
library(naivebayes)
library(bnclassify)

Naive_bayes_features <- setdiff(names(train), "Profit")
x <- train[, Naive_bayes_features] 
y <- train$Profit  

#Step 6: setup the 10-fold cross validation procedure
train_Control <- trainControl(
  method = "cv",
  number = 10
)

#Step 7: trained the model and see the results in a confusion matrix
nb.m1 <- train(
  x = x,
  y = y,
  method = "nb",
  trControl = train_Control
)


#Step 8:accuracy and confusion matrix are evaluated 
pred <- predict(nb.m1, newdata = test)

confusionMatrix(pred, test$Profit)

#FINAL RESULTS 

#True Negative: 0
#False Positive: 0
#False Negative: 10,135
#True positive 44,020

#ACCURACY 0.8129
#KAPPA 0

##########################END OF SCRIPT######################



