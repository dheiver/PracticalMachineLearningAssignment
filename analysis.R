# Website for more information: http://groupware.les.inf.puc-rio.br/har
# Training dataset from : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# Testing dataset from : https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Download the data if don't exist
if(!file.exists("training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                  "training.csv",
                  method = "curl")
}
if(!file.exists("testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                  "testing.csv",
                  method = "curl")
}

# Load some useful packages
if(!require(data.table)) {
    install.packages("data.table")
    require(data.table)
}
if(!require(dplyr)) {
    install.packages("dplyr")
    require(dplyr)
}
if(!require(caret)) {
    install.packages("caret")
    require(caret)
}

# Load the training and testing datasets
training <- tbl_df(fread("training.csv",na.strings=c('#DIV/0!', '', 'NA')))
testing  <- tbl_df(fread("testing.csv",na.strings=c('#DIV/0!', '', 'NA'))) 

# Now split the training into to as actual testing and validation
set.seed(1234) # Reproducibility
trainingDS <- createDataPartition( y = training$classe,
                                   p = 0.7,
                                   list = FALSE)
actual.training <- training[trainingDS,]
actual.validation <- training[-trainingDS,]

# Now clean-up the variables w/ zero variance
# Be careful, kick out the same variables in both cases
nzv <- nearZeroVar(actual.training)
actual.training <- actual.training[,-nzv]
actual.validation <- actual.validation[,-nzv]

# Remove variables that are mostly NA
mostlyNA <- sapply(actual.training,function(x) mean(is.na(x))) > 0.95
actual.training <- actual.training[,mostlyNA==FALSE]
actual.validation <- actual.validation[,mostlyNA==FALSE]

# At this point we're already down to 59 variables from 160
# See that the first 5 variables are identifiers that are
# not probably useful for prediction so get rid of those
# Dropping the total number of variables to 54 (53 for prediction)
actual.training <- actual.training[,-(1:5)]
actual.validation <- actual.validation[,-(1:5)]

# Now let's build a random forest model
set.seed(1234)
modelRF  <- train( classe ~.,
                   data = actual.training,
                   method = "rf",
                   trControl = trainControl(method="cv",number=3) )

# Now get the prediction in the validation portion and see how well we do
prediction.validation.rf <- predict(modelRF,actual.validation)
conf.matrix.rf <- confusionMatrix(prediction.validation.rf,actual.validation$classe)
print(conf.matrix.rf) # Accuracy = 0.998

# One can also build a generalized boosted model and compare its accuracy
# to random forest model
set.seed(1234)
modelBM <- train( classe ~.,
                  data = actual.training,
                  method = "gbm",
                  trControl = trainControl(method="repeatedcv",number = 5,repeats = 1),
                  verbose = FALSE)

# Now get the prediction in the validation portion and see how well we do
prediction.validation.bm <- predict(modelBM,actual.validation)
conf.matrix.bm <- confusionMatrix(prediction.validation.bm,actual.validation$classe)
print(conf.matrix.bm) # Accuracy = 0.9876
 
# At this point we see RF has marginally better performance (Accuracy : 0.998) 
# than the GBM (Accuracy : 0.9876) so we can go w/ either or ensemble them but that 
# might be an overkill at this point - in any case they yield the same result
prediction.testing.rf <- predict(modelRF,testing)
print(prediction.testing.rf)
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E
prediction.testing.bm <- predict(modelBM,testing)
print(prediction.testing.bm)
# [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E

# We can actually look at a few variables to see which ones are the strongest
# Simply look at the BM output
print(summary(modelBM))

# As can be see the most performant variablest are:
# var     rel.inf
# num_window                     num_window 21.35545512
# roll_belt                       roll_belt 18.27724938
# pitch_forearm               pitch_forearm 10.13583112
# magnet_dumbbell_z       magnet_dumbbell_z  6.72541291
# yaw_belt                         yaw_belt  6.51200291
# magnet_dumbbell_y       magnet_dumbbell_y  5.05649194

# Let's look at a few plots in the training set
qplot(num_window, roll_belt    , data = actual.training, col = classe)
qplot(num_window, pitch_forearm, data = actual.training, col = classe)
qplot(roll_belt , pitch_forearm, data = actual.training, col = classe)
