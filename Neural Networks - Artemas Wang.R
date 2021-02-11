# Author: Artemas Wang yay

# Loading required packages
library(tidyverse)
library(data.table)
library(tidyr)
library(neuralnet)
library(ISLR)



# Reading in required datasets

attention1 <- fread(paste0("attention1.csv"), 
                       header = T, stringsAsFactors = F, data.table = T)
attention2 <- fread(paste0("attention2.csv"), 
                       header = T, stringsAsFactors = F, data.table = T)
#
#
# changing colnames to relevant variable names
colnames(attention1) <- c("attention1","eyes","face.forward","chin.up","attention")
colnames(attention2) <- c("attention2","eyes","face.forward","chin.up","attention")

net <- neuralnet(attention ~ eyes + face.forward + chin.up, attention1, hidden = 1, threshold = 0.01)
net2 <- neuralnet(attention ~ eyes + face.forward + chin.up, attention2, hidden = 5, threshold = 0.01)
plot(net2)

# Trying alternate method
library(MASS)
index <- sample(1:nrow(attention1), round(0.75*nrow(attention1)))
train <- attention1[index,]
test <- attention1[-index,]
  
net_train <- neuralnet(attention ~ eyes + face.forward + chin.up, train, hidden = 1, threshold = 0.01)
plot(net_train)
net_test <- neuralnet(attention ~ eyes + face.forward + chin.up, test, hidden = 5, threshold = 0.01)
plot(net_test)


