# Author: Artemas Wang

# Loading Required packages
library(tidyverse)
library(data.table)
library(igraph)
library(tidyr)

# Reading in data
best_friends <- fread(paste0("best.friends.csv"), header = T, stringsAsFactors = F, data.table = T)
gets_on_with <- fread(paste0("get.on.with.csv"), header = T, stringsAsFactors = F, data.table = T)

# Data PreProcessing
both_data <- rbind(best_friends, gets_on_with) #542 rows
test <- both_data %>% select(from, to, gender.from) %>% arrange(from) #542 rows

# Turning variables in vector to put into graph() function
test$gender.from <- gsub('male', 1, test$gender.from)
test$gender.from <- gsub('fe1', 0, test$gender.from)
test$gender.from <- as.numeric(test$gender.from)
a <- as.numeric(test$from)
b <- as.numeric(test$to)

art <- graph(c(a,b), directed = F)
simplify(art)
test$color <- ifelse(test$gender.from == 1, "blue", "orange")

plot(art,
     layout = layout_with_fr,
     vertex.color = test$color, 
     vertex.size = 20,
     edge.color = "grey",
     edge.arrow.size = 0.05)





