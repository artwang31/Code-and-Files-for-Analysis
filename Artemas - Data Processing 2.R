---
  title: "Data Wranging 2"
author: "Artemas Wang"
date: "1/27/2021"
output: html_document
---
  ```{r - }
```

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r - lodaing packages}
library(dplyr)
library(data.table)
```

```{r - reading in data and checking out structure}
# Reading in data
original_data <- fread(paste0("FILE FILE FILE.csv"), header = T, stringsAsFactors = F, data.table = T)
```

```{r - cleaning up data}

# variable selection
english <- original_data %>% select(`Student Identifier`, `Student First Name`, `Student Last Name`,                                               `Current Grade`, `Activity Type`, `Screening Window Name`, `School Benchmark Category`, 
                                    `Current SGP Vector`)

# turning qualitative into numerics
english$levels <- ifelse(english$`School Benchmark Category` == 'Urgent Intervention', 1,
                         ifelse(english$`School Benchmark Category` == 'Intervention', 2,
                                ifelse(english$`School Benchmark Category` == 'On Watch', 3, 
                                       ifelse(english$`School Benchmark Category` == 'At/Above Benchmark', 4, 'no score')))) 

# selecting students who took the exam in 
english <- english %>% filter(english$`Screening Window Name` == 'Fall Window' | 
                                english$`Current SGP Vector` == 'FALL_WINTER')

# selecting students who took the exam in the fall
fall <- english %>% filter(english$`Screening Window Name` == 'Fall Window') %>% 
  select(-"Current SGP Vector") %>% 
  distinct() 
fall %>% count(`Current Grade`)

# selecting students who took the exam in the winter
winter <- english %>% filter(english$`Current SGP Vector` == 'FALL_WINTER') %>% 
  select(-"Screening Window Name") %>%
  distinct()
winter %>% count(`Current Grade`)

# merging datasets
english_both <- merge(fall, winter, by.x = "Student Identifier", by.y = "Student Identifier")

# selecting variables
english_both <- english_both %>% select(-`Student First Name.y`,-`Student Last Name.y`,-`Current Grade.y`,
                                        -`Activity Type.y`)
english_both %>% count(`Current Grade.x`)

```

```{r - improvement column and by how much}
english_both$levels.x <- as.numeric(as.character(english_both$levels.x))
english_both$levels.y <- as.numeric(as.character(english_both$levels.y))
english_both$status <- (english_both$levels.y - english_both$levels.x)

write.csv(english_both, file="FILE FILE FILE", row.names = FALSE)
```

```{r - cleaning data for all students and grade levels}
all_englishSTAR <- fread(paste0("FILE FILE FILE.csv"), header = T, stringsAsFactors = F, data.table = T)
```

```{r - cleaning up data}

# variable selection
all_englishSTAR <- all_englishSTAR %>% select(`Student Identifier`, `Student First Name`, `Student Last Name`,                                               `Current Grade`, `Activity Type`, `Screening Window Name`, 
                                              `School Benchmark Category`, `Current SGP Vector`)

# turning qualitative into numerics
all_englishSTAR$levels <- ifelse(all_englishSTAR$`School Benchmark Category` == 'Urgent Intervention', 1,
                                 ifelse(all_englishSTAR$`School Benchmark Category` == 'Intervention', 2,
                                        ifelse(all_englishSTAR$`School Benchmark Category` == 'On Watch', 3, 
                                               ifelse(all_englishSTAR$`School Benchmark Category` == 'At/Above Benchmark', 4, 'no score')))) 

# selecting students who took the exam in 
all_englishSTAR <- all_englishSTAR %>% filter(all_englishSTAR$`Screening Window Name` == 'Fall Window' | 
                                                all_englishSTAR$`Current SGP Vector` == 'FALL_WINTER')

# selecting students who took the exam in the fall
fall_all <- all_englishSTAR %>% filter(all_englishSTAR$`Screening Window Name` == 'Fall Window') %>% 
  select(-"Current SGP Vector") %>% 
  distinct() 
fall_all %>% count(`Current Grade`)

# selecting students who took the exam in the winter
winter_all <- all_englishSTAR %>% filter(all_englishSTAR$`Current SGP Vector` == 'FALL_WINTER') %>% 
  select(-"Screening Window Name") %>%
  distinct()
winter_all %>% count(`Current Grade`)

# merging datasets
english_all <- merge(fall_all, winter_all, by.x = "Student Identifier", by.y = "Student Identifier")

# selecting variables
english_all <- english_all %>% select(-`Student First Name.y`,-`Student Last Name.y`,-`Current Grade.y`,
                                      -`Activity Type.y`)
english_all %>% count(`Current Grade.x`)

```

```{r - improvement column and by how much}
english_all$levels.x <- as.numeric(as.character(english_all$levels.x))
english_all$levels.y <- as.numeric(as.character(english_all$levels.y))
english_all$status <- (english_all$levels.y - english_all$levels.x)

write.csv(english_all, file="FILE FILE FILE.csv", row.names = FALSE)
```


























