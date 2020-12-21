## This is a R script for PLS-DA method.

## Written and debugged by Arthur Zhu on 22/12/2020


### Part 1: Import packages and modules ###
## ======================================================================= ##
library('readxl') # for importing data from Excel files
library('pheatmap')
library('tidyverse')
library('mixOmics')

### Part 2: Data preparation ###
## ======================================================================= ##
all_data <- read_excel('Data.xlsx', sheet = 'main')
peaks <- all_data %>% dplyr::select(15:79)
peaks <- normalize(peaks)
peaks <- data.frame(sapply(peaks, function(x){x*1000}))
pls_dataframe <- cbind(all_data$Sample, all_data$Grade, peaks)
colnames(pls_dataframe)[1] <- 'Sample'
colnames(pls_dataframe)[2] <- 'Grade'
row.names(pls_dataframe) <- pls_dataframe$Sample
pls_dataframe$Sample <- NULL

### Part 3: PLS-DA method ###
## ======================================================================= ##
plsresult <- mixOmics::plsda(peaks, all_data$Grade, ncomp=5)
# ---------------------------------------------------------------------------
## Calculate the ROC AUC for PLS-DA in cross validation
set.seed(888)
perf.plsda <- perf(plsresult, validation="Mfold", folds=10, 
                   progressBar=FALSE, auc=TRUE, nrepeat=10)
