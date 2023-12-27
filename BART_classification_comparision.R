#Run this options first to ensure free space to run bartMachine
options(java.parameters="-Xmx5000m")
install.packages("skimr")
library(skimr)
library(pROC)
library(cvms)
library(mlbench)
library(ggplot2)
library(caret)
library(corrplot)
library(bartMachine)
library(e1071)
library(tidyverse)
library(MASS)
library(dplyr)
library(ggpubr)
theme_set(theme_pubr())

heart <- read.csv("/Users/insertusername/Downloads/heart.csv")


heart[, 'target'] <- as.factor(heart[, 'target'])
heart[, 'thal'] <- as.factor(heart[, 'thal'])
heart[, 'cp'] <- as.factor(heart[, 'cp'])
heart[, 'fbs'] <- as.factor(heart[, 'fbs'])
heart[, 'restecg'] <- as.factor(heart[, 'restecg'])
heart[, 'slope'] <- as.factor(heart[, 'slope'])
heart[, 'ca'] <- as.factor(heart[, 'ca'])
heart[, 'exang'] <- as.factor(heart[, 'exang'])
heart[, 'sex'] <- as.factor(heart[, 'sex'])

as_latex <- FALSE
if (as_latex == TRUE) {
  stargazer::stargazer(skim(heart))
} else {
  skim(heart)
}
# Split the data
set.seed(2718)
row.number = sample(1:nrow(heart), 0.7*nrow(heart))
train = heart[row.number,]
test = heart[-row.number,]
dim(train)
dim(test)

#This BART model was run first with all covariates
bart_machine1 = bartMachine(within(train, rm(target)), train$target,
                            num_trees=150, num_burn_in = 100,
                            num_iterations_after_burn_in = 1000,
                            prob_rule_class = 1, use_missing_data = TRUE,
                            seed = 42)

bart_machine = bartMachine(within(train, rm(target,thal,slope,fbs)), train$target,
                           num_trees=150, num_burn_in = 100,
                           num_iterations_after_burn_in = 1000,
                           prob_rule_class = 1, use_missing_data = TRUE,
                           seed = 42)
# Gives us the % acceptance by MCMC iteration, Tree nodes and leaves, and tree depth
plot_convergence_diagnostics(bart_machine)

# Determines the approximate proportion of trees in which the different 
# variables were used 
investigate_var_importance(bart_machine1, num_replicates_for_avg = 20)
investigate_var_importance(bart_machine, num_replicates_for_avg = 20)


# Create logistic model to compare with our BART model.
# This was run first with all covariates
logit1 = glm(formula = train$target~., data = train, family = 'binomial'(link='probit'))

train_mod <- within(train, rm(fbs,slope,thal))
test_mod <- within(test, rm(fbs,slope,thal))

logit = glm(formula = train_mod$target~., data = train_mod, family = 'binomial'(link='probit'))
# Create Support Vector Machines model
svm_model1 <- svm(train$target ~ ., data = train, 
                  type = "C-classification",
                  kernel = "linear")
svm_model <- svm(train_mod$target ~ ., data = train_mod, 
                 type = "C-classification",
                 kernel = "linear")

# Create QDA model
QDA_model1 <- qda(train$target~., data=train)
QDA_model <- qda(train_mod$target~., data=train_mod)

# Make predictions to compare our models
#-fbs - slope - thal
y_pred_bart <- predict(bart_machine1, within(test, rm(target)), type="prob")
bart_table <- table(Actualvalue = test$target, predictedValue = y_pred_bart<0.5)


y_pred_logit <- predict(logit1, test)
logit_table <- table(Actualvalue = test$target, predictedValue = y_pred_logit>0.5)

y_pred_svm <- predict(svm_model1, test)
svm_table <- table(Actualvalue = test$target, predictedValue = y_pred_svm)

y_pred_QDA <- predict(QDA_model1, test)
QDA_table <- table(Actualvalue = test$target, predictedValue = y_pred_QDA$posterior[,2]>0.5)


error_rate_bart <- (bart_table[1,2] + bart_table[2,1]) /  sum(bart_table) 
error_rate_logit <- (logit_table[1,2] + logit_table[2,1] ) / sum(logit_table)
error_rate_svm <- (svm_table[1,2] + svm_table[2,1] ) / sum(svm_table)
error_rate_QDA <- (QDA_table[1,2] + QDA_table[2,1] ) / sum(QDA_table)

error_rate_bart
error_rate_logit
error_rate_svm
error_rate_QDA

#Determine Confidence intervals
rates <- c(error_rate_bart,
           error_rate_logit,
           error_rate_svm,
           error_rate_QDA)
model_list <- c("BART", "Probit", "SVM", "QDA")


z <- qnorm(1-.05/2)

margin_error <- c()
for(i in 1:length(rates)){
  margin_error[i] <-  z*sqrt(rates[i]*(1-rates[i])/nrow(test)) 
}

margin_error
upper_bounds <- c()
for(i in 1:length(rates)){
  upper_bounds[i] <-  z*sqrt(rates[i]*(1-rates[i])/nrow(test)) 
}

bounds_df <- data.frame(Model = model_list, 
                        Rate = rates, 
                        lower = rates-margin_error, 
                        upper = rates+margin_error)

# ggplot2 plot with 95% confidence intervals
error_intervals <- ggplot(bounds_df, aes(Model, Rate,color='cyl')) +        
  geom_point() +
  ggtitle("95% Confidence Intervals for Error Rates")+
  geom_errorbar(aes(ymin = lower, ymax = upper)) +
  theme(legend.position = "none",
        axis.title.x = element_blank(),
        axis.title.y = element_blank())+
  theme_gray()

error_intervals

# Create confusion matrices
Actual <- factor(c(0, 0, 1, 1))
Predicted <- factor(c(0, 1, 0, 1))
bart_mat <- c(bart_table)
svm_mat <- c(svm_table)
qda_mat <- c(QDA_table)
logit_mat   <- c(logit_table)

df_logit <- data.frame(Actual,Predicted, logit_mat)
df_bart <- data.frame(Actual,Predicted, bart_mat)
df_svm <- data.frame(Actual,Predicted, svm_mat)
df_qda <- data.frame(Actual,Predicted, qda_mat)


probit_plot <- ggplot(data =  df_logit, mapping = aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = logit_mat), colour = "white") +
  ggtitle("Probit Confusion Matrix") +
  geom_text(aes(label = sprintf("%1.0f", logit_mat)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "pink") +
  theme() + theme(legend.position = "Probit Model")

bart_plot <- ggplot(data =  df_bart, mapping = aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = bart_mat), colour = "white") +
  ggtitle("BART Confusion Matrix") +
  geom_text(aes(label = sprintf("%1.0f", bart_mat)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "pink") +
  theme() + theme(legend.position = " ")

svm_plot <- ggplot(data =  df_svm, mapping = aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = svm_mat), colour = "white") +
  ggtitle("SVM Confusion Matrix") +
  geom_text(aes(label = sprintf("%1.0f", svm_mat)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "pink") +
  theme() + theme(legend.position = " ")

qda_plot <- ggplot(data =  df_qda, mapping = aes(x = Actual, y = Predicted)) +
  geom_tile(aes(fill = qda_mat), colour = "white") +
  ggtitle("QDA Confusion Matrix") +
  geom_text(aes(label = sprintf("%1.0f", qda_mat)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "pink") +
  theme() + theme(legend.position = " ")

figure <- ggarrange(bart_plot , probit_plot, svm_plot, 
                    qda_plot,
                    #labels = c("A", "Probit Matrix", "C", "D", "E"),
                    ncol = 2, nrow = 2)
figure

# Run comparistions with missing a data value
test_missing <- test

test_missing[sample(length(test), 59)] <- NA

y_pred_bart <- predict(bart_machine1, within(test_missing, rm(target)), type="prob")
bart_table_missing <- table(Actualvalue = test$target, predictedValue = y_pred_bart<0.5)
error_rate_bart_missing <- (bart_table_missing[1,2] + bart_table_missing[2,1]) /  sum(bart_table) 
error_rate_bart_missing

# Calculate the correlation matrix
corr_matrix <- cor(heart)

# Create the correlation plot
corrplot(corr_matrix, method = "circle")



