##########################################################################################################################
                                  #      #######################################    #
                                  #                ##################               #
                                  #     AUTHOR(s) - NANDISH PATEL(45539510)         #
                                  #                 RAJ CHOKSI(45533865)            #
                                  #                 TARUN KUMAR(45534632)           #
                                  #                 SUKHJEET KAUR(45665761)         #
                                  #                                                 #
                                  #     [STAT 828 DATA MINING]                      #
                                  #     FINAL PROJECT                               #
                                  #                ##################               #
                                  #      #######################################    #
##########################################################################################################################




##########################################################################################################################
                                  ###################################################
                                  
                                  #           FUNCTIONS CREATED AND USED            #
                                  
                                  ###################################################
##########################################################################################################################


###################### LOADING THE LIBRARIES ###################

load_Libraries <- function(){
  library(ggplot2)
  library(plyr)
  library(dplyr)
  library(tidyr)
  library(e1071)
  library(ROCR)
}

###################### OUTLIERS #######################

is_outlier <- function(var) {
  return(var < quantile(var, 0.25) - 1.5 * IQR(var) | var > quantile(var, 0.75) + 1.5 * IQR(var))
}

##########################################################################################################################




########################################################################################################################
                                  ###################################################
                                  
                                  #                       START                     #
                                  
                                  ###################################################
########################################################################################################################


#########################    LOADING LIBRARIES AND DATA    ########################

load_Libraries()

bank_data <- read.csv("bank-full.csv", header = T, sep = ";")

head(bank_data, n = 10)

summary(bank_data)

str(bank_data)

############################  EXPLORING THE DATA #############################

######### Renaming the variables ##########

names(bank_data)[names(bank_data) == "y"] <- "term_deposit_subscribed"

names(bank_data)[names(bank_data) == "default"] <- "default_credit"

####################### CLIENT DATA ##########################
##############################################################

################## Age ###################

summary(bank_data$age)

outliers <- bank_data$age[is_outlier(bank_data$age)]
length(outliers)

temp_age <- NULL
temp_age[bank_data$age >= 18 & bank_data$age <= 35] <- "Youth"
temp_age[bank_data$age >= 36 & bank_data$age <= 59] <- "Workforce"
temp_age[bank_data$age >= 60] <- "Retired"

bank_data$age <- as.factor(temp_age)

################## JOB  ###################

summary(bank_data$job)

################## MARITAL ###################

summary(bank_data$marital)

################## EDUCATION ###################

summary(bank_data$education)

################## DEFAULT CREDIT ###################

summary(bank_data$default_credit)

#################### BALANCE ######################

summary(bank_data$balance)

outliers <- bank_data$balance[is_outlier(bank_data$balance)]
length(outliers)

#################### LOAN ######################

summary(bank_data$loan)

##################### CONTACT RELATED DATA #######################
##################################################################

#################### CONTACT ######################

summary(bank_data$contact)

###################### DAY ########################

summary(bank_data$day)
#outliers <- bank_data$day[is_outlier(bank_data$day)]
#length(outliers)

###################### MONTH ########################

summary(bank_data$month)

###################### DURATION ########################

summary(bank_data$duration)

outliers <- bank_data$duration[is_outlier(bank_data$duration)]
length(outliers)

#################### CAMPAIGN RELATED DATA #######################
##################################################################

###################### CAMPAIGN ########################

summary(bank_data$campaign)

outliers <- bank_data$campaign[is_outlier(bank_data$campaign)]
length(outliers)

###################### PDAYS ########################

summary(bank_data$pdays)

outliers <- bank_data$pdays[is_outlier(bank_data$pdays)]
length(outliers)

###################### PREVIOUS ########################

summary(bank_data$previous)

outliers <- bank_data$previous[is_outlier(bank_data$previous)]
length(outliers)

###################### POUTCOME ########################

summary(bank_data$poutcome)

temp_poutcome <- as.character(bank_data$poutcome)

temp_poutcome[temp_poutcome == "other"] <- "unknown"

bank_data$poutcome <- as.factor(temp_poutcome)

######################    DATA PREPARATION   ########################

set.seed(100)

sample_bank_data <- bank_data %>% sample_n(round(nrow(bank_data)/2), replace = FALSE)

bank_data_notscaled <- sample_bank_data

bank_data_scaled <- sample_bank_data

bank_data_scaled$balance <- scale(bank_data_scaled$balance)
bank_data_scaled$duration <- scale(bank_data_scaled$duration)
bank_data_scaled$day <- scale(bank_data_scaled$day)
bank_data_scaled$campaign <- scale(bank_data_scaled$campaign)
bank_data_scaled$pdays <- scale(bank_data_scaled$pdays)
bank_data_scaled$previous <- scale(bank_data_scaled$previous)

summary(bank_data_scaled)

####################    SPLITTING THE DATA FOR MODELLING PURPOSES    ####################

set.seed(101010) #just to minimize the difference between row count in test and evaluation

split_indicator <- sample(3:1, size = nrow(sample_bank_data), prob = c(0.2, 0.2, 0.6), replace = TRUE)

bank_notscaled_train <- bank_data_notscaled[split_indicator == 1, ]
bank_notscaled_eval <- bank_data_notscaled[split_indicator == 2, ]
bank_notscaled_test <- bank_data_notscaled[split_indicator == 3, ]

bank_scaled_train <- bank_data_scaled[split_indicator == 1, ]
bank_scaled_eval <- bank_data_scaled[split_indicator == 2, ]
bank_scaled_test <- bank_data_scaled[split_indicator == 3, ]

#################################    SAVING THE DATA    #################################

write.csv(bank_data_notscaled, "bank_notscaled.csv", row.names = F)

write.csv(bank_notscaled_train, "bank_notscaled_train.csv", row.names = F)
write.csv(bank_notscaled_eval, "bank_notscaled_eval.csv", row.names = F)
write.csv(bank_notscaled_test, "bank_notscaled_test.csv", row.names = F)

write.csv(bank_data_scaled, "bank_scaled.csv", row.names = F)

write.csv(bank_scaled_train, "bank_scaled_train.csv", row.names = F)
write.csv(bank_scaled_eval, "bank_scaled_eval.csv", row.names = F)
write.csv(bank_scaled_test, "bank_scaled_test.csv", row.names = F)

########################################################################################################################
                                    ###################################################
                                    
                                    #                     MODELLING                   #
                                    
                                    ###################################################
########################################################################################################################

bank_notscaled_train <- read.csv("bank_notscaled_train.csv", header = T)

bank_notscaled_eval <- read.csv("bank_notscaled_eval.csv", header = T)
  
bank_notscaled_test <- read.csv("bank_notscaled_test.csv", header = T)

######################### NAIVE BAYES ################################

bank_train_bayes <- naiveBayes(term_deposit_subscribed ~ .,laplace = 2, data = bank_notscaled_train)

bank_train_bayes

summary(bank_train_bayes)

bank_train_bayes$tables

bank_train_bayes$apriori

bank_train_pred <- predict(bank_train_bayes, newdata = bank_notscaled_train, type = "class")

summary(bank_train_pred)

bank_notscaled_train$pred_term_deposit_subscribed_naive <- bank_train_pred

pred_table_train <- table(bank_notscaled_train$term_deposit_subscribed, bank_notscaled_train$pred_term_deposit_subscribed_naive)
rownames(pred_table_train) <- paste("Actual", rownames(pred_table_train), sep = ":")
colnames(pred_table_train) <- paste("Pred", colnames(pred_table_train), sep = ":")
pred_table_train

#Accuracy
((pred_table_train[1,1] + pred_table_train[2,2]) / dim(bank_notscaled_train)[1]) * 100

bank_eval_pred <- predict(bank_train_bayes, newdata = bank_notscaled_eval, type = "class")

summary(bank_eval_pred)

bank_notscaled_eval$pred_term_deposit_subscribed_naive <- bank_eval_pred

pred_table_eval <- table(bank_notscaled_eval$term_deposit_subscribed, bank_notscaled_eval$pred_term_deposit_subscribed_naive)
rownames(pred_table_eval) <- paste("Actual", rownames(pred_table_eval), sep = ":")
colnames(pred_table_eval) <- paste("Pred", colnames(pred_table_eval), sep = ":")
pred_table_eval

#Accuracy
((pred_table_eval[1,1] + pred_table_eval[2,2]) / dim(bank_notscaled_eval)[1]) * 100

bank_test_pred <- predict(bank_train_bayes, newdata = bank_notscaled_test, type = "class")

summary(bank_test_pred)

bank_notscaled_test$pred_term_deposit_subscribed_naive <- bank_test_pred

pred_table_test <- table(bank_notscaled_test$term_deposit_subscribed, bank_notscaled_test$pred_term_deposit_subscribed_naive)
rownames(pred_table_test) <- paste("Actual", rownames(pred_table_test), sep = ":")
colnames(pred_table_test) <- paste("Pred", colnames(pred_table_test), sep = ":")
pred_table_test

#Accuracy
((pred_table_test[1,1] + pred_table_test[2,2]) / dim(bank_notscaled_test)[1]) * 100


pred_yes <- prediction(predict(bank_train_bayes, newdata = bank_notscaled_test[,-17], type = "raw")[,2], 
                   bank_notscaled_test$term_deposit_subscribed) 
perf_yes <- performance(pred_yes,"tnr","fnr")
plot(perf_yes, main="ROC curve", colorize = T)


#####################################################################################################################################
#################################	GRAPHS	###############################
#####################################################################################################################################

sample_bank_data <- read.csv("bank_notscaled.csv", header = T)

ggplot(sample_bank_data, aes(x = reorder(age, age, function(x)-length(x)))) + 
  ggtitle("Age group of the Clients") + xlab("Age group") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14), 
        axis.text.x = element_text(angle = 0),axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Age groups") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = age))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Age groups vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Age groups") +
  geom_bar(aes(fill = age), position = "dodge")

################## JOB  ###################

ggplot(sample_bank_data, aes(x = reorder(job, job, function(x)-length(x)))) + 
  ggtitle("Occupations of the Clients") + xlab("Occupations") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14), 
        axis.text.x = element_text(angle = 90),axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Occupations") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = job))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Occupations vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Occupations") +
  geom_bar(aes(fill = job), position = "dodge")


################## MARITAL ###################

ggplot(sample_bank_data, aes(x = reorder(marital, marital, function(x)-length(x)))) + 
  ggtitle("Marital status of the Clients") + xlab("Marital status") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Marital status") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = marital))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Marital Status vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Marital Status") +
  geom_bar(aes(fill = marital), position = "dodge")

################## EDUCATION ###################

ggplot(sample_bank_data, aes(x = reorder(education, education, function(x)-length(x)))) + 
  ggtitle("Education qualification of the Clients") + xlab("Education qualification") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Education\nQualification") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = education))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Education level vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Education level") +
  geom_bar(aes(fill = education), position = "dodge")


################## DEFAULT CREDIT ###################

ggplot(sample_bank_data, aes(x = reorder(default_credit, default_credit, function(x)-length(x)))) + 
  ggtitle("Default credit of the Clients") + xlab("Default credit") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Default Credit") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = default_credit))


ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Default credit vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Default Credit") +
  geom_bar(aes(fill = default_credit), position = "dodge")


#################### BALANCE ######################

ggplot(sample_bank_data, aes(y = balance)) + 
  ggtitle("Distribution of balance of the Clients") + ylab("Balance of the Clients") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5, size = 18), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = balance)) +
  ggtitle("Distribution of Balance") + xlab("Balance in Euros") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  geom_histogram(bins = 20 , color = "black")


#################### LOAN ######################

ggplot(sample_bank_data, aes(x = reorder(loan, loan, function(x)-length(x)))) + 
  ggtitle("Personal Loan of the Clients") + xlab("Has Personal Loan?") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Has Personal Loan?") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = loan))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Loans vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Loan") +
  geom_bar(aes(fill = loan), position = "dodge")

##################### CONTACT RELATED DATA #######################
##################################################################

#################### CONTACT ######################

ggplot(sample_bank_data, aes(x = reorder(contact, contact, function(x)-length(x)))) + 
  ggtitle("How Clients were contacted?") + xlab("Communication Medium") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Communication Medium") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = contact))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Contact medium vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Communicaltion Medium") +
  geom_bar(aes(fill = contact), position = "dodge")

###################### DAY ########################

ggplot(sample_bank_data, aes(y = day)) + 
  ggtitle("Distribution of Day  of the Contact") + ylab("Day of the month") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = day)) +
  ggtitle("Distribution of Day of the contact") + xlab("Day of the month") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  scale_x_continuous(breaks = 1:31) + 
  geom_histogram(bins = 30, color = "black")

###################### MONTH ########################

ggplot(sample_bank_data, aes(x = reorder(month, month, function(x)-length(x)))) + 
  ggtitle("Which month Clients were contacted?") + xlab("Month") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Month") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = month))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Months vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Month") +
  geom_bar(aes(fill = month), position = "dodge")

###################### DURATION ########################

ggplot(sample_bank_data, aes(y = duration)) + 
  ggtitle("Distribution of Duration of the Contact") + ylab("Duration in Seconds") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5, size = 18), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = duration)) +
  ggtitle("Distribution of Duration of the Contact") + xlab("Duration in Seconds") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  geom_histogram(bins = 30, color = "black")

#################### CAMPAIGN RELATED DATA #######################
##################################################################

###################### CAMPAIGN ########################

ggplot(sample_bank_data, aes(y = campaign)) + 
  ggtitle("Distribution of Number of the Contacts during Campaign") + ylab("Number of Contacts") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5, size = 18), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = campaign)) +
  ggtitle("Distribution of Number of Contacts during Campaign") + xlab("Number of Contacts") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  geom_histogram(bins = 20, color = "black")

###################### PDAYS ########################

ggplot(sample_bank_data, aes(y = pdays)) + 
  ggtitle("Distribution of Number of days passed after client was contacted") + ylab("Number of Days") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5, size = 18), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = pdays)) +
  ggtitle("Distribution of Number of days passed after client was contacted") + xlab("Number of Days") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  geom_histogram(bins = 20, color = "black")

###################### PREVIOUS ########################

ggplot(sample_bank_data, aes(y = previous)) + 
  ggtitle("Distribution of Number of the Contacts before Campaign") + ylab("Number of Contacts") +
  geom_boxplot(outlier.colour="red", outlier.shape = 8,
               outlier.size = 2, notch = FALSE, fill = "lightblue") + 
  theme(axis.text.x = element_text(face = "bold", size = 12), axis.text.y = element_blank(),
        axis.ticks.y = element_blank(), plot.title = element_text(hjust = 0.5, size = 18), 
        axis.title = element_text(size = 14, face = "bold")) + 
  coord_flip()

ggplot(sample_bank_data, aes(x = previous)) +
  ggtitle("Distribution of Number of Contacts before Campaign") + xlab("Number of Contacts") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16,face = "bold")) +
  geom_histogram(bins = 20, color = "black")


###################### POUTCOME ########################

ggplot(sample_bank_data, aes(x = reorder(poutcome, poutcome, function(x)-length(x)))) + 
  ggtitle("Results from Previous Campaign") + xlab("Result") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Result") +
  geom_text(aes(label = scales::percent((..count..)/sum(..count..))), stat = "count", vjust = -0.20,  size = 8) +
  geom_bar(aes(fill = poutcome))

ggplot(sample_bank_data, aes(term_deposit_subscribed, ..count..)) + 
  ggtitle("Poutcome vs Term Deposit Subscribed") + xlab("Term Deposit Subscribed") + ylab("Count") +
  theme(plot.title = element_text(hjust = 0.5, size = 18), axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"), 
        legend.text = element_text(size = 12), legend.position = "top") + 
  labs(fill = "Poutcome") +
  geom_bar(aes(fill = poutcome), position = "dodge")

####### Density distribution on continuous variables ######

sample_bank_data_notscaled %>% 
  select(balance, day, duration, pdays, campaign, previous) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")
