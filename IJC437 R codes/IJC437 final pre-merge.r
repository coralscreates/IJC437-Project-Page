#step 1: set working directory should have 4 files: (1)official_songs_cleaned.csv, (2) official_is_pop_cleaned.csv (3) acoustic_features.csv (4) artists_cleaned .csv

setwd("~/Desktop/Assignment/cleaned data") # this was MY working directory, make sure to change it to match your setwork
library(tidyverse)
library(ggplot2)

#2. Download main dataset (official_songs_cleaned.csv) that will be the main merger and then the one that will be merged to that 
songs_data<-read_csv("official_songs_cleaned.csv", col_names = TRUE) #change the name of this dataframe
songs_data

song_pop<-read_csv("official_is_pop_cleaned.csv", col_names = TRUE) #this is the is pop file 
song_pop

#3 Now that you have created dataframes for the (1)official_songs_cleaned.csv and (2) official_is_pop_cleaned.csv you will merge them together 
songs_pop_dataset<-songs_data %>%
  left_join(song_pop, by="song_id") #left_join function is used to merge dataframes together based on a same variable in this case: song_id

#4. now i will add acoustic features to the dataset
song_acoustic<-read_tsv("acoustic_features.csv", col_names = TRUE) # the reason for tsv instead of csv= file is tsv

songs_p_a_dataset<-songs_pop_dataset %>%
  left_join(song_acoustic, by="song_id") #here you are adding acoustic features to is_pop and songs based on songid again

#5. So far dataset already consists of fles 1-3 now adding last one which is the artists file
songs_artist<-read_csv("artists_cleaned .csv", col_names=TRUE)

songs_dataset<-songs_p_a_dataset %>%
  left_join(songs_artist, by="artist_id") #this has now joined all files together amazing now have created a dataset with merged files!!

#6.Now  it is time to clean the data by this I need to remove any missing values 
cleaning_genreandrap<- complete.cases (
  songs_dataset$artist_type,
  songs_dataset$main_genre,
  songs_dataset$song_type) #this helps identify which rows are complete and which have missing data

dataset_songs<- songs_dataset[cleaning_genreandrap, ]
dataset_songs<- songs_dataset[
  complete.cases(songs_dataset$artist_type, songs_dataset$main_genre, songs_dataset$song_type, songs_dataset$is_pop),
] #rows were removed using complete case 


#7. Check the dataset so far: how many variables were removed went from [insert amount of rows] to  15635

View(dataset_songs) 

#8. checking if there are any missing values 
#any(is.na) function allows to see if any misisng values still present (FALSE- no NA found TRUE - NA found)
#sum(is.na) function allows to see numerical values if any left
any(is.na(dataset_songs$song_type)) # FALSE
any(is.na(dataset_songs$artist_type)) #FALSE
any(is.na(dataset_songs$main_genre)) #FALSE

sum(is.na(dataset_songs$is_pop)) 
sum(is.na(dataset_songs$song_type))
sum(is.na(dataset_songs$main_genre))

# all had output of 0 meaning all data that was NA is now gone 

#check how many rows have been left over
nrow(dataset_songs) #total row of 15635 left from original 20,405


write.csv(dataset_songs, "dataset_songs.csv", row.names = FALSE) # here i am just downloading the dataset for myself

#9: Exploratory Data Analysis(EDA): Here I am conducting an EDA of data for acoustic features individually (there are 9 features)
# first looking at the internal structure of each individual variable and then looking at the summary(mean, interquaartile range etc) of it
str(dataset_songs$popularity.x)
str(dataset_songs$danceability)
str(dataset_songs$acousticness)
str(dataset_songs$energy)
str(dataset_songs$instrumentalness)
str(dataset_songs$liveness)
str(dataset_songs$loudness)
str(dataset_songs$speechiness)
str(dataset_songs$valence)
str(dataset_songs$tempo)

summary(dataset_songs$popularity.x)
summary(dataset_songs$danceability)
summary(dataset_songs$acousticness)
summary(dataset_songs$energy)
summary(dataset_songs$instrumentalness)
summary(dataset_songs$liveness)
summary(dataset_songs$loudness)
summary(dataset_songs$speechiness)
summary(dataset_songs$valence)
summary(dataset_songs$tempo)

#10. A range of different packages used for EDA: tidyverse, MASS,rgl, corrplot (if package not install use this function install.packages("[insert package name]"))
library(tidyverse)
library(MASS)
library(rgl)

install.packages("corrplot")
library(corrplot)
#11: scatterplots for popularity and each of the nine acoustic features were conducted
# cor.test () function was also used to look at correlation alongside each scatterplot just to make numerical inferences of the relationship

#danceability EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= danceability)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$danceability ) 
# t= 15.164, df=15633, p value <2.2e16 (alt hypothesis: true correlation is not equal to 0) 95% CI: 0.1049227 0.1358182, sample estimates:cor 0.1203996

#acoustiness EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= acousticness)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$acousticness )
# t = -32.093, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0) 95 % CI: -0.2632712 -0.2338587 sample estimates:cor -0.2486223 


#energy EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= energy)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$energy )
#t = 20.758, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0) 95% CI: 0.1484898 0.1789989 sample estimates: cor  0.1637835 

#instrumentalness EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= instrumentalness)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$instrumentalness )
#t = 20.758, df = 15633, p-value < 2.2e-16(alt hypothesis: true correlation is not equal to 0) 95% CI: 0.1484898 0.178998 sample estimates:cor 0.1637835 

#liveness EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= liveness)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$liveness ) 
# t = -11.004, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0) 95% CI:-0.10320750 -0.07209859 sample estimates:cor -0.08767442 


#loudness EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= loudness)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$loudness )
#t = 42.338, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0), 95% CI 0.3065980 0.3347237, sample estimates:cor 0.3207316 


#speechiness EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= speechiness)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$speechiness)
# t = 19.851, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0) 95%CI  0.1414742 0.1720535 sample estimates: cor 0.1568014


#valence EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= valence)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$valence)
#t = -21.729, df = 15633, p-value < 2.2e-16 (alt hypothesis: true correlation is not equal to 0) 95% CI: -0.1863939 -0.1559629 sample estimates:cor -0.1712193 


#tempo EDA
ggplot(data= dataset_songs,
       aes(x=popularity.x, y= tempo)
)+ geom_point(colour="blue")

cor.test(dataset_songs$popularity.x, dataset_songs$tempo )
#t = 1.4631, df = 15633, p-value = 0.1435 (alt hypothesis: true correlation is not equal to 0) 95% CI: -0.003974618  0.027370925 sample estimates:cor 0.01170103 

#12. Now creating a correlation matrix to view all correlations in my graph
songs_cor_matrix <- cor(dataset_songs[, sapply(dataset_songs, is.numeric)]) #creating corrlation dataframe 
corrplot(songs_cor_matrix, method = "color", type = "upper", 
         tl.col = "black", addCoef.col = "black", number.cex = 0.6) #found weak correlation between popularity and acoustic features --> loudness strongest assoication from all features


#13. Creating a histogram to look at the individual distribution of each variable (popularity and each acoustic feature)
ggplot(dataset_songs, aes(x = popularity.x)) +
  geom_histogram(bins = 30) #this is just a basic histogram with no information so let us try one where we add more information and make it a bit more colourful

#popularity with a different colour and density curve help with readability 
#popularity --> negatively skewed
ggplot(dataset_songs, aes(x = popularity.x)) + 
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song popularity",
    y = "Frequency",
    title = "Distribution of song popularity with a Density Curve"
  )

#histograms for each acoustic feature
#danceability --> although looks slighlty normal it does have a negative skewed distribution 
ggplot(dataset_songs, aes(x = danceability)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song danceability",
    y = "Frequency",
    title = "Distribution of song danceability with Density Curve"
  )
#acousticness --> positive skewed distribution 
ggplot(dataset_songs, aes(x = acousticness)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song acousticness",
    y = "Frequency",
    title = "Distribution of song acousticness with Density Curve"
  )
#energy --> negatively skewed distribution
ggplot(dataset_songs, aes(x = energy)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song energy",
    y = "Frequency",
    title = "Distribution of song energy with Density Curve"
  )
#instrumentalness --> positively skewed distrubtion
ggplot(dataset_songs, aes(x = instrumentalness)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song instrumentalness",
    y = "Frequency",
    title = "Distribution of song instrumentalness with Density Curve"
  )
#liveness -> positively skewed distrubtion
ggplot(dataset_songs, aes(x = liveness)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song liveness",
    y = "Frequency",
    title = "Distribution of song liveness with Density Curve"
  )
#loudness--> negatively skewed distribution
ggplot(dataset_songs, aes(x = loudness)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song loudness",
    y = "Frequency",
    title = "Distribution of song loudness with Density Curve"
  )
#speechiness --> positively skewed distrubtion
ggplot(dataset_songs, aes(x = speechiness)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song speechiness",
    y = "Frequency",
    title = "Distribution of song speechiness with Density Curve"
  )
#valence --> negatively skewed distribution
ggplot(dataset_songs, aes(x = valence)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song valence",
    y = "Frequency",
    title = "Distribution of song valence with Density Curve"
  )
#tempo --> slight positively skewed distribution
ggplot(dataset_songs, aes(x = tempo)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song tempo",
    y = "Frequency",
    title = "Distribution of song tempo with a Density Curve"
  )

#14 following EDA decided to transformed the DV which is song popularity 
dataset_songs$pop_transformed<- log(dataset_songs$popularity.x +1) #log() function is used to transform the populairty variable

#15. run EDA on new (pop_transformed) variable
#scatterplot
ggplot(data= dataset_songs,
       aes(x=pop_transformed, y= danceability)
)+ geom_point(colour="blue")

#histogram
ggplot(dataset_songs, aes(x = pop_transformed)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "plum",
                 alpha = 0.6) +
  geom_density(colour = "black", linewidth = 1) +
  labs(
    x = "Song Popularity Transformed",
    y = "Frequency",
    title = "Distribution of Song Popularity Transformed with Density Curve"
  )

#from this can see that data became more skewed when popularity was transformed therefore kept original popualirty variable

#16. Now to prepare data for Multiple Linear Regression analysis the data has to be split into train and test

nrow(dataset_songs) #total row of 15635
set.seed(123) # this is to get the same split each time so that if I would like to rerun code I am able to get same outputs as mine

train_model <- sample(seq_len(nrow(dataset_songs)), 
                      size = 0.7 * nrow(dataset_songs))

songs_train <- dataset_songs[train_model, ]
nrow(songs_train) # 10,944  rows for train
songs_test  <- dataset_songs[-train_model, ]
nrow(songs_test) #4691 rows for test

songs_train[1:10,] # just to see what the dataset row for training sample will look like
summary(songs_train)
full_ac_model<- lm(popularity.x ~ danceability + acousticness +energy +instrumentalness 
                   +liveness +loudness +speechiness +valence + tempo, data= songs_train)
summary(full_ac_model)
coef(full_ac_model) # gives me the coefficient of the model without the additional reporting of summry 
#findings:  Residual standard error: 18.58 on 10934 degrees of freedom Multiple
#R-squared:  0.1954,	Adjusted R-squared:  0.1947 
#F-statistic:   295 on 9 and 10934 DF,  p-value: < 2.2e-16
#Intercept:  59.6095397
#Danceability: 20.9755812
#Acousticness:-12.8480756 
#Energy: -6.7957103
#Inatrumentalness: -3.4688467
#Liveness: -10.5396684        
#Loudness:1.5825857
#Speechiness: 17.0286566  
#Valence: -21.6864091 
#Tempo:0.0156123
#all variables found to be significant
plot(full_ac_model) #regression diagnostics (used to assess model limitations)


#17. now lets move onto prediction of linear regression model using train dataset

songs_test_p<- predict(full_ac_model,
        newdata = songs_test) #here I am predicting the values on the test set
summary(songs_test_p) #check the dataframe

songs_test_pc<- predict(full_ac_model, 
        newdata = songs_test, 
        interval='confidence') #here I am predicting with the confidence intervals included this time, this helsp assess the model uncertainity
summary(songs_test_pc) #check the dataframe

songs_test_new <- songs_test #made new data so that orginal is left unaltered

songs_test_new$predicted<- predict(full_ac_model, newdata=songs_test_new)#adding the predictions into the dataset so that it has a new column called predicted

songs_test_new$residuals<- songs_test_new$predicted- songs_test$popularity.x #this calculates the residuals

head(songs_test_new[, c("popularity.x", "predicted", "residuals")]) #check in here to look at prediction and residuals 

#15. Here we are calulcating different things, Sum of Squared Errors (SSE), Mean Squared Error (MSE) and the Root Mean Square Error(RMSE)
sse_test<- sum(songs_test_new$residuals**2) 
sse_test #1595097
mse_test<- mean(songs_test_new$residuals^2) 
mse_test # 340.0334
rmse_test <- sqrt(mean(songs_test_new$residuals^2))
rmse_test # 18.43999

# predicted values on test set
y_pred <- songs_test_new$predicted
# actual values on test set
y_actual <- songs_test_new$popularity.x

# Calculate R-squared
SSE <- sum((y_actual - y_pred)^2)           # Sum of Squared Errors
SST <- sum((y_actual - mean(y_actual))^2)  # Total Sum of Squares

r2_songs_test <- 1 - SSE/SST
r2_songs_test # 0.1979768

#18. Time for the EDA and visualise the predicted vs actual values
library(ggplot2)

ggplot(
  data = songs_test_new,
  aes(x = popularity.x, y = popularity.x)   # actual values on x and y for baseline
) +
  geom_point(size = 3, color = 'blue') +                           # actual values
  geom_point(aes(y = predicted), size = 2, shape = 1) +            # predicted values
  geom_segment(aes(xend = popularity.x, yend = predicted),
               color = 'red', alpha = 0.5) +                       # residual lines
  geom_abline(intercept = 0, slope = 1, color = 'gray', linetype = "dashed") +  # perfect prediction line
  labs(title = "Predicted vs Actual Song Popularity",
       x = "Actual Song Popularity",
       y = "Predicted Song Popularity")

#FIRST REGRESSION DONE which looked at all the acoustic features and song popualirty


# 19: Now moving onto the second regression which looks at the acoustic features that research has highlighted as most signifincant 
#repeating the same process I followed for the first regression model
research_a_model<- lm(popularity.x ~ danceability + valence +
                 loudness +speechiness, data=songs_train)
summary(research_a_model)
coef(research_a_model) # gives me the coefficient of the model without the additional reporting of summry 
#findings: Residual standard error: 18.87 on 10939 degrees of freedom Multiple R-squared:   0.17,	Adjusted R-squared:  0.1697 
#F-statistic:   560 on 4 and 10939 DF,  p-value: < 2.2e-16
#intercept 49.825488
#danceability 27.295314
#valence -23.223649
#loudness 1.682337
#speechiness 15.734348

plot(research_a_model) #regression diagnostics (used to assess model limitations)

#20: Now onto the looking at the test data set
songs_p<- predict(research_a_model,
                        newdata = songs_test)
summary(songs_p)

songs_p_c<- predict(research_a_model, 
                          newdata = songs_test, 
                          interval='confidence')
summary(songs_p_c)

songs_last_test <- songs_test #made new data so that orginal is left unaltered

songs_last_test$predicted<- predict(research_a_model, newdata=songs_last_test)

songs_last_test$residuals<- songs_last_test$predicted- songs_test$popularity.x

head(songs_last_test[, c("popularity.x", "predicted", "residuals")])

sse_last<- sum(songs_last_test$residuals**2)
sse_last #1646402
mse_last<- mean(songs_last_test$residuals^2)
mse_last #350.9704
rmse_last <- sqrt(mean(songs_last_test$residuals^2))
rmse_last #18.7342


# predicted values on test set
y_p<- songs_last_test$predicted
# actual values on test set
y_a<- songs_last_test$popularity.x

# Calculate R-squared
SSE_last <- sum((y_a - y_p)^2)           # Sum of Squared Errors
SST_last <- sum((y_a- mean(y_a))^2)  # Total Sum of Squares

r2_songs_target <- 1 - SSE_last/SST_last #this is the sum of sqaure erros/ total sum of squares
r2_songs_target <- 0.1362985


# REGRESSION MODELS FOR QUESTION 1 is done - will be able to compare the two models

#21. Now moving onto Question 2: does song type (collaboration vs solo) predict popularity?

#22. Running the EDA for this question to look at what data looks like
str(dataset_songs$popularity.x)
str(dataset_songs$song_type)

summary(dataset_songs$popularity.x)
summary(dataset_songs$song_type)

library(ggplot2)
#first a box plot: box plot best to look at categorical and continuous data
ggplot(dataset_songs, aes(x = song_type, y = popularity.x)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Popularity by Song Type",
       x = "Song Type",
       y = "Song Popularity") #box plot best to look at categorical and continuous data

#next a histogram to look at distribution of both (can see collaborations and solo on one chart) 
ggplot(dataset_songs, aes(x = popularity.x, fill = song_type)) +
  geom_histogram(alpha = 0.6, bins = 30, position = "identity") +
  labs(title = "Distribution of Song Popularity by Song Type",
       x = "Song Popularity",
       y = "Count") #however not the best EDA to use so mainly focused on results from box plot


dataset_songs$song_type <- factor(dataset_songs$song_type) #telling are that the song type is a categorical variable 
levels(dataset_songs$song_type) #checking what the different categories are (Solo or Collaboration)

#23. Carrying out the regression model just as in previous steps
s_type_model<-lm(popularity.x~ song_type, data= songs_train)
summary(s_type_model)
coef(s_type_model)
#findings: Residual standard error: 20.7 on 10942 degrees of freedom Multiple R-squared:  0.0005167,	Adjusted R-squared:  0.0004253 
#F-statistic: 5.656 on 1 and 10942 DF,  p-value: 0.01741
#intercept (collaboration): 40.675505 
#Solo: -1.816362
plot(s_type_model)

#24. now running regression on the test dataset
songs_type_p<-predict(s_type_model, 
                      newdata = songs_test)
summary(songs_type_p)

songs_type_c<- predict(s_type_model, 
                       newdata = songs_test,
                       interval = 'confidence')

summary(songs_type_c)

new_songs_test<-songs_test #made new dataset so that the orginal is left unaltered

new_songs_test$predicted<- predict(s_type_model, 
                                   newdata = new_songs_test)
new_songs_test$residuals<-new_songs_test$predicted-new_songs_test$popularity.x

head(new_songs_test[, c("popularity.x", "predicted", "residuals")])


sse_song<- sum(new_songs_test$residuals**2)
sse_song #1990390
mse_song<-mean(new_songs_test$residuals^2)
mse_song #424.2997
rmse_songs <-sqrt(mean(new_songs_test$residuals^2))
rmse_songs #20.59854


#now I am  doing predicted values on the test dataset
y_predicted<- new_songs_test$predicted 

#now I am doing actual values on test dataset
y_actually<-new_songs_test$popularity.x

#Now I will calcualte the R square value 
SSE_songs <- sum((y_actually-y_predicted)^2) #Sum of Squared Errors
SST_songs<-sum((y_actually-mean(y_actually))^2) #Total sum of squares

R2_songs_t<- 1-SSE_songs/SST_songs #this is the sum of square erros/total sum of squares
R2_songs_t #  -0.0007788028


#25. Now moving onto question 3: Does artist type influence song popualirty 

str(dataset_songs$popularity.x)
str(dataset_songs$artist_type)

summary(dataset_songs$popularity.x)
summary(dataset_songs$artist_type)

library(ggplot2)
#first a box plot: box plot best to look at categorical and continuous data
ggplot(dataset_songs, aes(x = artist_type, y = popularity.x)) +
  geom_boxplot(fill = "red") +
  labs(title = "Popularity by Artist Type",
       x = "Artist Type",
       y = "Song Popularity") 

dataset_songs$artist_type <- factor(dataset_songs$artist_type) #telling are that the artist type is a categorical variable 
levels(dataset_songs$artist_type) #checking what the five different categories are (Band, DJ, Duo, rapper and singer)

#26. Now the final regression model for question 3
artist_model<-lm(popularity.x~ artist_type, data= songs_train)
summary(artist_model)
coef(artist_model)
# findings: Residual standard error: 20.45 on 10939 degrees of freedom Multiple R-squared:  0.02496,	Adjusted R-squared:  0.0246  
#F-statistic:    70 on 4 and 10939 DF,  p-value: < 2.2e-16
#Intercept: 37.7981894 (sig)
#DJ: 12.6901826 (sig)
#DUO: 0.3901222
#Rapper: 13.6558075 (sig)
#Singer: 0.6163634 
plot(artist_model) #this produces four diagnostic plots of this regression 

#27. now running regression on the test dataset
final_model<-predict(artist_model, 
                      newdata = songs_test)
summary(final_model)

final_model_c<- predict(artist_model, 
                       newdata = songs_test,
                       interval = 'confidence')

summary(final_model)

final_songs_test<-songs_test #made new dataset so that the orginal is left unaltered

final_songs_test$predicted<- predict(artist_model, 
                                   newdata = final_songs_test)
final_songs_test$residuals<-final_songs_test$predicted-final_songs_test$popularity.x

head(final_songs_test[, c("popularity.x", "predicted", "residuals")])


sse_song<- sum(final_songs_test$residuals**2)
sse_song #1920494
mse_song<-mean(final_songs_test$residuals^2)
mse_song #409.3997
rmse_songs <-sqrt(mean(final_songs_test$residuals^2))
rmse_songs #20.23363


#now I am  doing predicted values on the test data set
y_p_final<- final_songs_test$predicted 

#now I am doing actual values on test data set
y_a_final<-final_songs_test$popularity.x

#Now I will calculate the R square value 
SSE_final <- sum((y_a_final-y_p_final)^2) #Sum of Squared Errors
SST_final<-sum((y_a_final-mean(y_a_final))^2) #Total sum of squares

R2_final<- 1-SSE_final/SST_final #this is the sum of square errors/total sum of squares
R2_final #  0.03436526

#Now the final regression is done moving onto the random forest model
#28. Running a Random Forest for artist type and popualrity 
#first need to install the package for this speciifc model

install.packages("randomForest")
library(randomForest)
rf_artist_model<-randomForest(
  popularity.x~ artist_type,
  data=songs_train)
rf_artist_model

#findings: Call: randomForest(formula = popularity.x ~ artist_type, data = songs_train)  Type of random forest: regression Number of trees: 500
# No. of variables tried at each split: 1 Mean of squared residuals: 418.4093 % Var explained: 2.41

rf_artist_model_p<-predict(rf_artist_model, newdata = songs_test) #this predicts on the test set instead of the train one 
#now I am looking at the residual 
rf_artist_residuals<-rf_artist_model_p-songs_test$popularity.x
#now I am looking at the RMSE 
rmse_artist<-sqrt(mean(rf_artist_residuals^2))
rmse_artist #20.23349 

#now I am moving onto the R sqaured 
SSE_artist<-sum(rf_artist_residuals^2)
SST_artist<-sum((songs_test$popularity.x - mean(songs_test$popularity.x))^2)
R2_artist_rf<- 1 -SSE_artist/SST_artist

R2_artist_rf #0.03437861


