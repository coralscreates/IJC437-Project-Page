# IJC437 Project Page 📚
IJC437: Introduction to Data Science Coursework Overview and Code

## 🎶Project: What are the characteristics that predict Song Popularity?
### 🎵Background Context
Music plays an influential and important part in many cultures and societies, bringing people from different backgrounds and identities together. The music industry is continuously evolving, particularly through digitalisation which has increased music consumption, global influence and revenue. Song popularity is commonly used as an indicator of musical success. Therefore, understanding the characteristics that contribute to song popularity is increasingly important for artists and industry stakeholders.

**Three main questions were explored to answer the question:**
1. Do acoustic features predict song popularity?
2. Does song type (collabrative or solo songs) predict song popularity?
3. Does artist type predict song popularity?

### 🎵Methodology 

**Data Cleaning**
- Microsoft Excel
- RStudio
  
**Explanatory Data Analysis**
  - RStudio
  - Scatterplots, correlation matrix, histograms, boxplots

**Data Analysis**
- Linear Regression Model
- Random Forest Model

**Libraries used** (*install with install.packages() function*): 
- tidyverse, ggplot2, MASS, rgl, corrplot and randomForest 
(*install with install.packages() function*)


### 🎵Key findings
1. 	Acoustic features **significantly** predict song popularity, with *all features contributing*, despite the model only explains approximately 20% of the variance.
2. 	Song type **significantly** predicts song popularity but explains very little variance, suggesting that other factors play a more significant role.
3. 	Artist type is **significantly** associated with song popularity in both linear regression and random forest models, with *DJs and rappers being the only significant predictors*.


### Repository Structure
**Cleaned CSV Files**
- dataset_songs.csv: fully merged & cleaned dataset
- official_songs_cleaned.csv: cleaned raw songs file
- official_is_pop_cleaned.csv: cleaned popularity file
- acoustic_features .csv: song acoustic features dataset file
- artists_cleaned.csv: cleaned artist information dataset file

**IJC437 R codes**
- IJC437 final code .r : R script to run analysis on final merged dataset
- IJC437 final pre-merged code.r: R script to merge datasets and then run analysis
  
## How To Run The Code
There are two options to run the code
### Option 1: Using The Fully Merged And Cleaned Dataset
1. Open the **Cleaned CSV Files** folder in the repository
2. Select and download the **"dataset_songs.csv"** file onto your laptop.
3. Open the **IJC437 R codes** folder in the repository
4. Select and download the **"IJC437 final code .r"** file and open it in RStudio
5. Import the **"dataset_songs.csv"** file into RStudio (following Step 1 in the R script)
6. Run the script from Step 1 to the end to run the code and complete the analysis 
   
### Option 2: Merging The Raw CSV Files
1. Open the **Cleaned CSV Files** folder in the repository
2. Download the relevant files onto your laptop:
   - **"official_songs_cleaned.csv"**
   - **"official_is_pop_cleaned.csv"**
   - **"acoustic_features.csv"**
   - **"artists_cleaned .csv"**
3. Open the **IJC437 R codes** folder in the repository
4.  Select and download the **"IJC437 final pre-merged code.r"** file and open it in RStudio
5. Follow each step in the script to merge the files and complete the analysis 
