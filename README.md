# IJC437-Project-Page
IJC437: Introduction to Data Science Coursework Overview and Code

## 🎶Project: What are the characteristics that predict Song Popularity?
### 🎵Background Context
Music is an influential and important part of many cultures and societies, bringing people from different backgrounds and identities together.  The music industry is continuously evolving, particularly through digitalisation which has increased music consumption, global influence and revenue. Song popularity is commonly used as an indicator of musical success and often reflects performance across multiple components, such as sales and chart rankings. For artists and industry stakeholders, understanding the characteristics that contribute to song popularity is increasingly important.

**Three main questions were explored to answer the question:**
1. Do acoustic features predict song popularity?
2. Does song type (collabrative or solo songs) predict song popularity?
3. Does artist type predict song popularity?

### 🎵Methodology 

**Data Cleaning**
- Microsoft Excel
  
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
- x
- x
- x

### Repository Structure
- dataset_songs.csv: Premerged and cleanded dataset
- official_songs_cleaned.csv: cleaned raw songs file
- official_is_pop_cleaned.csv: cleaned popularity file
- acoustic_features .csv: acoustic features of songs file
- artists_cleaned.csv: cleaned artist information file
- IJC437 final pre-merged code .r : R script to run analysis on pre-merged dataset
- IJC437 final code .r: R script to merge datasets and run analysis
- README.md: Project overview and instructions to run code
  
## How To Run The Code
There are two options to run the code
### Option 1: Using The Pre-Merged And Cleaned Dataset
1. Download the **"dataset_songs.csv"** file onto your laptop.
2. Download the **"IJC437 final pre-merged code .r"** file and open it in RStudio
3. Import the **"dataset_songs.csv"** file into RStudio (following Step 1 in the R script)
4. Run the script from Step 1 to the end to run the code and complete the analysis 
   
### Option 2: Merging The Raw CSV Files
1. Download the relevant files onto your laptop:
   - **"official_songs_cleaned.csv"**
   - **"official_is_pop_cleaned.csv"**
   - **"acoustic_features.csv"**
   - **"artists_cleaned .csv"**
3. Download the **"IJC437 final code .r"** file and open it in RStudio
4. Follow each step in the script to merge the files and complete the analysis 
