# data-preprocessing-for-machine-learning

First part:
    • read csv file in pandas dataframe 
    •  applied mean imputation and fill in empty values
    • then applied knn imputation function and put neighbours=5 and fill in empty values
    • then used knn classifier with training data 75% and testing data 25%. 

Second part:
    • read again the csv file updated in the first part
    • applied binning on the age attribute and created six bins
    • then calculated the entropy at five points of class attribute and find out the information gain by dividing the dataset with respect to the age attribute.
    • Then we divide the dataset in two parts at point where the information gain is highest.

Third part:
    • we normalized the whole csv file for bonus as only normalization of two attributes was required
    • the normalization is in the range from 0-1

Forth part:
    • found the most frequent top 20 words.
    • then took their cosine similarity.
