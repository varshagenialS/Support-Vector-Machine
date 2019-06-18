# Support-Vector-Machine
Support Vector Machine Algorithm for prediction using uci heart dissease dataset

support-vector machines is supervised learning model used for classification and regression analysis. 
svm is trained using dataset ,it calculates the dot product between samples of vector to classify a new data point.

uci heart disease dataset  contains 76 attributes, but only 14 are considered useful. 

14 attributes used mostly are: 
1. age 
2. sex 
3. cp 
4. trestbps 
5. chol 
6. fbs
7. restecg
8. thalach 
9. exang 
10. oldpeak 
11. slope 
12. ca 
13.  thal 
14. num (the predicted attribute) 

uci heart disease repository has 4 regions data,clevland region dataset has less null values 
hence using only clevland dataset  we get good accuracy
I have separately compared accuracy between clevland dataset and accuracy after merging all regions
also i have compared accuracy by removing columns which has more null values when i included all regions.
