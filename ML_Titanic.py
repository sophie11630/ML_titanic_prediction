#Import libraries
import pandas as pd
import os
import numpy as np
from sklearn                         import tree
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.datasets                import load_iris
from sklearn.model_selection         import train_test_split
from sklearn.naive_bayes             import GaussianNB
from sklearn                         import preprocessing

pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)

#%%%% Read data
dta_titanic_train          = pd.read_csv('data_titanic_train.csv')
dta_titanic_test           = pd.read_csv('data_titanic_test.csv')
dta_titanic_test_survive   = pd.read_csv('data_titanic_gender_submission.csv')

# change sex to a dummy variable
dta_titanic_train["Sex"]   = dta_titanic_train["Sex"].replace(["male","female"],[1,0])
dta_titanic_test["Sex"]    = dta_titanic_test["Sex"].replace(["male","female"],[1,0])

# merge two test_data
dta_titanic_test           = dta_titanic_test.merge(dta_titanic_test_survive, how='inner', on = "PassengerId")


#%%%% clean and adjust datasets
scaler        = preprocessing.StandardScaler()
# since some columns are unique for each passenger and we have many missing values in Cabin, I only kept several columns
column_list   = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

dta_titanic_train             = dta_titanic_train[column_list]
dta_titanic_train             = dta_titanic_train.dropna()
dta_titanic_train['Embarked'] = dta_titanic_train['Embarked'].replace(["C","Q","S"],[0,1,2])

dta_titanic_test              = dta_titanic_test[column_list]
dta_titanic_test              = dta_titanic_test.dropna()  
dta_titanic_test['Embarked']  = dta_titanic_test['Embarked'].replace(["C","Q","S"],[0,1,2])


#%%%% split dta_titanic_train into train and validating data
dta_titanic_train['ML_group']   = np.random.randint(100,size = dta_titanic_train.shape[0])
dta_titanic_train               = dta_titanic_train.sort_values(by='ML_group').reset_index()

# union train, validate, test data together
dta_titanic                     = pd.concat([dta_titanic_train, dta_titanic_test], ignore_index=True)

inx_train                       = dta_titanic.ML_group<70                     
inx_valid                       = dta_titanic.ML_group>=70
inx_test                        = dta_titanic.ML_group.isnull()


#%%%% TVT-splitting
feature_names  = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X       = scaler.fit_transform(dta_titanic[feature_names].values)

Y_train = dta_titanic.Survived[inx_train].to_list()
Y_valid = dta_titanic.Survived[inx_valid].to_list()
Y_test  = dta_titanic.Survived[inx_test].to_list()

X_train = X[np.where(inx_train)[0],:]
X_valid = X[np.where(inx_valid)[0],:]
X_test  = X[np.where(inx_test)[0],:]

right_to_total_dict2 = {}


#%%%% Linear Regression
model  = LinearRegression()
clf1 = model.fit(X_train, Y_train)

dta_titanic['Survived_hat_reg'] = np.concatenate(
        [
                clf1.predict(X_train),
                clf1.predict(X_valid),
                clf1.predict(X_test )
        ]
        ).round().astype(int)

#if diag_hat_reg greater than 1, make it 1
dta_titanic.loc[dta_titanic['Survived_hat_reg']>1,'Survived_hat_reg'] = 1
dta_titanic.loc[dta_titanic['Survived_hat_reg']<0,'Survived_hat_reg'] = 0

# confusion matrix for linear regression
conf_matrix1 = np.zeros([2,2])
for i in range(2):
    for j in range(2):
        conf_matrix1[i,j] = np.sum((dta_titanic[inx_test].Survived == i)*(dta_titanic[inx_test].Survived_hat_reg == j))

print(conf_matrix1)

# print the probability of right(almost right) prediction
right = conf_matrix1.diagonal(offset = 0).sum()
total = conf_matrix1.sum()
ratio_reg = round(right/total,4)
print(ratio_reg)

right_to_total_dict2["Linear regression"] = ratio_reg


#%%%% Naive Bayes Classification
clf2 = GaussianNB().fit(X_train, Y_train)
dta_titanic['Survived_hat_NB'] = np.concatenate(
        [
                clf1.predict(X_train),
                clf1.predict(X_valid),
                clf1.predict(X_test)
        ]
        ).round().astype(int)
dta_titanic.loc[dta_titanic['Survived_hat_NB']>1,'Survived_hat_NB'] = 1
dta_titanic.loc[dta_titanic['Survived_hat_NB']<0,'Survived_hat_NB'] = 0

# confusion matrix for  Naive Bayes
conf_matrix2 = np.zeros([2,2])
for i in range(2):
    for j in range(2):
        conf_matrix2[i,j] = np.sum((dta_titanic[inx_test].Survived == i)*(dta_titanic[inx_test].Survived_hat_NB ==j))

print(conf_matrix2)

#print the probability of right(almost right) prediction
right = conf_matrix2.diagonal(offset = 0).sum()
total = conf_matrix2.sum()
ratio_nb = round(right/total,4)
print(ratio_nb)

right_to_total_dict2["Naive Bayes"] = ratio_nb


#%%%% Trees
criterion_chosen     = ['entropy','gini']
random_state         = 96
max_depth            = 10
criterion_best       = {}
results_tree         = []

#select criterion and depth -- if the results are same, I chose the smallest depth and entropy
for criterion in criterion_chosen:
    results_list         = []
    k_dict_tree          = {}
    
    #select the best depth
    for depth in range(2,max_depth):
        clf3    = tree.DecisionTreeClassifier(
                criterion    = criterion, 
                max_depth    = depth,
                random_state = 96).fit(X_train, Y_train)
    
        results_list.append(
            np.concatenate(
                    [
                            clf3.predict(X_train),
                            clf3.predict(X_valid),
                            clf3.predict(X_test)
                    ]).round().astype(int)
            )
        
        dta_results_tree              = pd.DataFrame(results_list).transpose()
        dta_results_tree.loc[dta_results_tree[depth-2]>1,depth-2] = 1
        dta_results_tree.loc[dta_results_tree[depth-2]<0,depth-2] = 0
        dta_results_tree['inx_train'] = inx_train.to_list()
        dta_results_tree['inx_valid'] = inx_valid.to_list()
        dta_results_tree['inx_test']  = inx_test.to_list()
        dta_results_tree["Survived"] =  dta_titanic.Survived.copy()
                
        # use validating data for choosing hyperparameter
        conf_matrix3 = np.zeros([2,2])
        for i in range(2):
            for j in range(2):
                conf_matrix3[i,j] = np.sum((dta_results_tree[dta_results_tree.inx_valid].Survived == i) * (dta_results_tree[dta_results_tree.inx_valid][depth-2] == j))
                
        #print the probability of right(almost right) prediction
        right = conf_matrix3.diagonal(offset = 0).sum()
        total = conf_matrix3.sum()
        
        #add k and its right-to-total ratio to the dictionary
        k_dict_tree[depth] = round(right/total,4)
        
    # print each dictionary and find the smallest best depth
    print(k_dict_tree)
    
    # change values in dict to a list and find the first max ratio   
    ratio_list = list(k_dict_tree.values())
    
    # find the corresponding key -- our best depth
    max_index = ratio_list.index(max(ratio_list))
    better_depth = list(k_dict_tree.keys())[max_index]
    
    # below will have: ex. criterion_best = {"entropy": [3, 0.9492], "gini": [4, 0.9492]}
    criterion_best[criterion] = [better_depth, max(ratio_list)]
    
    #append dta_results_tree to list
    #at the end we will have two dta_results_tree in list for entropy and gini respectively
    results_tree.append(dta_results_tree)
    
# choose entropy or gini and their best depth 
ent_gini_choice_list = np.array(list(criterion_best.values()))
better_ratio = list(ent_gini_choice_list[:,1])
better_index = better_ratio.index(max(better_ratio))
ent_or_gini  = criterion_chosen[better_index]
best_depth   = int(list(ent_gini_choice_list[:,0])[better_index])
print("The depth that gives give the best right-to-total ratio is: ", ent_or_gini, ", ", best_depth)
    
    
#use test data to construct conf_matrix:

#the final dta_results_tree for criterion_best
dta_results_tree = results_tree[better_index]

conf_matrix_tree = np.zeros([2,2])
for i in range(2):
    for j in range(2):
        conf_matrix_tree[i,j] = np.sum((dta_results_tree[dta_results_tree.inx_test].Survived == i) * (dta_results_tree[dta_results_tree.inx_test][best_depth-2] == j))

print("")
print("We use test data to construct a confusion_matrix: ")
print(conf_matrix_tree)
right_tree = conf_matrix_tree.diagonal(offset = 0).sum()
total_tree = conf_matrix_tree.sum()
ratio_tree = round(right_tree/total_tree,4)
print(ratio_tree)

right_to_total_dict2["Trees"] = ratio_tree


#%%%% KNN
max_k_nn = 11
k_dict_knn = {}
results_list_knn = []

# do for 1 to 10 neighbors
for k in range(1,max_k_nn):
    clf4      = KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
    results_list_knn.append(
            np.concatenate(
                    [
                            clf4.predict(X_train),
                            clf4.predict(X_valid),
                            clf4.predict(X_test)
                    ]))
    
   
    dta_results_knn              = pd.DataFrame(results_list_knn).transpose()
    dta_results_knn.loc[dta_results_knn[k-1]>1,k-1] = 1
    dta_results_knn.loc[dta_results_knn[k-1]<0,k-1] = 0
    dta_results_knn['inx_train'] = inx_train.to_list()
    dta_results_knn['inx_valid'] = inx_valid.to_list()
    dta_results_knn['inx_test']  = inx_test.to_list()
    dta_results_knn["Survived"] = dta_titanic.Survived.copy()

    
    #confusion matrix
    conf_matrix4 = np.zeros([2,2])
    for i in range(2):
        for j in range(2):
            conf_matrix4[i,j] = np.sum((dta_results_knn[dta_results_knn.inx_valid].Survived == i) * (dta_results_knn[dta_results_knn.inx_valid][k-1] == j))
            
    right = conf_matrix4.diagonal(offset = 0).sum()
    total = conf_matrix4.sum()
    
    #add k and its right-to-total ratio to the dictionary
    k_dict_knn[k] = round(right/total,4)
    
# print dictionary and find the smallest best k
print(k_dict_knn)

ratio_list_knn = list(k_dict_knn.values())
max_index_knn = ratio_list_knn.index(max(ratio_list_knn))
best_k = list(k_dict_knn.keys())[max_index_knn]
print("The k that gives give the best right-to-total ratio is: ", best_k)

#use test data to construct conf_matrix

conf_matrix_knn = np.zeros([2,2])
for i in range(2):
    for j in range(2):
        conf_matrix_knn[i,j] = np.sum((dta_results_knn[dta_results_knn.inx_test].Survived == i) * (dta_results_knn[dta_results_knn.inx_test][best_k-1] == j))

print("")
print("We use test data to construct a confusion_matrix: ")
print(conf_matrix_knn)
right_knn = conf_matrix_knn.diagonal(offset = 0).sum()
total_knn = conf_matrix_knn.sum()
ratio_knn = round(right_knn/total_knn,4)
print(ratio_knn)

right_to_total_dict2["KNN"] = ratio_knn


#%%%% Lasso
clf5 = linear_model.Lasso(alpha=0.1)
clf5.fit(X_train, Y_train)

dta_titanic['Survived_hat_lasso']             = np.concatenate(
        [
                clf5.predict(X_train),
                clf5.predict(X_valid),
                clf5.predict(X_test)
        ]).round().astype(int)

dta_titanic.loc[dta_titanic['Survived_hat_lasso']>1,'Survived_hat_lasso'] = 1
dta_titanic.loc[dta_titanic['Survived_hat_lasso']<0,'Survived_hat_lasso'] = 0

# confusion matrix for lasso
conf_matrix5 = np.zeros([2,2])
for i in range(2):
    for j in range(2):
        conf_matrix5[i,j] = np.sum((dta_titanic[inx_test].Survived == i)*(dta_titanic[inx_test].Survived_hat_lasso==j))    

print(conf_matrix5)

# print the probability of right(almost right) prediction
right = conf_matrix5.diagonal(offset = 0).sum() 
total = conf_matrix5.sum()
right_to_total = round(right/total,4)
print(right_to_total)

right_to_total_dict2["Lasso"] = right_to_total


#%%%% Compare which classifier is the best here
print(right_to_total_dict2)

# predictive model that gives the best oos prediction
best_mod = [model for model, value in right_to_total_dict2.items() if value == max(right_to_total_dict2.values())]
        
# print the maximum right_to_total
print("")
max_rrt = max(right_to_total_dict2.values())

print("From the results above, ", end = "") 
print(*best_mod, sep = ", ", end=" ")
print("give(s) us the best prediction, and the right-to-total ratio for them is {}.".format(max_rrt))


