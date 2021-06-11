#!/usr/bin/env python
# coding: utf-8

# In[75]:


#clear workspace
get_ipython().run_line_magic('reset', '-f')


# In[ ]:





# In[195]:


#importing packages needed

import pandas as pd #for data manipulation
import numpy as np #for numerical 
import matplotlib.pyplot as plt #. for basic plots
import seaborn as sns #variety of visualisation pattersn
get_ipython().run_line_magic('matplotlib', 'inline')

###importing machime learning libraries and functionalities

from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting


 
from IPython.display import Image  



from sklearn import tree
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

print  ('Your pandas version is: %s' % pd.__version__)


# In[77]:


#loading dataset into workspace
heart = pd.read_csv("heart.csv")


# ## DATA UNDERSTANDING

# In[78]:


#quick look at data types
heart.info()

##An initial look at the dataset shows it has 303 observations which corresponds 
#to 303 patients and 14 different feature variables.

#Some of the variables like sex, cp(chest pain type), or the target variable are in 
#the wrong data type integers and will 
#need to be converted to categorical variables before any analysis/machine learning carried out.

## Understanding the Data

Origin of Data: Kaggle

Domain background on heart disease and what each feature variable means

Initially, the dataset contains 76 features or attributes from 303 patients; however, published studies chose only 14 features that are relevant in predicting heart disease. Hence, here we will be using the dataset consisting of 303 patients with 14 features set.

Selection bias might be an issue with data as sample might bot be representative of the population of cardiac patients in Cleveland or even America/World. This selection bias can result in erroneous conclusions.

(Mathur, P., Srivastava, S., Xu, X., & Mehta, J. L. (2020). Artificial Intelligence, Machine Learning, and Cardiovascular Disease. Clinical Medicine Insights. Cardiology, 14, 1179546820927404. https://doi.org/10.1177/1179546820927404.)


age of patient in years

sex of patient (1 = male; 0 = female)

Cp: chest pain type (4 values) -(0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)

Trestbps: resting blood pressure  (in mm Hg on admission to the hospital)

Chol: serum cholesterol in mg/dl

Fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)

Restecg: resting electrocardiographic results (values 0,1,2)  (0 = normal; 1 = having ST-T; 2 = hypertrophy)

Thalac: maximum heart rate achieved 

Exang: exercise induced angina  (1 = yes; 0 = no)

oldpeak: ST depression induced by exercise relative to rest

Slope: the slope of the peak exercise ST segment  

Ca: number of major vessels (0-3) colored by fluoroscopy

thal : a blood disorder called thalasemia [1 = normal, 2 = fixed defect, 3 = reversible defect]

Target : whether patient has a heart condition (1) or not (0)
# ## Business Question: What causes a heart condition?
# 
# https://www.bhf.org.uk/informationsupport/heart-matters-magazine/medical/angina-common-questions#:~:text=Angina%20is%20a%20symptom%20caused,%2C%20jaw%2C%20back%20or%20stomach.
# 
# 
# 
# British Medical Journal """Angina
# pectoris (derived from the Latin verb ‘angere’ to
# strangle) is chest discomfort of cardiac origin. It is a
# common clinical manifestation of IHD with an estimated prevalence of 3%–4% in UK adults"""
# 
# (Ford TJ, Berry CAngina: contemporary diagnosis and managementHeart 2020;106:387-398.)
# 
# 
# 
# Introduction
# 
# In order to answer the question of what causes a heart condition, heavy domain expertise on heart disease is required. This is because understanding what the variables actually measure and how they are measured is vital in determining causes of heart disease. 
# 
# According to the NHS website (reference: https://www.nhs.uk/conditions/coronary-heart-disease/), Coronary Heart disease (CHD) is responsible for a large number of deaths globally. Another name is Ischaemic Heart diesease.
# 
# With the prevalence of coronavirus, the British Heart Foundation (BHF https://www.bhf.org.uk/informationsupport/heart-matters-magazine/news/coronavirus-and-your-health) has declared that patients with a heart disease are more at risk of complications from coronavirus.
# 
# 

# In[79]:


#quick view of data
heart.head()

#display of top five rows of dataset looks at the type of values 
#and further confirms that some variables especially those whose values a
#are 0 or 1 should be categorical and not integers"""


# In[80]:


#check unique values in variable
heart.nunique()

#unique values in each variable further reveals variables with discrete number (2,4,3) are categorical


# In[81]:


heart.target.value_counts()

##number of observations in each class of the target variable is not balanced.
##In other words, this sample dataset has more patients with heart disease than those without.

##This sample might not be representative of the hospital or the country or even the world.
##it would be interesting to know the timeframe of this data too, when was the data collected
#It is possible that data collected at a different time might be completely different in terms of
#the number of people with or without heart disease
##so have to be careful when drawing any inferences from the data.


# In[82]:


##check categorical variables
heart.sex.value_counts()

##the data reveals there are more males (201) than females (95) amongst patients.
countFemale = len(heart[heart.sex == 0])
countMale = len(heart[heart.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(heart.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(heart.sex))*100)))


# In[83]:


###check categorical variables
heart.cp.value_counts()


# In[84]:


##check categorical variables
heart.fbs.value_counts()


# In[85]:


##check categorical variables
heart.restecg.value_counts()


# In[86]:


##check categorical variables
heart.slope.value_counts()


# In[87]:


##check categorical variables
heart.ca.value_counts()


# In[88]:


##check categorical variables
heart.exang.value_counts()


# In[89]:


##check categorical variables
heart.oldpeak.value_counts()


# ## DATA CLEANING.

# In[90]:



##ca=4 is incorrect as the maximum number of major bloood vessels is 3
heart.loc[heart['ca'] == 4, 'ca'] = np.NaN


# In[91]:


##check categorical variables
heart.ca.unique()



# In[92]:


##feature thal ranges from 1 to 3. 
##the value 0 is implausible(incorrect/errenous value) - out of range for the likert scale
#This implies the two values with zero need to be changed to missing numbers


heart.loc[heart['thal']== 0, 'thal'] = np.NaN



# In[93]:


##check missing values agian
heart.isnull().sum()


# In[94]:


##dropping 7 rows with at least one missing value

## these 7 rows were derived from the thal and ca variables containing implausible values
##dropped 7 missing values as it makes up only 0.165% of the data

heart_thalNA = heart.dropna()


# In[95]:


heart_thalNA.info()

20/3878


# In[96]:


##checking there are no thal==0 values
heart_thalNA['thal'].unique()



# In[97]:


##checking for duplicate rows

##there are no duplicated rows in the data

duplicated_rows = heart_thalNA[heart_thalNA.duplicated(keep=False)]
duplicated_rows


# In[98]:


##statistical summary description of data
heart_thalNA.describe()


# ###EDA:      NUMERICAL FEATURES - CHECKING FOR OUTLIERS

# In[99]:


ax_cont = sns.boxplot(data = heart_thalNA.loc[:, ['age','trestbps','chol','thalach', 'oldpeak']],
                    orient="h", palette="Set2")


# ### Outliers Identification and Removal

# In[100]:


##Inspired from https://towardsdatascience.com/exploratory-data-analysis-on-heart-disease-uci-data-set-ae129e47b323

# define continuous variables & plot
continuous_variables = ['age','trestbps','chol','thalach', 'oldpeak']  



def outliers(df_out, drop = False):
    for variable in df_out.columns:
        
        feature_data = df_out[variable]
        
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given variable
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given variable
        
        IQR = Q3-Q1 #Interquartile Range
        
        outlier_step = IQR * 1.5 #observations 1.5 * IQR greater or less than are dropped as outliers
        
##defining list of outliers as numbers less or greater than 1.5IQR
        
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        
        if not drop:
            print('For the variable {}, No of Outliers is {}'.format(variable, len(outliers)))
        if drop:
            
            heart_thalNA.drop(outliers, inplace = True, errors = 'ignore')
            
            
            print('Outliers from {} variable removed'.format(variable))
            
            
outliers(heart_thalNA[continuous_variables])



# In[101]:


##dropping outliers from the data
outliers(heart_thalNA[continuous_variables], drop=True)


# In[104]:


heart_thalNA.describe()  ##277 data values now left after removing outliers.


# In[105]:


#check for missing values
heart_thalNA.isnull().sum()

#no missing values in dataset.


# In[106]:


##check out frequency distribution of target
heart_thalNA['target'].value_counts()

# - Target : whether patient has a heart condition (0) or not (1)
#0 - Do not have heart condition
#no_heartCond = target[target == 0].shape[0]
#print(no_heartCond)

# - Target : whether patient has a heart condition (0) or not (1)
#have_heartCond = target[target == 1].shape[0]

#print(have_heartCond)


# After checking for implauisble values and dropping the 7 (0.0165%) implausible values which were converted to NA. 
# 
# Then checking for duplicated rows and outliers. then dropping outliers which were about 20 outliers (0.5%) of the new dataframe.
# 
# Data is now cleaner and we can go visualise the data and begin analysis
# 
# We have 154 patients without a heart condition in this dataset and 123 patients with a heart condition.

# In[107]:


#obtaining both features and target variables
features = heart_thalNA.iloc[:,0:13]

features.head()


# In[ ]:





# ## Furtheer data Cleaning

# 
# 

# In[109]:


##convert variables to their correct types

heart2 = heart_thalNA.copy()

heart2['sex'] = heart2['sex'].astype("category")
heart2['cp'] = heart2['cp'].astype("category")
heart2['fbs'] = heart2['fbs'].astype("category")
<- 

heart2['restecg'] = heart2['restecg'].astype("category")
heart2['exang'] = heart2['exang'].astype("category")
heart2['slope'] = heart2['slope'].astype("category")
heart2['ca'] = heart2['ca'].astype("category")
heart2['thal'] = heart2['thal'].astype("category")


# In[110]:


heart2.info()


# In[111]:


##
heart3 = heart2.copy()


#renaming columns with names for better visualisations

heart3 = heart3.rename(columns={"cp": "chestPain_type", "trestbps": "restBlood_pressure", 
                      "chol": "serum_cholesterol", 
                      "fbs": "fastbloodSugar",
                      "restecg": "restecg_results",
                      "thalach": "max_heartrate",
                      "exang": "chestpain_exercise",
                      "oldpeak": "induced_STdepression",
                      "slope": "slope_peak",
                      "ca": "number_vesselscolored",
                      "thal": "defect_type",
                      "target": "heart_disease"
                     })


# In[112]:


heart3.heart_disease.value_counts()

heart3.nunique()


# In[119]:


#change values in categorical variables in the heart3 dataframe

#def change(sex):
#    if sex == 0:
#        return 'female'
#    else:
#        return 'male'
    
#heart3['sex'] = heart3['sex'].apply(change)

heart3['heart_disease'] = heart3.heart_disease.replace({1: "Disease", 0: "No_disease"})
heart3['sex'] = heart3.sex.replace({1: "Male", 0: "Female"})
heart3['chestPain_type'] = heart3.chestPain_type.replace({
                          0: "typical_angina", 
                          1: "atypical_angina", 
                          2:"non-anginal pain",
                          3: "asymtomatic"})

heart3['chestpain_exercise'] = heart3.chestpain_exercise.replace({1: "Yes", 0: "No"})

heart3['fastbloodSugar'] = heart3.fastbloodSugar.replace({1: "True", 0: "False"})

heart3['slope_peak'] = heart3.slope_peak.replace({0: "upsloping", 1: "flat", 2:"downsloping"})

heart3['defect_type'] = heart3.defect_type.replace({1: "fixed_defect", 2: "reversable_defect", 3:"normal"})

     
heart3['restecg_results'] = heart3.restecg_results.replace({0: "normal", 1:"ST_abnormality", 2:"hypertrophy"})
    


    




# ## EXPLORATORY DATA ANALYSIS including Visualisation.

# In[120]:


## Target(Dependent) Variable

#
sns.set(style="darkgrid")
disease_plot = sns.countplot(x = 'heart_disease', data=heart3)



plt.title('Distribution of patients with disease.')
plt.ylabel('Number of patients', fontsize=12)
plt.xlabel('heart disease', fontsize=12)

#to create percentages for each plot
total = len(heart3)

for p in disease_plot.patches:
    
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    
    x = p.get_x() + p.get_width()
    y = p.get_height()
    
    disease_plot.annotate(percentage, (x, y), va='center')


plt.tight_layout()


plt.show()

##the plot shows that in the data, 55.6% of patients had the disease compared to 44.4% of patients without the distance.

##clearly a class imbalnce issue.

##have to do some random sampling when splitting data into train and test sets.





# In[121]:


##check out frequency distribution of categorical variables
sex_patient = heart3['sex'].value_counts()

#207 male in the data and 96 female

print(sex_patient)





#heart3.info()


# In[122]:


#check plots of male, female against whether patient has heart disease or not
sns.set(style="darkgrid")
sex_plot = sns.countplot(x = 'sex', data=heart3, hue='heart_disease')



plt.title('Distribution of male and female heart disease patients')
plt.ylabel('Number of patients', fontsize=12)
plt.xlabel('sex', fontsize=12)

#to create percentages for each plot
total = len(heart3)

for p in sex_plot.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    sex_plot.annotate(percentage, (x, y), ha='center', va='center')


plt.tight_layout()


plt.show()


##the plot shows that 31.4% of patients who had heart the disease were male compared to
#24.2% for patients who had the disease who were female.

#For patients WITHOUT disease 38.3% were male and 6.1% were female.


# In[117]:


chest_pain = sns.countplot(data= heart3, x='chestPain_type', hue='heart_disease')

plt.title('Heart diagnosis across different levels of chest pain')


plt.ylabel('Number of patients', fontsize=12)

plt.xlabel('chest pain type', fontsize=12)

#to create percentages for each plot
total = len(heart3)

for p in chest_pain.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    chest_pain.annotate(percentage, (x, y), ha='center', va='center')


plt.tight_layout()


plt.show()


 


# For the feature chest pain type, patients who were asymtomatic that is
# patients who did not show any of the symptoms, only 5.4% had the disease compared to 2.5% without the disease.
# 
# However,, for patients with pain that was non-anginal (i.e non chest pain) - 22.4% had the disease compared to 6.1%
# 
# for patients with atypical angina
# (angina pectoris(typical angina) without associated classical symptoms e.g tightness of chest,shortness of breath.
# Symptoms of atypical include nausea, weakness, sweating
# Ref https://www.ncbi.nlm.nih.gov/medgen/149267#:~:text=Definition,from%20NCI%5D)
# 
# 
# 
# patients with typical angina - 13.4% have disease compared to 32.9% without disease
# this is interesting because you will expect patients with typical angina syptoms to be more likely to have heart disease. Most people diagnosed with angina have underlying heart disease according to the BHF.
# 
# However according to the BHF(https://www.bhf.org.uk/informationsupport/heart-matters-magazine/medical/angina-common-questions#:~:text=Angina%20is%20a%20symptom%20caused,%2C%20jaw%2C%20back%20or%20stomach.), Just because you have typical angina symptoms doesnt mean you are going to have a heart disease.
# 
# Our data again might not been representative of worlwide data on heart disease aptients.

# In[123]:


fast_blood = sns.countplot(data= heart3, x='fastbloodSugar', hue='heart_disease')


plt.title('Heart diagnosis across different fasting blood sugar levels')


plt.ylabel('Number of patients', fontsize=12)

plt.xlabel('fast blood sugar levels', fontsize=12)


#to create percentages for each plot
total = len(heart3)

for p in fast_blood.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    fast_blood.annotate(percentage, (x, y), ha='center', va='center')


plt.tight_layout()


plt.show()


# Fasting bloood sugar measures blood sugar levels. It is an indicator of diabetes.
# 
# A fbs above 120mm/d is considered True diabetic while below is false. The plot shows there are higher number of heart disease patients without diabetes. so this might not be a discriminatory feature between those who have had disease and those who do not.
# 
# Higher amounts of sugar in the blood than normal means your body has not got enough insulin to breakdown glucose.
# 
# Over time higher blood sugar levels damage blood vessels like the one leading to the heart and may cause heart disease.
# 
# In the plot above, 6.1% of patients who have had disease had a fasting blood sugar above 120mmHg compared to 38.3% patients with a lower (120mm/d) fasting blood sugar.
# 
# However, it is important to note that fasting blood sugar levels are not constant and change depending on so many factors like time of day, whether one had ameal or not etc.
# 

# In[124]:


rest_ecg = sns.countplot(data= heart3, x='restecg_results', hue='heart_disease')

plt.title('Heart diagnosis across different rest ecg results')


plt.ylabel('Number of patients', fontsize=12)

plt.xlabel('rest ECG results', fontsize=12)

#to create percentages for each plot
total = len(heart3)

for p in rest_ecg.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    
    rest_ecg.annotate(percentage, (x, y), ha='center', va='center')


plt.tight_layout()


plt.show()

#resting electrocardiographic results

#According to the NHS (https://www.nhs.uk/conditions/electrocardiogram/)
#An electrocardiogram (ECG) is a simple test that can be used to check your heart's rhythm and electrical activity.



#For patients with a normal restecg result, 22.7% have the disease, however an abnormal St-T wave shows that 32.5% 
#have the disease compared to 18.1%. This feature  to be a strong indicator that a patient will have the disease. While a probable or defeinite left ventricular hypertrophy doesnt show
#any discriminatoryy power between diseased patients and non-diseased patients.


# In[125]:


chestPain_ex = sns.countplot(data= heart3, x='chestpain_exercise', hue='heart_disease')

plt.title('Heart diagnosis across different levels of exercise-induced chest pain')
plt.ylabel('Number of patients', fontsize=12)
plt.xlabel('any exercise-induced chest pain', fontsize=12)

#to create percentages for each plot
total = len(heart3)

for p in chestPain_ex.patches:
    
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    chestPain_ex.annotate(percentage, (x, y), ha='center', va='center')
    
    


plt.tight_layout()


plt.show()

heart3.info()

##The plot below shows that forpatients with exercise-induced chest pain, those with no heart disease(23.8%) outweighs those with
#heart disease(7.9%). this feature might not have enough discriminatory power to help predict.


# In[131]:


# for plotting, group categorical features in cat_feat
# to create dist in 8 feature, 9th is the target, 

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(17,10))

categorical_features = ['sex', 'chestPain_type', 'fastbloodSugar', 'restecg_results', 'chestpain_exercise',
            'slope_peak', 'number_vesselscolored', 'defect_type', 'heart_disease']

#for feature in categorical_features:
#        percentage = f'{100 * feature.get_height() / total:.1f}%\n'
#        x = feature.get_x() + feature.get_width() / 2
#        y = feature.get_height()
#    
#        feature.annotate(percentage, (x, y), ha='center', va='center')



for idx, feature in enumerate(categorical_features):
    ax = axes[int(idx/3), idx%3]
    if feature != 'heart_disease':
        sns.countplot(x=feature, hue='heart_disease', data=heart3, ax=ax, palette='Set2')
        
    


# ## Visualising continuous features
# #create list with 5 continuos features
# continous_features = ['age', 'serum_cholesterol', 'max_heartrate', 'restBlood_pressure','induced_STdepression']  
# 
# #create pairplot sof continuos features with target class variable as hue
# sns.pairplot(heart3[continous_features + ['heart_disease']], hue='heart_disease')
# 
# 
# ##For the age variable, the plots show a normal distribution that older people between 50 and 60 with heart disease.
# #Dataset contains older people than younger. Positve correlatioon betwwen age and serum cholesterol age and resting blood pressure
# #no relationship with induced sT depression(old peak). negative correlation between age and max heart rate.
# 
# #resting blood pressure might not have enough discrimnatory power as there doesnt seem to be much difference between diseased 
# #and non-diseased from the plots
# 
# heart3.nunique()

# In[ ]:





# In[45]:


# to understand the relationship between age and chol in each of the target based on sex.
sns.lmplot(x="age", y="serum_cholesterol", hue="sex", col="heart_disease",
           markers=["o", "x"],
           palette="Set2",
           data=heart3)
plt.show()

#For those with the disease, there is a positive correlation for female patients as they get older
#serum cholesterol increases while male patients with the disease, the serum cholesterioldoesnt seem to change at all.


#So cholesterol and age might be intereting features to include


# In[46]:


##correlation plots using seaborn

corr = heart3.corr()

g = sns.heatmap(corr,  vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.2f', cmap='coolwarm')

sns.despine()

g.figure.set_size_inches(14,10)

plt.show()

##correlation plot shows generally ot very stron positive or negative correlations.

##the strongest negative correlation in the data are between age and maximum heart rate and maximuum heart rate 
#and induced sT depression.

#The strongest positive correlations in this data exist between restin g blood pressure and age so as age 
#increases resting blood pressure increases too which is consistent with medical literature.

##


# In[47]:


#Histograms of continuos variables
plt.figure(figsize=(16,7))
sns.distplot(heart3['age'], bins=40)
plt.title('Age distribution of Patients')


# In[48]:


pd.crosstab(heart3.age,  heart3.heart_disease).plot(kind="bar",figsize=(20,6))


plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


##The histogram shows the distribution of heart disease is more frequent in older patients than younger patients.


# In[49]:


#plot of cserum cholesterol variable

#pd.crosstab(heart3.serum_cholesterol,  heart3.heart_disease).plot(kind="line",figsize=(50,20))


#plt.title('Heart Disease Frequency for Serum Cholesterol Feature')
#plt.xlabel('Serum Cholesterol')
#plt.ylabel('Frequency')



#plt.savefig('heartDiseaseAndSerumCholesterol.png')

#plt.show()


heart3.info()


# ### FEATURE ENGINEERING

# In[132]:


##Creating dummy variables for all categorical variables

###drop_first=True drops first category for each categorical variable

df1 = pd.get_dummies(heart3, columns=['sex', 'chestPain_type', 'fastbloodSugar','restecg_results',
                              'chestpain_exercise', 'slope_peak', 'number_vesselscolored', 'defect_type'], drop_first=True)
df1


# In[133]:


df1['heart_disease'] = df1.heart_disease.replace({"Disease":1, "No_disease":0})


# In[134]:


df1.info()


# ### MODELS
# 

# In[135]:



##separate into independent and dependent variables

X = df1.drop("heart_disease", 1)  ##without target class or column feature
y = df1.heart_disease  ##target claass feature/column

##TRAIN-test SPLIT of the data = 80-20
X_train, X_test, y_train, y_test = train_test_split(
                                X, 
                                y, 
                                test_size = .2,
                                random_state=10)

X_train.shape, y_train.shape, X_test.shape , y_test.shape


# In[136]:


# # calculate the correlation matrix
corr = df1.corr()

# plot the heatmap
fig = plt.figure(figsize=(20,16))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            linewidths=.75)


# Random Forest Classiier

# In[185]:





#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100, max_depth=3)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[186]:




# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[187]:


##plotting a decision tree from the random Forest


estimator = clf.estimators_[5]

feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values


# In[188]:


plt.figure(figsize=(50,50))
_ = tree.plot_tree(clf.estimators_[0], feature_names=X_train.columns, filled=True)

clf.estimators_[0].tree_.max_depth


# In[151]:


##make a prediction if a specific patient with a set of characteristics will have heart deisease
#using clf.predict([list of column values in column order])


# ## Feature Importance

# In[152]:


feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_imp


# In[153]:


# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[154]:


### Based on feature importance, a new model is refitted to see if accuracy is improved


# Split dataset into features and labels
X2 = df1[["defect_type_reversable_defect",      
"max_heartrate", "induced_STdepression", "age", "serum_cholesterol",              
"defect_type_normal", "chestPain_type_typical_angina", "restBlood_pressure",                 
"chestpain_exercise_Yes", "slope_peak_flat", "sex_Male"]]  
           
####Removed features
#number_vesselscolored_1.0          0.031712
#number_vesselscolored_2.0          0.029262
#chestPain_type_non-anginal pain    0.019471
#restecg_results_normal             0.017626
#number_vesselscolored_3.0          0.013983
#fastbloodSugar_True                0.009514
#chestPain_type_atypical_angina     0.008298
#slope_peak_upsloping               0.006431
#restecg_results_hypertrophy        0.000058
           
           
y2=df1['heart_disease']                                       

# Split dataset into training set and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.80, random_state=10) # 80% training and 20% test


# In[155]:


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X2_train,y2_train)

# prediction on test set
y2_pred=clf.predict(X2_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y2_test, y2_pred))


# In[156]:



#number_vesselscolored_1.0          0.031712
#number_vesselscolored_2.0          0.029262
#chestPain_type_non-anginal pain    0.019471
#restecg_results_normal             0.017626
#number_vesselscolored_3.0


# In[157]:



# Split dataset into features and labels
X3 = df1[["defect_type_reversable_defect",      
"max_heartrate", "induced_STdepression", "age", "serum_cholesterol",              
"defect_type_normal", "chestPain_type_typical_angina", "restBlood_pressure",                 
"chestpain_exercise_Yes", "slope_peak_flat", "sex_Male",
"number_vesselscolored_1.0",         
"number_vesselscolored_2.0",      
"chestPain_type_non-anginal pain",    
"restecg_results_normal",             
"number_vesselscolored_3.0"]]  


# In[158]:



y3=df1['heart_disease']                                       

# Split dataset into training set and test set
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.80, random_state=10) # 80% training and 20% test

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X3_train,y3_train)

# prediction on test set
y3_pred=clf.predict(X3_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y3_test, y3_pred))


# In[148]:


fn=X_train.columns
cn=X_train.columns[-1]


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=1000)


tree.plot_tree(clf.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);


fig.savefig('rf_individualtree.png')


# In[190]:


# Create Decision Tree classifer object
dt1 = DecisionTreeClassifier()

# Train Decision Tree Classifer
dt1 = dt1.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = dt1.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[192]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dt1, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('heart.png')
Image(graph.create_png())


# In[ ]:







