# Talha Rizwan Malik 19I-0652
# Syed Muhammad Ibtisam 19i-0422
# Muhammad Anser Qureshi 19I-0680
import sklearn
from scipy.stats import entropy as en
from sklearn import preprocessing
import csv
import pandas as pd
from sklearn.impute import KNNImputer
# data cleaning through KNN imputation
import pandas as pd
import numpy as np

df = pd.read_csv('derm.csv')
print(type(df))
print(df)
dfs = df.iloc[1:, :]
knn = KNNImputer(n_neighbors=15)
dfs = pd.DataFrame(knn.fit_transform(dfs), columns=dfs.columns)
#dfs.to_csv('derm.0.2.csv')

#finding the min and max value of age
min_value = dfs['age'].min()
max_value = dfs['age'].max()

#binning the age values by means
#creating 6 bins of age
arr = pd.cut(dfs['age'], bins=6)

arr1 = pd.cut(dfs['age'], bins=6).value_counts()
labels = [6.25, 18.75, 31.25, 43.75, 56.25, 68.75 ]

interval_range = pd.interval_range(start=0,freq=12.5,end=75.0,closed='left')
dfs['category_bin_means'] = pd.cut(dfs['age'], bins=np.linspace(0, 75, 7), labels=labels, include_lowest=True)


#sorting the data w.r.t age
sorted_df = dfs. sort_values(by=['age'], ascending=True)
sorted_df.to_csv('derm.0.4.csv')
df = sorted_df

#finding the total entropy of class attribute
a = df['class']

total_row_count=364

a = np.array(a)
data=np.unique(a,return_counts=True)
ent=en(a)
print("the entropy of complete attribute is: ",ent)
print("\n")


#finding different information gains and comparing them


#finding the first information gain
df1 = df.iloc[1:24]
df2 = df.iloc[24:365]

d1 = np.array(df1['class'])
d2 = np.array(df2['class'])

data1=np.unique(d1,return_counts=True)
ent1=en(d1)
#print("the entropy is: ",ent1)

data2=np.unique(d2,return_counts=True)
ent2=en(d2)
#print("the entropy is: ",ent2)

row_count1 = sum(1 for row in d1)
row_count2 = sum(1 for row in d2)

# now finding the information gain
cal_ent = ((row_count1/total_row_count)*ent1)+(row_count2/total_row_count)*ent2
information_gain_first=ent-cal_ent
print("the first information gain is : ",information_gain_first)



#finding the second information gain
df1 = df.iloc[1:98]
df2 = df.iloc[98:365]

d1 = np.array(df1['class'])
d2 = np.array(df2['class'])

data1=np.unique(d1,return_counts=True)
ent1=en(d1)
#print("the entropy is: ",ent1)


data2=np.unique(d2,return_counts=True)
ent2=en(d2)
#print("the entropy is: ",ent2)


row_count1 = sum(1 for row in d1)
row_count2 = sum(1 for row in d2)

# now finding the information gain
cal_ent = ((row_count1/total_row_count)*ent1)+(row_count2/total_row_count)*ent2
information_gain_second=ent-cal_ent
print("the second information gain is : ",information_gain_second)




#finding the third information gain
df1 = df.iloc[1:204]
df2 = df.iloc[204:365]

d1 = np.array(df1['class'])
d2 = np.array(df2['class'])

data1=np.unique(d1,return_counts=True)
ent1=en(d1)
#print("the entropy is: ",ent1)


data2=np.unique(d2,return_counts=True)
ent2=en(d2)
#print("the entropy is: ",ent2)


row_count1 = sum(1 for row in d1)
row_count2 = sum(1 for row in d2)

# now finding the information gain
cal_ent = ((row_count1/total_row_count)*ent1)+(row_count2/total_row_count)*ent2
information_gain_third=ent-cal_ent
print("the third information gain is : ",information_gain_third)




#finding the forth information gain
df1 = df.iloc[1:293]
df2 = df.iloc[293:365]

d1 = np.array(df1['class'])
d2 = np.array(df2['class'])

data1=np.unique(d1,return_counts=True)
ent1=en(d1)
#print("the entropy is: ",ent1)


data2=np.unique(d2,return_counts=True)
ent2=en(d2)
#print("the entropy is: ",ent2)


row_count1 = sum(1 for row in d1)
row_count2 = sum(1 for row in d2)

# now finding the information gain
cal_ent = ((row_count1/total_row_count)*ent1)+(row_count2/total_row_count)*ent2
information_gain_forth=ent-cal_ent
print("the forth information gain is : ",information_gain_forth)



#finding the fifth information gain
df1 = df.iloc[1:354]
df2 = df.iloc[354:365]

d1 = np.array(df1['class'])
d2 = np.array(df2['class'])

data1=np.unique(d1,return_counts=True)
ent1=en(d1)
#print("the entropy is: ",ent1)

data2=np.unique(d2,return_counts=True)
ent2=en(d2)
#print("the entropy is: ",ent2)

row_count1 = sum(1 for row in d1)
row_count2 = sum(1 for row in d2)

# now finding the information gain
cal_ent = ((row_count1/total_row_count)*ent1)+(row_count2/total_row_count)*ent2
information_gain_fifth=ent-cal_ent
print("the fifth information gain is : ",information_gain_fifth)


#as the highest information gain is in the third division so, dividing according
# to third and merging the whole dataframe back togather

df1 = df.iloc[1:204]
df2 = df.iloc[204:365]

df1 = df1.assign(entropy_based_desc='0')
df2 = df2.assign(entropy_based_desc='1')

#cincatinationg the both dataframes to create back the original csv file
df_row = pd.concat([df1, df2])

print(df_row)

df=df_row
df.to_csv('derm.0.4.csv')


################################################ normalization of data


#noramlized the whole csv file for bonus marks
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)


df.columns = [ 'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon', 'polygonal_papules','follicular_papules', 'oral_mucosal_involvement',
              'knee_and_elbow_involvement', 'scalp_involvement', 'family_history', 'melanin_incontinence',
              'eosinophils_in_the_infiltrate', 'pnl_infiltrate', 'fibrosis_of_the_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis',
              'parakeratosis', 'clubbing_of_the_rete_ridges','elongation_of_the_rete_ridges','thinning_of_the_suprapapillary_epidermis','spongiform_pustule','munro_microabcess',
              'focal_hypergranulosis', 'disappearance_of_the_granular_layer','vacuolisation_and_damage_of_basal_layer','spongiosis','saw-tooth_appearance_of_retes',
              'follicular_horn_plug',	'perifollicular_parakeratosis',	'inflammatory_monoluclear_inflitrate','band-like_infiltrate','age','class', 'category_bin_means','entropy_based_desc']
print(df)
df.to_csv('derm.0.5.csv')

##done normalization on the whole dataframe