
# coding: utf-8

# Read files without pandas, it takes longer an it is more complicated

# ## How to open a csv file using pandas

# In[5]:


## This is how to read a csv file without using pandas

import unicodecsv

def read_csv(filename):

    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

female_completion_rate = read_csv('female_completion_rate.csv')


# In[6]:


def get_unique_countries(data):
    unique_countries= set()
    for country in data:
        unique_countries.add(country['Country'])
    return unique_countries

countries = get_unique_countries(female_completion_rate)
print(len(countries))



# In[11]:


##Now we can do the same process using pandas, it will be faster and easier
import pandas as pd


# In[12]:


female_completion_rate = pd.read_csv('female_completion_rate.csv')


# In[13]:


len(female_completion_rate['Country'].unique())


# ## How to read data using numpy

# In[15]:


import numpy as np


# In[7]:


countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
])


# In[8]:


employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
])


# In[9]:


# You could treat arrays like lists:

print countries[0]
print countries[0:3]


# In[10]:


# Using formulas from numpy

print employment.mean()
print employment.std()
print employment.max()
print employment.sum()


# In[11]:


## For your knowledge:

for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        print 'Country {} has employment {}'.format(country,
                country_employment)


# In[21]:


## How to find the country with max employment like a list

def max_employment(countries, employment):
    max_country = None
    max_employment = 0

    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]

        if country_employment > max_employment:
            max_country = country
            max_employment = country_employment

    return (max_country, max_employment)
max_country1 = max_employment(countries, employment)
print (max_country1)


# In[23]:


## How to find the country with max employment using numpy

def max_employment2(countries, employment):
    i = employment.argmax()
    return (countries[i],employment[i])

max_country2 = max_employment2(countries, employment)
print (max_country2)


# ## How to operate with arrays

# In[24]:


import numpy as np

female_completion = np.array([
    97.35583,  104.62379,  103.02998,   95.14321,  103.69019,
    98.49185,  100.88828,   95.43974,   92.11484,   91.54804,
    95.98029,   98.22902,   96.12179,  119.28105,   97.84627,
    29.07386,   38.41644,   90.70509,   51.7478 ,   95.45072
])

male_completion = np.array([
     95.47622,  100.66476,   99.7926 ,   91.48936,  103.22096,
     97.80458,  103.81398,   88.11736,   93.55611,   87.76347,
    102.45714,   98.73953,   92.22388,  115.3892 ,   98.70502,
     37.00692,   45.39401,   91.22084,   62.42028,   90.66958
])


# In[25]:


## How to find the total education completion for each country

def overall_completion_rate(female_completion, male_completion):
    return (female_completion + male_completion) / 2.


# In[29]:


## How to find the standardized value (standard deviation) for each country

def standardize_data(values):
    standardized_values = (values - values.mean()) / values.std()
    return standardized_values

val = standardize_data(employment)
print val


# ## Numpy Index Arrays

# In[1]:


import numpy as np

## Different exercises

a = np.array([1, 2, 3, 4])
b = np.array([True, True, False, False])

print a[b]
print a[np.array([True, False, True, False])]


a = np.array([1, 2, 3, 2, 1])
b = (a >= 2)

print a[b]
print a[a >= 2]


a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 2, 1])

print b == 2
print a[b == 2]


# In[2]:


time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])


# In[4]:


def mean_time_for_paid_students(time_spent, days_to_cancel):
    return time_spent[days_to_cancel >= 7].mean()

number1 = mean_time_for_paid_students(time_spent, days_to_cancel)
print(number1)


# ## Pandas Series

# Pandas series are like numpy arrays but they allow for more functionalities

# In[14]:


import pandas as pd

countries = ['Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
             'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan',
             'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
             'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia']

life_expectancy_values = [74.7,  75. ,  83.4,  57.6,  74.6,  75.4,  72.3,  81.5,  80.2,
                          70.3,  72.1,  76.4,  68.1,  75.2,  69.8,  79.4,  70.8,  62.7,
                          67.3,  70.6]

gdp_values = [ 1681.61390973,   2155.48523109,  21495.80508273,    562.98768478,
              13495.1274663 ,   9388.68852258,   1424.19056199,  24765.54890176,
              27036.48733192,   1945.63754911,  21721.61840978,  13373.21993972,
                483.97086804,   9783.98417323,   2253.46411147,  25034.66692293,
               3680.91642923,    366.04496652,   1175.92638695,   1132.21387981]

# Life expectancy and gdp data in 2007 for 20 countries

life_expectancy = pd.Series(life_expectancy_values)
gdp = pd.Series(gdp_values)


# In[ ]:


# Accessing elements and slicing

print life_expectancy[0]
print gdp[3:6]


# In[ ]:


# Looping

for country_life_expectancy in life_expectancy:
    print 'Examining life expectancy {}'.format(country_life_expectancy)


# In[ ]:


# Pandas functions

print life_expectancy.mean()
print life_expectancy.std()
print gdp.max()
print gdp.sum()


# In[ ]:


# Vectorized operations and index arrays

if False:
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([1, 2, 1, 2])

    print a + b
    print a * 2
    print a >= 3
    print a[a >= 3]


# In[17]:


''' Fill in this function to calculate the number of data points for which the directions of
variable1 and variable2 relative to the mean are the same, and the number of data points for
which they are different. Direction here means whether each value is above or below its mean'''

def variable_correlation(variable1, variable2):
    both_above = (variable1 > variable1.mean()) & (variable2 > variable2.mean())
    both_below = (variable1 < variable1.mean()) & (variable2 < variable2.mean())

    is_same_direction = both_above | both_below

#You can sum the number of True's
    num_same_direction = is_same_direction.sum()

    num_different_direction = len(variable1) - num_same_direction

    return(num_same_direction,num_different_direction)

result1 = variable_correlation(life_expectancy,gdp)
print(result1)


# ## Benefits of using a pandas series

# In[18]:


import numpy as np
import pandas as pd

a = np.array([1,2,3,4])
s = pd.Series([1,2,3,4])

# One benefit is that pandas has the function describe() to show statistics about the series
s.describe()


# In[23]:


# The main difference is that pandas has something called an index

# This is how to do it with numpy
#countries = np.arrays(['Albania','Algeria','Andorra','Angola'])
#life_expectancy = np.arrays([74.7, 75., 83.4, 57.6])

# This is how to do it with pandas
life_expectancy = pd.Series([74.7, 75., 83.4, 57.6],
                           index=['Albania','Algeria','Andorra','Angola'])

life_expectancy


# In[24]:


#Numpy arrays are like souped-up Python lists
#Pandas Series is like a cross between a list and a dictionary

life_expectancy[0]


# In[26]:


life_expectancy.loc['Angola']


# In[27]:


#Without specifying the index, it will automatically start at 0
# Access elements by position

life_expectancy.iloc[0]


# In[28]:


countries = [
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina',
    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas',
    'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium',
    'Belize', 'Benin', 'Bhutan', 'Bolivia',
    'Bosnia and Herzegovina'
]

employment_values = [
    55.70000076,  51.40000153,  50.5       ,  75.69999695,
    58.40000153,  40.09999847,  61.5       ,  57.09999847,
    60.90000153,  66.59999847,  60.40000153,  68.09999847,
    66.90000153,  53.40000153,  48.59999847,  56.79999924,
    71.59999847,  58.40000153,  70.40000153,  41.20000076
]

# Employment data in 2007 for 20 countries
employment = pd.Series(employment_values, index=countries)

def max_employment(employment):
    max_country = employment.argmax()              # Replace this with your code
    max_value = employment.loc[max_country]        # Replace this with your code

    return (max_country, max_value)


# ## Vectorized operations and Series

# In[ ]:


# Addition when indexes are the same

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print s1 + s2


# In[ ]:


# Indexes have same elements in a different order

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['b', 'd', 'a', 'c'])
print s1 + s2


# In[30]:


# Indexes overlap, but do not have exactly the same elements

s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([10, 20, 30, 40], index=['c', 'd', 'e', 'f'])
sum_result = s1 + s2
print(sum_result)


# In[31]:


#How to drop NaN values
sum_result.dropna()


# In[32]:


#Alternate solution: Treat missing values as 0 before the addition
s1.add(s2, fill_value=0)


# ## Apply() function

# In[11]:


## Helps perform calculation that are not built into pandas
## It takes a series and a function and create a new series by applying the function to every \
## element of the original series

import pandas as pd
states = pd.Series([
    'California'
    'OH'
    'Michigan'
    'NY'
])


# In[12]:


# I want 'CA', 'OH', 'MI', 'NY'
def clean_state(state):
    if len(state) == 2:
        return state
    elif state == 'California':
        return 'CA'
    elif state == 'Michigan':
        return 'MI'


# In[13]:


# Calculation using loops

clean_states = []
for state in states:
    clean_states.append(clean_state(state))
clean_states = pd.Series(clean_states)
print(clean_states)


# In[6]:


## Calculation using appy()
## I give the entire function as an input to apply

clean_states = states.apply(clean_state)
print(clean_states)


# ## Using matplotlib

# In[1]:


import pandas as pd
import seaborn as sns

employment = pd.read_csv('employment_above_15.csv', index_col='Country')
female_completion = pd.read_csv('female_completion_rate.csv', index_col='Country')
male_completion = pd.read_csv('male_completion_rate.csv', index_col='Country')
life_expectancy = pd.read_csv('life_expectancy.csv', index_col='Country')
gdp = pd.read_csv('gdp_per_capita.csv', index_col='Country')


# In[2]:


employment_us = employment.loc['United States']
female_completion_us = female_completion.loc['United States']
male_completion_us = male_completion.loc['United States']
life_expectancy_us = life_expectancy.loc['United States']
gdp_us = gdp.loc['United States']


# In[3]:


get_ipython().magic(u'pylab inline')
employment_us.plot()
