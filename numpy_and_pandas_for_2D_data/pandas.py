import pandas as pd
import numpy as np

#Pandas dataframes - every column could be a different type of data, not line numpay arrays
#Another benefit is that dataframes have indexes

enrollments_df = pd.DataFrame({
    'account_key': [448, 448, 448, 448, 448],
    'status': ['canceled','canceled','canceled','canceled','current'],
    'join_date': ['2014-11-10','2014-11-05','2015-01-27','2014-11-05','2015-03-10'],
    'days_to_cancel': [65,5,0,0,np.nan],
    'is_udacity': [True, True, True, True, True]
})

#Now you can take the mean of each numerical column
enrollments_df.mean()

#Exercise
ridership_df = pd.DataFrame({
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
})

# Change False to True for each block of code to see what it does

# DataFrame creation
if False:
    # You can create a DataFrame out of a dictionary mapping column names to values
    df_1 = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print df_1

    # You can also use a list of lists or a 2D NumPy array
    df_2 = pd.DataFrame([[0, 1, 2], [3, 4, 5]], columns=['A', 'B', 'C'])
    print df_2


# Accessing elements
if False:
    print ridership_df.iloc[0]
    print ridership_df.loc['05-05-11']
    print ridership_df['R003']
    print ridership_df.iloc[1, 3]

# Accessing multiple rows
if False:
    print ridership_df.iloc[1:4]

# Accessing multiple columns
if False:
    print ridership_df[['R003', 'R005']]

# Pandas axis
if False:
    df = pd.DataFrame({'A': [0, 1, 2], 'B': [3, 4, 5]})
    print df.sum()
    print df.sum(axis=1)
    print df.values.sum()

def mean_riders_for_max_station(ridership_df):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.

    This is the same as a previous exercise, but this time the
    input is a Pandas DataFrame rather than a 2D NumPy array.
    '''
    max_riders_first_day = ridership_df.iloc[0].argmax()

    overall_mean = ridership_df.values.mean() # Replace this with your code
    mean_for_max = ridership_df[max_riders_first_day].mean() # Replace this with your code

    return (overall_mean, mean_for_max)

#DataFrames are a great way data structure to represent csv's:

subway_df = pd.read_csv('nyc_subway_weather.csv')
subway_df.head() #head just to print the first five lines
subway_df.describe() #to see some statistics about each column

### Exercise about correlation and standarization
subway_df = pd.read_csv('nyc_subway_weather.csv')
def correlation(x, y):
    '''
    Fill in this function to compute the correlation between the two
    input variables. Each input is either a NumPy array or a Pandas
    Series.

    correlation = average of (x in standard units) times (y in standard units)

    Remember to pass the argument "ddof=0" to the Pandas std() function!
    '''

    std_x = (x - x.mean())/ x.std(ddof=0)
    std_y = (y - y.mean())/ y.std(ddof=0)
    return (std_x * std_y).mean()

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

print correlation(entries, rain)
print correlation(entries, temp)
print correlation(rain, temp)
print correlation(entries, cum_entries)

## Instead of using the formula std_x = (x - x.mean())/ x.std(ddof=0) to find the coefficient
## you could use NumPy's corrcoef() function can be used to calculate Pearson's r.

import numpy as np

def correlation(x, y):

    std_n = np.corrcoef(x,y)
    print std_n

print correlation(entries, rain)

### Vectorized operations with Pandas
# Adding DataFrames with the column names
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]})
    print df1 + df2

# Adding DataFrames with overlapping column names
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    df2 = pd.DataFrame({'d': [10, 20, 30], 'c': [40, 50, 60], 'b': [70, 80, 90]})
    print df1 + df2

# Adding DataFrames with overlapping row indexes
if False:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]},
                       index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame({'a': [10, 20, 30], 'b': [40, 50, 60], 'c': [70, 80, 90]},
                       index=['row4', 'row3', 'row2'])
    print df1 + df2

# --- Quiz ---
# Cumulative entries and exits for one station for a few hours.
entries_and_exits = pd.DataFrame({
    'ENTRIESn': [3144312, 3144335, 3144353, 3144424, 3144594,
                 3144808, 3144895, 3144905, 3144941, 3145094],
    'EXITSn': [1088151, 1088159, 1088177, 1088231, 1088275,
               1088317, 1088328, 1088331, 1088420, 1088753]
})

def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits (entries in the first column, exits in the second) and
    return a DataFrame with hourly entries and exits (entries in the
    first column, exits in the second).
    '''
    return entries_and_exits - entries_and_exits.shift(1)

print(get_hourly_entries_and_exits(entries_and_exits))


## applymap() function for Pandas
# DataFrame applymap()
if False:
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [10, 20, 30],
        'c': [5, 10, 15]
    })

    def add_one(x):
        return x + 1

    print df.applymap(add_one)

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio',
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grades(grades):
    '''
    Fill in this function to convert the given DataFrame of numerical
    grades to letter grades. Return a new DataFrame with the converted
    grade.

    The conversion rule is:
        90-100 -> A
        80-89  -> B
        70-79  -> C
        60-69  -> D
        0-59   -> F
    '''
    def grades1(x):
        if x >= 90:
            return 'A'
        elif x >= 80:
            return 'B'
        elif x >= 70:
            return 'C'
        elif x >= 60:
            return 'D'
        else:
            return 'F'

    return grades.applymap(grades1)

print(convert_grades(grades_df))

## apply() function for Pandas

def convert_grades_curve(exam_grades):
    # Pandas has a bult-in function that will perform this calculation
    # This will give the bottom 0% to 10% of students the grade 'F',
    # 10% to 20% the grade 'D', and so on. You can read more about
    # the qcut() function here:
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    return pd.qcut(exam_grades,
                   [0, 0.1, 0.2, 0.5, 0.8, 1],
                   labels=['F', 'D', 'C', 'B', 'A'])

# qcut() operates on a list, array, or Series. This is the
# result of running the function on a single column of the
# DataFrame.
print convert_grades_curve(grades_df['exam1'])

# qcut() does not work on DataFrames, but we can use apply()
# to call the function on each column separately
print grades_df.apply(convert_grades_curve)


def standardize(df):

    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    '''
    std_var = (df - df.mean()) / df.std(ddof=0)
    return std_var

#print standardize(grades_df['exam1']) This is done just for the first column
print grades_df.apply(standardize)

## apply() function for Pandas case 2
import numpy as np
import pandas as pd
df = pd.DataFrame({
    'a' : [4, 5, 3, 1, 2],
    'b' : [20, 10, 40, 50, 30],
    'c' : [25, 20, 5, 15, 10]
})

print df.apply(np.mean)
print df.apply(np.max)

def second_largest_in_column(df):
        '''
        Fill in this function to return the second-largest value of each
        column of the input DataFrame.
        '''
        sorted_column = df.sort_values(ascending=False)
        return sorted_column.iloc[1]

def second_largest(df):
    return df.apply(second_largest_in_column)

print(second_largest(df))

## Adding a DataFrame to a Series
import pandas as pd

# Adding a Series to a square DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })

    print df
    print '' # Create a blank line between outputs
    print df + s

# Adding a Series to a one-row DataFrame

if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})

    print df
    print '' # Create a blank line between outputs
    print df + s

# Adding a Series to a one-column DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10, 20, 30, 40]})

    print df
    print '' # Create a blank line between outputs
    print df + s

# I could you use the .add() function, it allows to sum through columns or rows if you specify.

# df.add(s, axis = 'columns') gets the same result as the previous segment
# df.add(s, axis = 'index') gets each element of the series for each column
# The functions sub(), mul(), div()

print df.add(s, axis = 'index')

# Adding when DataFrame column names match Series index
if False:
    s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })

    print df
    print '' # Create a blank line between outputs
    print df + s

# Adding when DataFrame column names don't match Series index
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })

    print df
    print '' # Create a blank line between outputs
    print df + s

## Group by with pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Examine DataFrame
print example_df

grouped_data = example_df.groupby('even')
# The groups attribute is a dictionary mapping keys to lists of row indexes
print grouped_data.groups

# Group by multiple columns
grouped_data = example_df.groupby(['even', 'above_three'])
print grouped_data.groups

# Get sum of each group
grouped_data = example_df.groupby('even')
print grouped_data.sum()

subway_df = pd.read_csv('nyc_subway_weather.csv')
ridership_by_day = subway_df.groupby('day_week').mean()['ENTRIESn_hourly']
print(ridership_by_day)
