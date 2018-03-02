import numpy as np

# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

# Change False to True for each block of code to see what it does

# Accessing elements
if False:
    print ridership[1, 3]
    print ridership[1:3, 3:5]
    print ridership[1, :]

# Vectorized operations on rows or columns
if False:
    print ridership[0, :] + ridership[1, :]
    print ridership[:, 0] + ridership[:, 1]

# Vectorized operations on entire arrays
if False:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    print a + b

def mean_riders_for_max_station(ridership):
    ## Find the station with the maximum riders on the first day
    max_station = ridership[0, :].argmax()

    ## Find the mean riders per day for that station
    mean_for_max = ridership[: max_station].mean()

    ## Find the overall mean for comparison
    overall_mean = ridership.mean()

    return(overall_mean,mean_for_max)


print ridership.sum(axis=0) # Axis = 0 is columns
print ridership.sum(axis=1) # Axis = 1 is lines

def min_and_max_riders_per_day(ridership):
    '''
    Fill in this function. First, for each subway station, calculate the
    mean ridership per day. Then, out of all the subway stations, return the
    maximum and minimum of these values. That is, find the maximum
    mean-ridership-per-day and the minimum mean-ridership-per-day for any
    subway station.
    '''

    mean_ridership_per_day = ridership.mean(axis=0)
    max_daily_ridership = mean_ridership_per_day.max()     # Replace this with your code
    min_daily_ridership = mean_ridership_per_day.min()     # Replace this with your code

    return (max_daily_ridership, min_daily_ridership)

## Numpy
### In Numpy each element is converted to an string, even if it is a number
enrollments = np.array([
    ['account_key','status','join_date','days_to_cancel','is_udacity'],
    [448,'canceled','2014-11-10','65','True'],
    [448,'canceled','2014-11-05','5','True'],
    [448,'canceled','2015-01-27','0','True'],
    [448,'canceled','2014-11-10','0','True'],
    [448,'current','2015-03-10','np.nan','True'],
])

enrollments[:,3].mean() #Column 3 - every line
#The formula does not work because it is a string
