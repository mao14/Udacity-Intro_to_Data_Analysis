import unicodecsv

def read_csv(filename):
    with open(filename,'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

enrollments = read_csv('enrollments.csv')
daily_engagement = read_csv('daily_engagement.csv')
project_submissions = read_csv('project_submissions.csv')


#Treat the data
from datetime import datetime as dt

#Convert string into time
def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%Y-%m-%d')

#Convert string into number
def parse_int(i):
    if i == '':
        return None
    else:
        return int(i)

#Clean up enrollment table
for  enrollment in enrollments:
    enrollment['cancel_date'] = parse_date(enrollment['cancel_date'])
    enrollment['days_to_cancel'] = parse_int(enrollment['days_to_cancel'])
    enrollment['is_canceled'] = enrollment['is_canceled'] == 'True'
    enrollment['is_udacity'] = enrollment['is_udacity'] == 'True'
    enrollment['join_date'] = parse_date(enrollment['join_date'])

#Clean up engagement table
for engagement_record in daily_engagement:
    engagement_record['lessons_completed'] = int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited'] = int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed'] = int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited'] = float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date'] = parse_date(engagement_record['utc_date'])

#Clean up submissions table
for submission in project_submissions:
    submission['completion_date'] = parse_date(submission['completion_date'])
    submission['creation_date'] = parse_date(submission['creation_date'])

# Exercise find the number of total rows and the number of unique students

#Number of rows - enrollments
print (len(enrollments))
#number of unique students - enrollments
unique_enrolled_students = set()

for i in enrollments:
    unique_enrolled_students.add(i['account_key'])
print (len(unique_enrolled_students))


#Number of rows - daily_engagement
print (len(daily_engagement))

#number of unique students - daily_engagement
unique_engaged_students = set()

for i in daily_engagement:
    unique_engaged_students.add(i['acct'])
print (len(unique_engaged_students))

#Number of rows - project_submissions
#print (len(project_submissions))

#number of unique students - project_submissions
unique_submissions = set()

for i in project_submissions:
    unique_submissions.add(i['account_key'])
print (len(unique_submissions))

#rename the column acct to account_key
for i in daily_engagement:
    i['account_key'] = i['acct']
    del i['acct']


## Find the total number of rows and the number of unique students (account keys)
## in each table.

def get_unique_students(data):
    unique_students = set()
    for student in data:
        unique_students.add(student['account_key'])
    return unique_students

len(enrollments)
unique_enrolled_students = get_unique_students(enrollments)
len(unique_enrolled_students)

len(daily_engagement)
unique_engaged_students = get_unique_students(daily_engagement)
len(unique_engaged_students)

len(project_submissions)
unique_projects_submitted = get_unique_students(project_submissions)
len(unique_projects_submitted)

## Missing Engagement Records
for enrollment in enrollments:
    student = enrollment['account_key']
    if student not in unique_engaged_students:
        print enrollment
        break

## Find the number of surprising data points (enrollments missing from
## the engagement table) that remain, if any.

num_problems_students = 0

for enrollment in enrollments:
    student = enrollment['account_key']
    if student not in unique_engaged_students \
            and enrollment['days_to_cancel'] != 0:
        num_problems_students += 1
        print enrollment

# Create a set of the account keys for all Udacity test accounts
udacity_test_accounts = set()
for enrollment in enrollments:
    if enrollment['is_udacity']:
        udacity_test_accounts.add(enrollment['account_key'])
len(udacity_test_accounts)
