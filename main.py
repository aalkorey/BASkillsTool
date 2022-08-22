import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import inline as inline
import matplotlib as plt
import numpy as np
import pandas as pd
import json
import string
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot
# %matplotlib inline
from textblob import Word
from textblob import TextBlob
import os
import itertools
import math

# Web Scraping Section

# Set headers for response variable for retrieving indeed homepage html info
headers = {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:102.0) Gecko/20100101 Firefox/102.0'}

no_of_pages = 4
joblist = []
# For loop to retrieve and parse specified job info for each page
for page in range(no_of_pages):
    # Connecting to Canadian indeed page
    url = f'https://ca.indeed.com/jobs?q=business%20analyst&l=Vancouver BC&start={page * 10}'
    # Get request to indeed with headers above
    response = requests.get(url, headers=headers)
    html = response.text
    # Scraping the web
    soup = BeautifulSoup(html, 'html.parser')
    # Outermost entry point of HTML
    outermost_point = soup.find('div', attrs={'id': 'mosaic-zone-jobcards'})
    # UL lists where the data is stored
    for i in outermost_point.find('ul'):
        # Job itle
        job_title = i.find('h2')
        if job_title is not None:
            jobs = job_title.find('span').text
        # Company Name
        if i.find('span', {'class': 'companyName'}) is not None:
            company = i.find('span', {'class': 'companyName'}).text
        # Company location
        if i.find('div', {'class': 'companyLocation'}) is not None:
            location = i.find('div', {'class': 'companyLocation'}).text
        # HREF links: To be used to got o full job descriptions
        if i.find('a') is not None:
            links = i.find('a', {'class': 'jcs-JobTitle'})['href']
        # Salary if available
        if i.find('div', {'class': 'attribute_snippet'}) is not None:
            salary = i.find('div', {'class': 'attribute_snippet'}).text
        else:
            salary = 'No salary listed'
        # Job post date
        if i.find('span', attrs={'class': 'date'}) is not None:
            job_posted_date = i.find('span', attrs={'class': 'date'}).text

        # Put everything together in a list of lists for the default dictionary
        joblist.append([jobs, company, location, salary, job_posted_date, links])

# Put together in a list and make a dictionary with keys and a list of values from above 'indeed_posts'
indeed_dict_list = defaultdict(list)
# Fields for the dataframe
indeed_spec = ['job', 'Company', 'Location', 'Salary', 'Date Posted', 'href']

job_descr_txt = []
# Indeed DF with columns made above and the stored data from scraping
data_table = pd.DataFrame(joblist, columns=indeed_spec)
# Convert Series to a list of strings
indeed_links = list(data_table['href'])
# Iterator will be index value for default_dict_list
for i in range(len(indeed_links)):
    url_href = 'https://ca.indeed.com' + indeed_links[i]
    res = requests.get(url_href, headers=headers)
    html_ = res.text
    soup_ = BeautifulSoup(html_, 'html.parser')

    if soup_.find('div', {'class': 'jobsearch-jobDescriptionText'}) is not None:
        for ii in soup_.find('div', {'class': 'jobsearch-jobDescriptionText'}):
            try:
                job_descr_txt.append([i, ''.join(ii.text.strip())])
            except AttributeError:
                job_descr_txt.append([i, ''])

# Make a dictionary with values as lists
dct_lst = defaultdict(list)
for i in job_descr_txt:
    # Key value pairs for default_dict_list
    dct_lst[i[0]].append(i[1])

dict_lst_jobsDescr = []
# String join: list of lists of strings
for i in dct_lst.values():
    dict_lst_jobsDescr.append(''.join(i))

data_table['summary'] = pd.Series(dict_lst_jobsDescr)
data_table2 = data_table
data_table2 = data_table2.drop('href', axis=1)

# Job Data Analysis Section

test = data_table2
# Make a copy of the original test dataframe so as to append original information to cleaned up job descrption information
test_1 = test
# Delete empty rows
test = test.dropna()
# Delete information other than job description summary to make data cleaning process easier
test = test.drop(['company', 'location', 'salary', 'date'], axis=1)
# Make all of the text lower case
test['summary'] = test['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Remove tabulation and punctuation
test['summary'] = test['summary'].str.replace('[^\w\s]', ' ', regex=True)
# Remove digits
test['summary'] = test['summary'].str.replace('\d+', '', regex=True)
# Remove stop words
stop = stopwords.words('english')
test['summary'] = test['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# After doing some exploratory data visualization (word cloud) found more stop words to remove; remove them here
other_stop_words = ['summary', 'job', 'analyst', 'fraser', 'name', 'dtype', 'delive', 'object', 'heart', 'best',
                    'dtype', 'coast', 'hel', 'BHJOB13022_12315', 'co', 'attendance', 'time', 'exp', 'arelululemon',
                    'apparel', 'vancouver', 'seattle', 'position', 'contract', 'BHJOB13022_12294', 'roleare', 'union',
                    'non', 'category', 'aaps', 'profil', 'II', 'considered', 'gro', 'jm', 'description', 'looking',
                    'roleherschel', 'co', 'llp', 'emc', 'career', 'statement', 'process', 'analystvancouver', 'p',
                    'recognized', 'emergency', 'analyticswho', 'u', 'high', 'remote', 'inters', 'sits', 'fl', 'w',
                    'solid', 'least', 'year', 'overview', 'victoria', 'founded', 'britis', 'powerex', 'wholly',
                    'energy', 'intermediate', 'owned', 'corp', 'junior', 'professional', 'affiliation', 'hris',
                    'fast', 'thrive', 'serv', 'summarydo', 'paced', 'oc', 'analy', 'brand', 'analystwho', 'reporting',
                    'arelulu', 'bo', 'bu', 'immediate', 'imagex', 'managerwho', 'setting', 'work', 'equivalent',
                    'developped', 'ass']
test['summary'] = test['summary'].apply(lambda x: " ".join(x for x in x.split() if x not in other_stop_words))

# List of technical skills to be looked for that I thought of
initial_technical_skills = ['hadoop', 'flask', 'pandas', 'scikit-learn', 'numpy', 'mysql', 'mongodb', 'nltk', 'fastai',
                            'keras',
                            'pytorch', 'tensorflow', 'linux', 'django', 'react', 'reactjs', 'ai', 'ui', 'tableau',
                            'jupyter',
                            'ms office', 'excel', 'word', 'powerpoint', 'nosql', 'data cleaning', 'data preparation',
                            'data exploration', 'statistical skills', 'data visualization', 'reports generation',
                            'domain knowledge', 'spreadsheet software', 'machine learning', 'data warehousing',
                            'project management', 'web scraping', 'data collection', 'exploratory data analysis',
                            'sentiment analysis', 'end-to-end', 'data analysis', 'data mapping',
                            'system mapping', 'data querying', 'data mining', 'data validation', 'data testing',
                            'end to end',
                            'statistical programming', 'report generation', 'power bi', 'use cases', 'wireframes',
                            'data models', 'entity relationship diagrams', 'system context diagrams',
                            'data flow diagrams',
                            'data modeling', 'matplotlib', 'seaborn', 'statsmodels']

# Retrieval and parsing EMSI global skill set to be used
url2 = "https://auth.emsicloud.com/connect/token"
payload1 = "client_id=di1pg7gmbtxlpcb4&client_secret=bd1gdGGO&grant_type=client_credentials&scope=emsi_open"
headers2 = {'Content-Type': 'application/x-www-form-urlencoded'}
response1 = requests.request("POST", url2, data=payload1, headers=headers2)
json_data2 = json.loads(response1.text)
# Getting access token
items1 = []
for value in json_data2.values():
    items1.append(value)
access_token1 = items1[0]
# Retrieving hard skills from EMSI skills list
url3 = "https://emsiservices.com/skills/versions/latest/skills"
querystring1 = {"typeIds": "ST1", "fields": "name"}
headers10 = {'Authorization': "Bearer " + access_token1}
response_ = requests.request("GET", url3, headers=headers10, params=querystring1)
json_data10 = json.loads(response_.text)
skills10 = []
for i in json_data10["data"]:
    for value in i.values():
        skills10.append(value)
for i in range(len(skills10)):
    skills10[i] = skills10[i].lower()

# Combining initial technical skills list with EMSI skills hard skill list minus the duplicated found amongst the two lists
duplicate_technical_skills = list(set(initial_technical_skills).intersection(skills10))

c = set(skills10) - set(duplicate_technical_skills)
d = set(initial_technical_skills) - set(duplicate_technical_skills)
# Final technical skills list to be used
technical_skills = list(c) + list(d)

# Programming languages to be looked for
programming_languages = ['python', 'c', 'r', 'c++', 'java', 'scala', 'spark', 'php', 'sql', 'css', 'stata',
                         'javascript', 'java', 'julia', 'ruby', 'matlab', 'c#', 'sas', 'html', 'rust', 'perl', 'go',
                         'elm', 'typescript', 'swift', 'unity', 'objective-c', 'kotlin', 'node.js', 'bash', 'shell',
                         'powershell', 'dart', 'assembly', 'vba', 'groovy']

# Same process done for technical skills but now for soft skills
url1 = "https://auth.emsicloud.com/connect/token"
payload = "client_id=di1pg7gmbtxlpcb4&client_secret=bd1gdGGO&grant_type=client_credentials&scope=emsi_open"
headers1 = {'Content-Type': 'application/x-www-form-urlencoded'}
response = requests.request("POST", url1, data=payload, headers=headers1)
json_data1 = json.loads(response.text)
items = []
for value in json_data1.values():
    items.append(value)
access_token = items[0]
url = "https://emsiservices.com/skills/versions/latest/skills"
querystring = {"typeIds": "ST2", "fields": "name"}
headers = {'Authorization': "Bearer " + access_token}
response = requests.request("GET", url, headers=headers, params=querystring)
json_data = json.loads(response.text)
s_skills = []
for i in json_data["data"]:
    for value in i.values():
        s_skills.append(value)
for i in range(len(s_skills)):
    s_skills[i] = s_skills[i].lower()

# Finding overlap between skills in the programming languages, technical skills, and soft skills lists with the list of strings from the job summary text
span_1 = 2
span_2 = 3
span_3 = 4
# Make an empty data frame to be filled with job information
output = pd.DataFrame()
for i in range(0, len(test)):
    raw_skills = test['summary'][i].split()

    top_programming_languages = list(set(programming_languages).intersection(raw_skills))

    top_technical_skills_0 = list(set(technical_skills).intersection(raw_skills))

    raw_skills_1 = []
    for j in range(0, len(raw_skills), span_1):
        raw_skills_1.append(" ".join(raw_skills[j:j + span_1]))
    top_technical_skills_1 = list(set(technical_skills).intersection(raw_skills_1))

    # Ensure that all possible pairs of words are analyzed by starting at att posisble points in the initial list  of raw skills
    raw_skills_2 = []
    for k in range(1, len(raw_skills), span_1):
        raw_skills_2.append(" ".join(raw_skills[k:k + span_1]))
    top_technical_skills_2 = list(set(technical_skills).intersection(raw_skills_2))

    raw_skills_3 = []
    for a in range(0, len(raw_skills), span_2):
        raw_skills_3.append(" ".join(raw_skills[a:a + span_2]))
    top_technical_skills_3 = list(set(technical_skills).intersection(raw_skills_3))

    raw_skills_4 = []
    for b in range(1, len(raw_skills), span_2):
        raw_skills_4.append(" ".join(raw_skills[b:b + span_2]))
    top_technical_skills_4 = list(set(technical_skills).intersection(raw_skills_4))

    raw_skills_5 = []
    for c in range(2, len(raw_skills), span_2):
        raw_skills_5.append(" ".join(raw_skills[c:c + span_2]))
    top_technical_skills_5 = list(set(technical_skills).intersection(raw_skills_5))

    top_technical_skills = top_technical_skills_0 + top_technical_skills_1 + top_technical_skills_2 + top_technical_skills_3 + top_technical_skills_4 + top_technical_skills_5

    top_soft_skills_0 = list(set(s_skills).intersection(raw_skills))

    raw_skills_6 = []
    for d in range(0, len(raw_skills), span_1):
        raw_skills_6.append(" ".join(raw_skills[d:d + span_1]))
    top_soft_skills_1 = list(set(s_skills).intersection(raw_skills_6))

    raw_skills_7 = []
    for e in range(1, len(raw_skills), span_1):
        raw_skills_7.append(" ".join(raw_skills[e:e + span_1]))
    top_soft_skills_2 = list(set(s_skills).intersection(raw_skills_7))

    raw_skills_8 = []
    for f in range(0, len(raw_skills), span_2):
        raw_skills_8.append(" ".join(raw_skills[f:f + span_2]))
    top_soft_skills_3 = list(set(s_skills).intersection(raw_skills_8))

    raw_skills_9 = []
    for g in range(1, len(raw_skills), span_2):
        raw_skills_9.append(" ".join(raw_skills[g:g + span_2]))
    top_soft_skills_4 = list(set(s_skills).intersection(raw_skills_9))

    raw_skills_10 = []
    for h in range(2, len(raw_skills), span_2):
        raw_skills_10.append(" ".join(raw_skills[h:h + span_2]))
    top_soft_skills_5 = list(set(s_skills).intersection(raw_skills_10))

    raw_skills_11 = []
    for m in range(0, len(raw_skills), span_3):
        raw_skills_11.append(" ".join(raw_skills[m:m + span_3]))
    top_soft_skills_6 = list(set(s_skills).intersection(raw_skills_11))

    raw_skills_12 = []
    for n in range(1, len(raw_skills), span_3):
        raw_skills_12.append(" ".join(raw_skills[n:n + span_3]))
    top_soft_skills_7 = list(set(s_skills).intersection(raw_skills_12))

    raw_skills_13 = []
    for o in range(2, len(raw_skills), span_3):
        raw_skills_13.append(" ".join(raw_skills[o:o + span_3]))
    top_soft_skills_8 = list(set(s_skills).intersection(raw_skills_13))

    raw_skills_14 = []
    for p in range(3, len(raw_skills), span_3):
        raw_skills_14.append(" ".join(raw_skills[p:p + span_3]))
    top_soft_skills_9 = list(set(s_skills).intersection(raw_skills_14))

    top_soft_skills = top_soft_skills_0 + top_soft_skills_1 + top_soft_skills_2 + top_soft_skills_3 + top_soft_skills_4 + top_soft_skills_5 + top_soft_skills_6 + top_soft_skills_7 + top_soft_skills_8 + top_soft_skills_9

    # Make a dictionary with all the  job information, and append it to the already made empty data frame
    output = output.append({'Job': test['job'][i],
                            'Programming Languages': top_programming_languages,
                            'Technical Skills': top_technical_skills,
                            'Soft Skills': top_soft_skills},
                           ignore_index=True)

# Drop information found in original data frame to avoid duplicates, as well as unnecessary information
test_1 = test_1.drop('summary', axis=1)
test_1 = test_1.drop('job', axis=1)
test_1 = test_1.drop('Unnamed: 0', axis=1)
test_1 = test_1.drop('date', axis=1)
# Concatenate the original and newly made dataframes into one containing all the relevant job information
result = pd.concat([output, test_1], axis=1)
# Adjust the order of information in the final dataframe
result.rename(columns={'Job': 'Job', 'Programming Languages': 'Programming Languages',
                       'Technical Skills': 'Technical Skills', 'Soft Skills': 'Soft Skills',
                       'company': 'Company Name', 'location': 'Job Location',
                       'salary': 'Salary'}, inplace=True)
# Refine column names in final data frame
result = result[["Job", "Company Name", "Job Location", "Programming Languages",
                 "Technical Skills", "Soft Skills", "Salary"]]
# Clean salary information
salary_info = result
job_info = salary_info.drop(['Unnamed: 0', 'Salary'], axis=1)
salary_info = salary_info.drop(
    ['Unnamed: 0', 'Job', 'Company Name', 'Job Location', 'Programming Languages', 'Technical Skills',
     'Soft Skills'], axis=1)

# Strip out extraneous characters
salary_info["Salary"] = salary_info["Salary"].str.replace("\n", "")
salary_info["Salary"] = salary_info["Salary"].str.replace(",", "")
salary_info["Salary"] = salary_info["Salary"].str.replace("$", "")
# Replace different terms used for 'no salary listed' with one common term
salary_info["Salary"] = salary_info["Salary"].str.replace("Full-time", "No salary listed")
salary_info["Salary"] = salary_info["Salary"].str.replace("Temporary", "No salary listed")
salary_info["Salary"] = salary_info["Salary"].str.replace("Temporary +1", "No salary listed")
salary_info["Salary"] = salary_info["Salary"].str.replace("No salary listed +1", "No salary listed")

# Make a new column with pay scale information
salary_info["og_salary_period"] = np.nan
salary_info.loc[salary_info["Salary"].str.contains("year"), "og_salary_period"] = "year"
salary_info.loc[salary_info["Salary"].str.contains("month"), "og_salary_period"] = "month"
salary_info.loc[salary_info["Salary"].str.contains("week"), "og_salary_period"] = "week"
salary_info.loc[salary_info["Salary"].str.contains("day"), "og_salary_period"] = "day"
salary_info.loc[salary_info["Salary"].str.contains("hour"), "og_salary_period"] = "hour"

# Isolate salaries by pay scale into separate data frames
salary_data = salary_info[salary_info["Salary"] != "No salary listed"]
salary_info = salary_info[~salary_info.isin(salary_data)].dropna(how="all")
salary_info["Salary"].replace("No salary listed", np.nan, inplace=True)
salary_info["Salary"].astype('float')
salary_data.dropna(inplace=True)

# Remove string data associated with pay scale
year_salaries = salary_data[salary_data["Salary"].str.contains("year")]
month_salaries = salary_data[salary_data["Salary"].str.contains("month")]
week_salaries = salary_data[salary_data["Salary"].str.contains("week")]
day_salaries = salary_data[salary_data["Salary"].str.contains("day")]
hour_salaries = salary_data[salary_data["Salary"].str.contains("hour")]

year_salaries["Salary"] = year_salaries["Salary"].str.replace("a year", "")
month_salaries["Salary"] = month_salaries["Salary"].str.replace("a month", "")
week_salaries["Salary"] = week_salaries["Salary"].str.replace("a week", "")
day_salaries["Salary"] = day_salaries["Salary"].str.replace("a day", "")
hour_salaries["Salary"] = hour_salaries["Salary"].str.replace("an hour", "")


# Define a function that detects when salary information is provided in a range and returns an average of the two values
def split_sal(i):
    try:
        splt = i.split("–", 1)
        first = float(splt[0])
        second = float(splt[1])
        return (first + second) / 2
    except:
        return float(i)


# Define a function that rounds the numbers to the nearest multiple of 100
def round_num(i):
    return "{:,}".format(round(i, -2))


# Clean and scale salary data
year_salaries["Salary"] = year_salaries["Salary"].apply(split_sal)
year_salaries["Salary"] = year_salaries["Salary"].apply(round_num)

month_salaries["Salary"] = month_salaries["Salary"].apply(split_sal)
month_salaries["Salary"] = month_salaries["Salary"] * 12
month_salaries["Salary"] = month_salaries["Salary"].apply(round_num)

week_salaries["Salary"] = week_salaries["Salary"].apply(split_sal)
week_salaries["Salary"] = week_salaries["Salary"] * 52
week_salaries["Salary"] = week_salaries["Salary"].apply(round_num)

day_salaries["Salary"] = day_salaries["Salary"].apply(split_sal)
day_salaries["Salary"] = day_salaries["Salary"] * 260
day_salaries["Salary"] = day_salaries["Salary"].apply(round_num)

hour_salaries["Salary"] = hour_salaries["Salary"].apply(split_sal)
hour_salaries["Salary"] = hour_salaries["Salary"] * 2080
hour_salaries["Salary"] = hour_salaries["Salary"].apply(round_num)

# Append the various salary dataframes together, adn rejoin them to the original dataframe with all the job info
combined_salaries = pd.concat([year_salaries, month_salaries, week_salaries,
                               day_salaries, hour_salaries], axis=0)
combined_salaries = combined_salaries.drop(['og_salary_period'], axis=1)
combined_salaries.sort_index()
job_info = pd.concat([job_info, combined_salaries], axis=1)
# Rename salary column
job_info.rename(columns={'Salary': 'Averaged Annual Salary in CAD'}, inplace=True)

# Data visualization section

# Visualizations for frequency rankings for each column

# Adjust programming language, technical, and soft skill information so that each iem in each list can analyzed individually
job_info["Programming Languages"] = job_info["Programming Languages"].apply(eval)
job_info["Technical Skills"] = job_info["Technical Skills"].apply(eval)
job_info["Soft Skills"] = job_info["Soft Skills"].apply(eval)

job_info_1 = job_info
job_info_2 = job_info
job_info_3 = job_info

# Define function to conceptualize a specified column as a 2D array to reduce its dimensions from 2 to 1 so pandas functions can be applied
def to_1D(series):
    return pd.Series([x for _list in series for x in _list])


# Bar graph for programming language frequency ranks
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
cont1 = ax.bar(to_1D(job_info["Programming Languages"]).value_counts().index, to_1D(job_info["Programming Languages"]).value_counts().values)
ax.bar_label(cont1, color='xkcd:off white', fontsize=6)
ax.set_xlabel("Programming Language", size=12)
ax.set_ylabel("Frequency", size=12)
ax.set_title("Top Programming Languages", size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.subplots_adjust(bottom=0.25)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('pl_frequency_ranks.png')

# Bar graph for technical skills frequency ranks
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
cont2 = ax.bar(to_1D(job_info["Technical Skills"]).value_counts().index[:19], to_1D(job_info["Technical Skills"]).value_counts().values[:19])
ax.bar_label(cont2, fontsize=5, color='xkcd:off white')
ax.set_xlabel("Technical Skill", size=12)
ax.set_ylabel("Frequency", size=12)
ax.set_title("Top Technical Skills", size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.4)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('ts_frequency_ranks.png')

# Bar graph for soft skills frequency ranks
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
cont3 = ax.bar(to_1D(job_info["Soft Skills"]).value_counts().index[:17], to_1D(job_info["Soft Skills"]).value_counts().values[:17])
ax.bar_label(cont3, fontsize=5, color='xkcd:off white')
ax.set_xlabel("Soft Skill", size=12)
ax.set_ylabel("Frequency", size=12)
ax.set_title("Top Soft Skills", size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.4)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('ss_frequency_ranks.png')

# Bar graph for job type frequency ranks
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
cont4 = ax.bar(job_info['Job'].value_counts().index[:11], job_info['Job'].value_counts().values[:11])
ax.bar_label(cont4, color='xkcd:off white', fontsize=5)
ax.set_xlabel("Job", size=12)
ax.set_ylabel("Frequency", size=12)
ax.set_title("Top Job Types", size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.5)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('jt_frequency_ranks.png')

# Bar graph for hiring company frequency ranks
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
cont4 = ax.bar(job_info['Company Name'].value_counts().index[:15], job_info['Company Name'].value_counts().values[:15])
ax.bar_label(cont4, color='xkcd:off white', fontsize=5)
ax.set_xlabel("Company Name", size=12)
ax.set_ylabel("Frequency", size=12)
ax.set_title("Top Hiring Companies", size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.4)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('hc_frequency_ranks.png')


# Define function to add count and percentages to pieces of pie chart
def label_function(val):
    return f'{val/100 * len(job_info):.0f} ({val:.0f}%)'


# Pie chart for programming language frequency ranks
plt.pyplot.figure(facecolor='xkcd:almost black')
textprops = dict(fontsize=6, color='xkcd:off white')
plt.pyplot.pie(to_1D(job_info["Programming Languages"]).value_counts().values, labels=to_1D(job_info["Programming Languages"]).value_counts().index, autopct=label_function, textprops=textprops)
plt.pyplot.title('Division of Programming Languages Among Jobs', size=10, color='xkcd:off white')
plt.pyplot.savefig('pl_pie_chart.png')

# Donut chart for the same info as pie chart
plt.pyplot.figure(facecolor='xkcd:almost black')
textprops1 = dict(fontsize=6, color='xkcd:off white')
plt.pyplot.pie(to_1D(job_info["Programming Languages"]).value_counts().values, labels=to_1D(job_info["Programming Languages"]).value_counts().index, autopct=label_function, textprops=textprops1, wedgeprops={'width': 0.3})
plt.pyplot.title('Division of Programming Languages Among Jobs', size=10, color='xkcd:off white')
plt.pyplot.savefig('pl_donut_chart.png')

# Visualization for association between programming languages and salary

# Isolate programming language and salary columns
job_info_1 = job_info_1.drop(['Job', 'Company Name', 'Job Location', 'Technical Skills', 'Soft Skills'], axis=1)
# Replace empty lists in programming languages column with NaN values
job_info_1["Programming Languages"] = job_info_1["Programming Languages"].apply(lambda y: np.nan if len(y) == 0 else y)
# Replace empty spaces in salary column with NaN values
job_info_1["Averaged Annual Salary in CAD"].replace('', np.nan, inplace=True)
# Drop all rows with a NaN value in it
job_info_1 = job_info_1.dropna()

# Add all values from programming language column to a list
pls = job_info_1["Programming Languages"].tolist()
# Make it a list of strings instead of a list of lists
pls_1 = list(itertools.chain(*pls))

# Add all values from salary column to a list
salaries = job_info_1["Averaged Annual Salary in CAD"].tolist()
# Make a list of the length of items in each list of programming languages for each row in the programming language column
lengthOfItems = []
for index, row in job_info_1.iterrows():
    lengthOfItems.append(len(row["Programming Languages"]))
# Make a list of each of the salaries, where each salary shows up as many times as the length of the corresponding list of programming languages in each row
orderedSalaries = []
for i in range(len(lengthOfItems)):
    for j in range(lengthOfItems[i]):
        orderedSalaries.append(salaries[i])

# Add ordered salary and programming languages lists to a data frame
pl_salary_association = pd.DataFrame({"Programming Languages": pls_1, "Salaries": orderedSalaries})
# Merge rows with duplicate items in the programming languages column while listing out their associated salaries
pl_salary_association = pl_salary_association["Salaries"].groupby([pl_salary_association['Programming Languages']]).apply(list).reset_index()


# Define a function to return average of numbers in salary list for each language
def average(lst):
    return sum(lst) / len(lst)


# Apply function to get averages
pl_salary_association["Salaries"] = pl_salary_association["Salaries"].apply(average)
pl_salary_association = pl_salary_association.sort_values(by=['Salaries'], ascending=False)

# Make the bar chart for the programming languages and salary association
fig, ax = plt.pyplot.subplots(figsize=(14, 4))
asso = ax.bar(pl_salary_association["Programming Languages"], pl_salary_association["Salaries"])
ax.bar_label(asso, color='xkcd:off white', fontsize=6)
ax.set_xlabel('Programming Languages', size=12)
ax.set_ylabel("Salaries", size=12)
ax.set_title('Association Between Programming Languages and Averaged Salaries', size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.subplots_adjust(bottom=0.25)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('pl_s_association.png')

# Visualization for association between technical skills and salary

job_info_2 = job_info_2.drop(['Job', 'Company Name', 'Job Location', 'Programming Languages', 'Soft Skills'], axis=1)
job_info_2["Averaged Annual Salary in CAD"].replace('', np.nan, inplace=True)
job_info_2 = job_info_2.dropna()

tss = job_info_2["Technical Skills"].tolist()
tss_1 = list(itertools.chain(*tss))

salaries_1 = job_info_2["Averaged Annual Salary in CAD"].tolist()
lengthOfItems_1 = []
for index, row in job_info_2.iterrows():
    lengthOfItems_1.append(len(row["Technical Skills"]))

orderedSalaries_1 = []
for i in range(len(lengthOfItems_1)):
    for j in range(lengthOfItems_1[i]):
        orderedSalaries_1.append(salaries_1[i])

ts_salary_association = pd.DataFrame({"Technical Skills": tss_1, "Salaries": orderedSalaries_1})
ts_salary_association = ts_salary_association["Salaries"].groupby([ts_salary_association['Technical Skills']]).apply(list).reset_index()

ts_salary_association["Salaries"] = ts_salary_association["Salaries"].apply(average)
ts_salary_association = ts_salary_association.sort_values(by=['Salaries'], ascending=False)

fig, ax = plt.pyplot.subplots(figsize=(14, 4))
asso1 = ax.bar(ts_salary_association["Technical Skills"][:10], ts_salary_association["Salaries"][:10])
ax.bar_label(asso1, color='xkcd:off white', fontsize=5)
ax.set_xlabel('Technical Skills', size=12)
ax.set_ylabel("Salaries", size=12)
ax.set_title('Association Between Technical Skills and Averaged Salaries', size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.4)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('ts_s_association.png')

# Visualization for association between soft skills and salary

job_info_3 = job_info_3.drop(['Job', 'Company Name', 'Job Location', 'Programming Languages', 'Technical Skills'], axis=1)
job_info_3["Averaged Annual Salary in CAD"].replace('', np.nan, inplace=True)
job_info_3 = job_info_3.dropna()

sss = job_info_3["Soft Skills"].tolist()
sss_1 = list(itertools.chain(*sss))

salaries_2 = job_info_3["Averaged Annual Salary in CAD"].tolist()
lengthOfItems_2 = []
for index, row in job_info_3.iterrows():
    lengthOfItems_2.append(len(row["Soft Skills"]))

orderedSalaries_2 = []
for i in range(len(lengthOfItems_2)):
    for j in range(lengthOfItems_2[i]):
        orderedSalaries_2.append(salaries_2[i])

ss_salary_association = pd.DataFrame({"Soft Skills": sss_1, "Salaries": orderedSalaries_2})
ss_salary_association = ss_salary_association["Salaries"].groupby([ss_salary_association['Soft Skills']]).apply(list).reset_index()

ss_salary_association["Salaries"] = ss_salary_association["Salaries"].apply(average)
ss_salary_association = ss_salary_association.sort_values(by=['Salaries'], ascending=False)

fig, ax = plt.pyplot.subplots(figsize=(14, 4))
asso2 = ax.bar(ss_salary_association["Soft Skills"][:10], ss_salary_association["Salaries"][:10])
ax.bar_label(asso2, color='xkcd:off white', fontsize=5)
ax.set_xlabel('Soft Skills', size=12)
ax.set_ylabel("Salaries", size=12)
ax.set_title('Association Between Soft Skills and Averaged Salaries', size=14)
ax.spines['bottom'].set_color('xkcd:off white')
ax.spines['top'].set_color('xkcd:off white')
ax.spines['right'].set_color('xkcd:off white')
ax.spines['left'].set_color('xkcd:off white')
ax.xaxis.label.set_color('xkcd:off white')
ax.yaxis.label.set_color('xkcd:off white')
ax.title.set_color('xkcd:off white')
ax.tick_params(axis='x', colors='xkcd:off white')
ax.tick_params(axis='y', colors='xkcd:off white')
ax.set_facecolor('xkcd:almost black')
fig.patch.set_facecolor('xkcd:almost black')
plt.pyplot.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.pyplot.subplots_adjust(bottom=0.4)
plt.pyplot.margins(x=0)
plt.pyplot.savefig('ss_s_association.png')


# Report generation section

# Set up multiple variables to store the titles, text within the report
page_title_text = 'Summary of Results Report'
title_text = 'Qualifications Summary for the Role of a Business Analyst'
main_df_text = 'All Data Pertaining to Each Job'
association_df1_text = 'Data Pertaining to Association Between Programming Languages and Salary'
association_df2_text = 'Data Pertaining to Association Between Technical Skills and Salary'
association_df3_text = 'Data Pertaining to Association Between Soft Skills and Salary'

# Combine variables in a long f-string
html = f'''
    <html>
        <head>
            <title>{page_title_text}</title>
        <head>
        <body>
            <h1>{title_text}</h1>
            <img src='pl_frequency_ranks.png' width="700">
            <img src='ts_frequency_ranks.png' width="700">
            <img src='ss_frequency_ranks.png' width="700">
            <img src='jt_frequency_ranks.png' width="700">
            <img src='pl_pie_chart.png' width="700">
            <img src='pl_donut_chart.png' width="700">
            <img src='hc_frequency_ranks.png' width="700">
            <img src='pl_s_association.png' width="700">
            <img src='ts_s_association.png' width="700">
            <img src='ss_s_association.png' width="700">
            <h2>{main_df_text}</h2>
            {job_info.to_html()}
            <h2>{association_df1_text}</h2>
            {pl_salary_association.to_html()}
            <h2>{association_df2_text}</h2>
            {ts_salary_association.to_html()}
            <h2>{association_df3_text}</h2>
            {ss_salary_association.to_html()}
        <body>
    <html>
    '''

# Write the html string as an HTML file, thus generating the final report
with open('jobs_description_html_report_final.html', 'w') as f:
    f.write(html)
