#importing libraries
import requests
import pandas as pd
import json

with open('token.json', 'r') as file:
    config = json.load(file)
    github_token = config['github_token']

# Authentication to github
headers = {'Authorization': f'{github_token}'}

#URL for GitHub API
base_url = 'https://api.github.com/search/repositories'

# Query parameters to get count of stargazers of min 50
params = {
    'q': 'stars:>50',
    'sort': 'stars',
    'order': 'desc',
    'per_page': 100
}

# List to store repositories data
repo = []

# Fetching 1000 repositories
for page in range(1, 11):  # 10 pages, as 100 items per page
    params['page'] = page
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code == 200:
        repo.extend(response.json()['items'])
    else:
        print(f'Failed to fetch data: {response.status_code}')
        break
print(f"Total repositories fetched: {len(repo)}")

#loading the data into json format
with open('repositories.json', 'w') as f:
      json.dump(repo, f, indent=4) 
