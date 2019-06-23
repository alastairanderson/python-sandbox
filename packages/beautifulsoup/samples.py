'''requirements.txt
beautifulsoup4==4.7.1
lxml==4.3.4

Notes:
- PyPi package 'requests' is required for retrieving HTML page with the following code
'''

# constants
LOCAL_HTML_FILE = "./sample.html"
URL = "https://www.ft.com/news-feed?page=1"

#region retrieve + save HTML
import os           # built-in Python package
import requests     # https://pypi.org/project/requests/

if os.path.isfile(LOCAL_HTML_FILE):
    file = open(LOCAL_HTML_FILE, "r")
    raw_html = file.read()
    file.close()
else:
    response = requests.get(URL)

    if response:
        if response.status_code == 200:
            raw_html = response.content.decode()
            with open(LOCAL_HTML_FILE, "w") as f:
                f.write(html)
                f.close()
#endregion

from bs4 import BeautifulSoup       # https://www.crummy.com/software/BeautifulSoup/bs4/doc/


# Create a bs object from the raw html with a specified parser (lxml)
soup = BeautifulSoup(raw_html, "lxml")   # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup

# Kinds of objects
## bs4.element.Tag
tag = soup.li

## .name
tag.name

## Attributes
### access value of an individual attribute
tag['class']    # access value of the class attribute
tag.attrs       # access the dictionary of the attributes

### add, remove, and modify a tag’s attributes
tag['id'] = 'verybold'
tag['another-attribute'] = 1

del tag['id']
del tag['another-attribute']

# tag['id']             # KeyError: 'id'
tag.get('id')           # None

## Multi-valued attributes



# Searching the tree
## Filters
### a string
list_items = soup.find_all('li')

#### find_all() returns a list of bs4.element.Tag

### a regex
# --- finds all the tags whose names start with the letter “b”; in this case, the <body> tag and the <b> tag
import re
for tag in soup.find_all(re.compile("^b")):
    print(tag.name)

# --- finds all the tags whose names contain the letter ‘t’
for tag in soup.find_all(re.compile("t")):
    print(tag.name)




# Example 1: Extract a collection of elements 



# entries_identifier = {
#     "element": "li",
#     "attributes": { "class": "o-teaser-collection__item" }
# }

# all_elements = content.find_all(content_identifier["element"], attrs=content_identifier["attributes"])
