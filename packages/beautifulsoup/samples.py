'''requirements.txt
beautifulsoup4==4.7.1
lxml==4.3.4

Notes:
- PyPi package 'requests' is required for retrieving HTML page with the following code
'''

# constants
# LOCAL_HTML_FILE = "./sample.html"
# URL = "https://www.ft.com/news-feed?page=1"

#region retrieve + save HTML
# import os           # built-in Python package
# import requests     # https://pypi.org/project/requests/

# if os.path.isfile(LOCAL_HTML_FILE):
#     file = open(LOCAL_HTML_FILE, "r")
#     raw_html = file.read()
#     file.close()
# else:
#     response = requests.get(URL)

#     if response:
#         if response.status_code == 200:
#             raw_html = response.content.decode()
#             with open(LOCAL_HTML_FILE, "w") as f:
#                 f.write(html)
#                 f.close()
#endregion

from bs4 import BeautifulSoup       # https://www.crummy.com/software/BeautifulSoup/bs4/doc/

# Create a bs object from the raw html with a specified parser (lxml)
sample_html = '<b class="boldest">Extremely bold</b>'
soup = BeautifulSoup(sample_html, "lxml")     # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup

# Kinds of objects
## Tag
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#tag
tag = soup.b
# type(tag) == bs4.element.Tag

## .name
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#name
tag.name

## Attributes
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#attributes
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
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#multi-valued-attributes

css_soup = BeautifulSoup('<p class="body"></p>')
css_soup.p['class']
# returned as a list: ["body"]

css_soup = BeautifulSoup('<p class="body strikeout"></p>')
css_soup.p['class']     
# returned as a list: ["body", "strikeout"]

# If an attribute looks like it has more than one value, but it’s not a multi-valued attribute 
# as defined by any version of the HTML standard, Beautiful Soup will leave the attribute alone
# id_soup = BeautifulSoup('<p id="my id"></p>')
# id_soup.p['id']
# 'my id'
# NOTE: class, rel, rev, accept-charset, headers, and accesskey accept multiple values as per HTML5 standard

### to get an attribute's values as a list, even if they aren't in a list
# id_list = id_soup.get_attribute_list('id')
# mv_dt_list == ['my id']

## Navigable String
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigablestring
### A string corresponds to a bit of text within a tag

tag.string
# type(tag.string) == bs4.element.NavigableString
# You can’t edit a string in place, but you can replace one string with another
tag.string.replace_with("No longer bold")
# tag == '<b class="boldest">No longer bold</b>'



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
