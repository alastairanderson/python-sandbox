'''requirements.txt
beautifulsoup4==4.7.1
lxml==4.3.4

Notes:
- Open in the 'packages/beautifulsoup' folder and venv created in their also.
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
    # response = requests.get(URL, headers=headers)
    response = requests.get(URL)

    if response:
        if response.status_code == 200:
            raw_html = response.content.decode()
            with open(LOCAL_HTML_FILE, "w") as f:
                f.write(html)
                f.close()
#endregion

from bs4 import BeautifulSoup

html_content = BeautifulSoup(raw_html, "lxml")

print(html_content)

# entries_identifier = {
#     "element": "li",
#     "attributes": { "class": "o-teaser-collection__item" }
# }

# all_elements = content.find_all(content_identifier["element"], attrs=content_identifier["attributes"])
