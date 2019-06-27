'''requirements.txt
beautifulsoup4==4.7.1
lxml==4.3.4

Notes:
- PyPi package 'requests' is required for retrieving HTML page with the following code
- Documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
'''

from bs4 import BeautifulSoup

sample_html = '<b class="boldest">Extremely bold</b>'

# Create a BeautifulSoup object with a specified parser (lxml)
soup = BeautifulSoup(sample_html, features="lxml")      # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup
                                                        # Represents the document; Mostly treated as a Tag object. 
                                                        # Supports most of 'Navigating the tree' and 'Searching the tree'.

# soup.name == u'[document]'
# It has no name and no attributes - but has been given a special value for the .name attribute.


#region Kinds of objects

#region Tag
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#tag

tag = soup.b    # type(tag) == bs4.element.Tag
#endregion

#region Name
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#name
tag.name        # 'b'
                
#endregion

#region Attributes
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#attributes

tag['class']    # access value of the class attribute
tag.attrs       # access the dictionary of the attributes

tag['id'] = 'verybold'          # Add / Modify
tag['another-attribute'] = 1    # Add / Modify
del tag['id']                   # Delete
del tag['another-attribute']    # Delete
# tag['id']                     # KeyError: 'id'
tag.get('id')                   # None

#endregion

#region Multi-valued attributes
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#multi-valued-attributes

css_soup = BeautifulSoup('<p class="body"></p>', features="lxml")
css_soup.p['class']             # returned as a list: ["body"]

css_soup = BeautifulSoup('<p class="body strikeout"></p>', features="lxml")
css_soup.p['class']             # returned as a list: ["body", "strikeout"]
                                

# If an attribute looks like it has more than one value, but it’s not a multi-valued attribute 
# as defined by any version of the HTML standard, Beautiful Soup will leave the attribute alone
# id_soup = BeautifulSoup('<p id="my id"></p>')
# id_soup.p['id']
# 'my id'
# NOTE: class, rel, rev, accept-charset, headers, and accesskey accept multiple values as per HTML5 standard

# to get an attribute's values as a list, even if they aren't in a list
# id_list = id_soup.get_attribute_list('id')
# mv_dt_list == ['my id']

#endregion

#region Navigable String
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigablestring
# A string corresponds to a bit of text within a tag

tag.string
# type(tag.string) == bs4.element.NavigableString
# You can’t edit a string in place, but you can replace one string with another
tag.string.replace_with("No longer bold")
# tag == '<b class="boldest">No longer bold</b>'

# supports most of the features described in Navigating the tree and Searching the tree (see below), but not all. 
# In particular, since a string can’t contain anything (the way a tag may contain a string or another tag), strings 
# don’t support the .contents or .string attributes, or the find() method.

# If you want to use a NavigableString outside of Beautiful Soup, you should call str() on it to turn it into a 
# normal Python Unicode string. 
# If you don’t, your string will carry around a reference to the entire Beautiful Soup parse tree, 
# even when you’re done using Beautiful Soup. This is a big waste of memory.
normal_python_string = str(tag.string)
# type(normal_python_string) == str

#endregion

#region Comments and other special strings
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#comments-and-other-special-strings

# Tag, NavigableString, and BeautifulSoup cover almost everything you’ll see in an HTML or XML file, but 
# there are a few leftover bits. The only one you’ll probably ever need to worry about is the comment:

markup = "<b><!--Hey, buddy. Want to buy a used parser?--></b>"
soup = BeautifulSoup(markup, features="lxml")
comment = soup.b.string
# comment == 'Hey, buddy. Want to buy a used parser?'
# type(comment) == <class 'bs4.element.Comment'>

# soup.b.prettify()      # view a prettified version of the <b> tag

# Beautiful Soup defines classes for anything else that might show up in an XML document: 
# CData, ProcessingInstruction, Declaration, and Doctype. 
# Just like Comment, these classes are subclasses of NavigableString that add something extra to the string. 


from bs4 import CData                       # An example that replaces the comment with a CDATA block
cdata = CData("A CDATA block")
comment.replace_with(cdata)

soup.b.prettify()                           # Beautifies the HTML


# StackOverflow - What is CDATA in HTML?
# https://stackoverflow.com/questions/7092236/what-is-cdata-in-html

#endregion

#endregion - Kinds of objects


#region Navigating the tree
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-the-tree

#region The "Three Sisters" HTML document
html_doc = """<html>
        <head><title>The Dormouse's story</title></head>
        <body>
            <p class="title"><b>The Dormouse's story</b></p>
            <p class="story">Once upon a time there were three little sisters; and their names were
            <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
            <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
            <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
            and they lived at the bottom of a well.</p>
            <p class="story">...</p>
        </body>
    </html>"""

soup = BeautifulSoup(html_doc, features="lxml")
#endregion

#region Going down
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-down

#region Navigating using tag names           
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-using-tag-names

soup.head                               # <head><title>The Dormouse's story</title></head>
soup.title                              # <title>The Dormouse's story</title>


soup.body.b                             # You can do use this trick again and again to zoom in on a certain part of the 
                                        # parse tree. 
                                        # This code gets the first <b> tag beneath the <body> tag:
                                        # <b>The Dormouse's story</b>


soup.a                                  # Using a tag name as an attribute will give you only the first tag by that name
                                        # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

#endregion

#region .contents and .children
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children

head_tag = soup.head                    # head_tag == <head><title>The Dormouse's story</title></head>       
head_tag.contents                       # A tag’s children are available in a list called .contents
                                        # ["<title>The Dormouse's story</title>"]

title_tag = head_tag.contents[0]        # title_tag == '<title>The Dormouse's story</title>'
title_tag.contents                      # ["The Dormouse's story"]


len(soup.contents)                      # BeautifulSoup object has children: the HTML tag - Count: 1
soup.contents[0].name                   # 'html'


text = title_tag.contents[0]            # A string does not have .contents, because it can’t contain anything

                                        # text.contents - this would throw the following error:
                                        # AttributeError: 'NavigableString' object has no attribute 'contents'


for child in title_tag.children:        # Iterate over a tag’s children:
    child                               # child == "The Dormouse's story"

#endregion

#region .descendants
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#descendants

for child in head_tag.descendants:      # The .contents and .children attributes only consider a tag’s direct children. 
    child                               # The .descendants attribute iterates over all of a tag’s children, recursively
                                        #     <title>The Dormouse's story</title>
                                        #     The Dormouse's story

len(list(soup.children))                # The BeautifulSoup object only has one direct child (the <html> tag)
len(list(soup.descendants))             # but it has a lot of descendants (28 here)

#endregion

#region .strings
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#string

title_tag.string                        # If a tag has only one child, and that child is a NavigableString, 
                                        # the child is made available as .string:
                                        # 'The Dormouse's story'
                                        # type: <class 'bs4.element.NavigableString'>


head_tag.contents                       # If a tag’s only child is another tag, and that tag has a .string, 
                                        # then the parent tag is considered to have the same .string as its child:
                                        # [<title>The Dormouse's story</title>]

head_tag.string                         # 'The Dormouse's story'


soup.html.string                        # If a tag contains more than one thing, then it’s not clear what .string should 
                                        # refer to, so .string is defined to be None

#endregion

#region .strings and stripped_strings
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#strings-and-stripped-strings

for string in soup.strings:             # If there’s more than one thing inside a tag, you can look at just the strings 
    repr(string)                        

for string in soup.stripped_strings:    # remove whitespace using .stripped_strings generator 
    print(repr(string))                 # strings consisting entirely of whitespace are ignored, and whitespace at the 
                                        # beginning and end of strings is removed.

#endregion

#endregion - Going down

#region Going up
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-up

#region .parent
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parent

title_tag = soup.title                  
title_tag                               # <title>The Dormouse's story</title>

title_tag.parent                        # can access an element’s parent with the .parent attribute
                                        # <head><title>The Dormouse's story</title></head>


title_tag.string.parent                 # The title string itself has a parent: the <title> tag that contains it
                                        # <title>The Dormouse's story</title>
#endregion - .parent

#region .parents
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parents
# iterate over all of an element’s parents

link = soup.a
link                # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

for parent in link.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)

# p
# body
# html
# [document]
# None

#endregion - .parents

#endregion - Going up

#region Going sideways
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-sideways

#region sample html
sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></b></a>", features="lxml")

# <b> and <c> tags are are at the same level and siblings 
# They’re both direct children of the same tag. 

# When a document is pretty-printed, siblings show up at the same indentation level. 
# You can also use this relationship in the code you write.

print(sibling_soup.prettify())

#endregion - sample html

#region .next_sibling and .previous_sibling
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#next-sibling-and-previous-sibling

sibling_soup.b.next_sibling                 # <c>text2</c>
sibling_soup.c.previous_sibling             # <b>text1</b>

sibling_soup.b.previous_sibling             # .previous_sibling is None, because there’s nothing before it the same level
sibling_soup.c.next_sibling                 # Similarly, the <c> tag has a .previous_sibling but no .next_sibling

sibling_soup.b.string                       # u'text1'
sibling_soup.b.string.next_sibling          # None
                                            # strings “text1” and “text2” are not siblings, 
                                            # because they don’t have the same parent



#endregion - .next_sibling and .previous_sibling

#endregion - Going sideways



#endregion - Navigating the tree








#region Searching the tree
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

#endregion


# Example 1: Extract a collection of elements 



# entries_identifier = {
#     "element": "li",
#     "attributes": { "class": "o-teaser-collection__item" }
# }

# all_elements = content.find_all(content_identifier["element"], attrs=content_identifier["attributes"])
