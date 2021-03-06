'''requirements.txt
beautifulsoup4==4.7.1
lxml==4.3.4

Notes:
- PyPi package 'requests' is required for retrieving HTML page with the following code
- Documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Release Notes: https://bazaar.launchpad.net/%7Eleonardr/beautifulsoup/bs4/view/head:/CHANGELOG
- For the 'Differences between parsers', we need to 'pip install html5lib'
'''

from bs4 import BeautifulSoup

sample_html = '<b class="boldest">Extremely bold</b>'

'''
Create a BeautifulSoup object with a specified parser (lxml)
'''
soup = BeautifulSoup(sample_html, features="lxml")      # https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup
                                                        # Represents the document; Mostly treated as a Tag object. 
                                                        # Supports most of 'Navigating the tree' and 'Searching the tree'.
'''
soup.name == u'[document]'
It has no name and no attributes - but has been given a special value for the .name attribute.
'''
#region Kinds of objects

#region Tag
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#tag
'''
tag = soup.b    # type(tag) == bs4.element.Tag

#endregion

#region Name
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#name
'''
tag.name        # 'b'
                
#endregion

#region Attributes
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#attributes
'''
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
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#multi-valued-attributes
'''
css_soup = BeautifulSoup('<p class="body"></p>', features="lxml")
css_soup.p['class']             # returned as a list: ["body"]

css_soup = BeautifulSoup('<p class="body strikeout"></p>', features="lxml")
css_soup.p['class']             # returned as a list: ["body", "strikeout"]
                                

'''
If an attribute looks like it has more than one value, but it’s not a multi-valued attribute 
as defined by any version of the HTML standard, Beautiful Soup will leave the attribute alone

    id_soup = BeautifulSoup('<p id="my id"></p>')
    id_soup.p['id']
    # 'my id'

NOTE: class, rel, rev, accept-charset, headers, and accesskey accept multiple values as per HTML5 standard

to get an attribute's values as a list, even if they aren't in a list

    id_list = id_soup.get_attribute_list('id')
    mv_dt_list == ['my id']
'''

#endregion

#region Navigable String
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigablestring
# A string corresponds to a bit of text within a tag

tag.string
'''
type(tag.string) == bs4.element.NavigableString
You can’t edit a string in place, but you can replace one string with another...
'''

tag.string.replace_with("No longer bold")
'''
tag == '<b class="boldest">No longer bold</b>'

supports most of the features described in Navigating the tree and Searching the tree (see below), but not all. 
In particular, since a string can’t contain anything (the way a tag may contain a string or another tag), strings 
don’t support the .contents or .string attributes, or the find() method.

If you want to use a NavigableString outside of Beautiful Soup, you should call str() on it to turn it into a 
normal Python Unicode string. 

If you don’t, your string will carry around a reference to the entire Beautiful Soup parse tree, 
even when you’re done using Beautiful Soup. This is a big waste of memory.
'''

normal_python_string = str(tag.string)
# type(normal_python_string) == str

#endregion

#region Comments and other special strings
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#comments-and-other-special-strings

Tag, NavigableString, and BeautifulSoup cover almost everything you’ll see in an HTML or XML file, but 
there are a few leftover bits. The only one you’ll probably ever need to worry about is the comment:
'''

markup = "<b><!--Hey, buddy. Want to buy a used parser?--></b>"
soup = BeautifulSoup(markup, features="lxml")
comment = soup.b.string
'''
# comment == 'Hey, buddy. Want to buy a used parser?'
# type(comment) == <class 'bs4.element.Comment'>
'''

soup.b.prettify()      # view a prettified version of the <b> tag
'''
Beautiful Soup defines classes for anything else that might show up in an XML document: 
CData, ProcessingInstruction, Declaration, and Doctype. 
Just like Comment, these classes are subclasses of NavigableString that add something extra to the string. 
'''

from bs4 import CData                       # An example that replaces the comment with a CDATA block
cdata = CData("A CDATA block")
comment.replace_with(cdata)

soup.b.prettify()                           # Beautifies the HTML

'''
StackOverflow - What is CDATA in HTML?
https://stackoverflow.com/questions/7092236/what-is-cdata-in-html
'''
#endregion

#endregion - Kinds of objects


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
#endregion - The "Three Sisters" HTML document


#region Navigating the tree
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-the-tree
'''

#region Going down
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-down
'''

#region Navigating using tag names
'''          
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#navigating-using-tag-names
'''
soup.head       # <head><title>The Dormouse's story</title></head>
soup.title      # <title>The Dormouse's story</title>


soup.body.b     # You can do use this trick again and again to zoom in on a certain part of the 
                # parse tree. 
                # This code gets the first <b> tag beneath the <body> tag:
                # <b>The Dormouse's story</b>


soup.a          # Using a tag name as an attribute will give you only the first tag by that name
                # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
#endregion

#region .contents and .children
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#contents-and-children
'''
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
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#descendants
'''
for child in head_tag.descendants:      # The .contents and .children attributes only consider a tag’s direct children. 
    child                               # The .descendants attribute iterates over all of a tag’s children, recursively
                                        #     <title>The Dormouse's story</title>
                                        #     The Dormouse's story

len(list(soup.children))                # The BeautifulSoup object only has one direct child (the <html> tag)
len(list(soup.descendants))             # but it has a lot of descendants (28 here)

#endregion

#region .strings
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#string
'''
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
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#strings-and-stripped-strings
'''
for string in soup.strings:             # If there’s more than one thing inside a tag, you can look at just the strings 
    repr(string)                        

for string in soup.stripped_strings:    # remove whitespace using .stripped_strings generator 
    repr(string)                        # strings consisting entirely of whitespace are ignored, and whitespace at the 
                                        # beginning and end of strings is removed.
#endregion

#endregion - Going down

#region Going up
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-up
'''
#region .parent
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parent
'''
title_tag = soup.title                  
title_tag                               # <title>The Dormouse's story</title>

title_tag.parent                        # can access an element’s parent with the .parent attribute
                                        # <head><title>The Dormouse's story</title></head>


title_tag.string.parent                 # The title string itself has a parent: the <title> tag that contains it
                                        # <title>The Dormouse's story</title>
#endregion - .parent

#region .parents
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parents
'''
link = soup.a
link                                # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

for parent in link.parents:         # iterate over all of an element’s parents
    if parent is None:              # p
        parent                      # body
    else:                           # html
        parent.name                 # [document]
                                    # None
#endregion - .parents

#endregion - Going up

#region Going sideways
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-sideways
'''
#region sample html
sibling_soup = BeautifulSoup("<a><b>text1</b><c>text2</c></b></a>", features="lxml")

'''
<b> and <c> tags are are at the same level and siblings 
They’re both direct children of the same tag. 

When a document is pretty-printed, siblings show up at the same indentation level. 
You can also use this relationship in the code you write.
'''
sibling_soup.prettify()

#endregion - sample html

#region .next_sibling and .previous_sibling
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#next-sibling-and-previous-sibling
'''
sibling_soup.b.next_sibling                 # <c>text2</c>
sibling_soup.c.previous_sibling             # <b>text1</b>

sibling_soup.b.previous_sibling             # .previous_sibling is None, because there’s nothing before it the same level
sibling_soup.c.next_sibling                 # Similarly, the <c> tag has a .previous_sibling but no .next_sibling

sibling_soup.b.string                       # u'text1'
sibling_soup.b.string.next_sibling          # None
                                            # strings “text1” and “text2” are not siblings, 
                                            # because they don’t have the same parent
'''
Going back to the “three sisters” document
      <a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
      <a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
      <a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
You might think that the .next_sibling of the first <a> tag would be the second <a> tag. 
But actually, it’s a string: the comma and newline that separate the first <a> tag from the second:
'''
link = soup.a
link                                        # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
link.next_sibling                           # u',\n'

link.next_sibling.next_sibling              # The second <a> tag is actually the .next_sibling of the comma
                                            # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>

#endregion - .next_sibling and .previous_sibling

#region .next_siblings and .previous_siblings
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#next-siblings-and-previous-siblings
'''
for sibling in soup.a.next_siblings:
    repr(sibling)                           # u',\n'
                                            # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
                                            # u' and\n'
                                            # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
                                            # u'; and they lived at the bottom of a well.'
                                            # None

for sibling in soup.find(id="link3").previous_siblings:
    repr(sibling)                           # ' and\n'
                                            # <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
                                            # u',\n'
                                            # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
                                            # u'Once upon a time there were three little sisters; and their names were\n'
                                            # None

#endregion - .next_siblings and .previous_siblings

#endregion - Going sideways

#region Going back and forth
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#going-back-and-forth
'''
#region Intro
'''
    look at the beginning of the “three sisters” document:
          <html><head><title>The Dormouse's story</title></head>
          <p class="title"><b>The Dormouse's story</b></p>

    An HTML parser takes this string of characters and turns it into a series of events: 
          - “open an <html> tag”, 
          - “open a <head> tag”, 
          - “open a <title> tag”, 
          - “add a string”, 
          - “close the <title> tag”, 
          - “open a <p> tag”, and so on. 

    Beautiful Soup offers tools for reconstructing the initial parse of the document.
'''
#endregion Intro

#region .next_element and .previous_element
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#next-element-and-previous-element

The .next_element attribute of a string or tag points to whatever was parsed immediately afterwards. 
It might be the same as .next_sibling, but it’s usually drastically different.

Here’s the final <a> tag in the “three sisters” document. Its .next_sibling is a string: 
the conclusion of the sentence that was interrupted by the start of the <a> tag.:
'''
last_a_tag = soup.find("a", id="link3")
last_a_tag                                      # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>
last_a_tag.next_sibling                         # '; and they lived at the bottom of a well.'

'''
But the .next_element of that <a> tag, the thing that was parsed immediately after the <a> tag, 
is not the rest of that sentence: it’s the word “Tillie”:
'''
last_a_tag.next_element                             # u'Tillie'

'''
That’s because in the original markup, the word “Tillie” appeared before that semicolon.
The parser encountered an <a> tag, then the word “Tillie”, then the closing </a> tag, then the semicolon and 
rest of the sentence. The semicolon is on the same level as the <a> tag, but the word “Tillie” was encountered first.

The .previous_element attribute is the exact opposite of .next_element. 
It points to whatever element was parsed immediately before this one:
'''
last_a_tag.previous_element                         # u' and\n'
last_a_tag.previous_element.next_element            # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

#endregion .next_element and .previous_element

#region .next_elements and .previous_elements
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#next-elements-and-previous-elements
'''
for element in last_a_tag.next_elements:
    repr(element)                                   # u'Tillie'
                                                    # u';\nand they lived at the bottom of a well.'
                                                    # u'\n\n'
                                                    # <p class="story">...</p>
                                                    # u'...'
                                                    # u'\n'
                                                    # None

#endregion - .next_elements and .previous_elements

#endregion Going back and forth

#endregion - Navigating the tree


#region Searching the tree
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree
'''
#region Kinds of filters
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#kinds-of-filters
'''
#region A string
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#a-string
'''
soup.find_all('b')      # perform a match against that exact string. 
                        # This code finds all the <b> tags in the document
                        # [<b>The Dormouse's story</b>]

                        # If you pass in a byte string, Beautiful Soup will assume the string is encoded 
                        # as UTF-8. You can avoid this by passing in a Unicode string instead.

                        # NOTE: find_all() returns a list of bs4.element.Tag
#endregion - A string

#region A regular expression
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#a-regular-expression
'''
import re
for tag in soup.find_all(re.compile("^b")):     # finds all the tags whose names start with the letter “b”; 
    tag.name                                    # in this case, the <body> tag and the <b> tag
                                                
                                                #       body
                                                #       b


for tag in soup.find_all(re.compile("t")):      # finds all the tags whose names contain the letter ‘t’
    tag.name
                                                #       html
                                                #       title     
#endregion - A regular expression

#region A list
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#a-list
'''
soup.find_all(["a", "b"])               # [<b>The Dormouse's story</b>,
                                        #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                        #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                        #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
#endregion - A list

#region True
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#true
'''
for tag in soup.find_all(True):         # The value True matches everything it can. 
    tag.name                            # This code finds all the tags in the document, but none of the text strings

                                        #       html
                                        #       head
                                        #       title
                                        #       body
                                        #       p
                                        #       b ...
#endregion - True

#region A function
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#a-function

define a function that takes an element as its only argument. 
The function should return True if the argument matches, and False otherwise.

returns True if a tag defines the “class” attribute but doesn’t define the “id” attribute
'''
def has_class_but_no_id(tag):           
    return tag.has_attr('class') and not tag.has_attr('id')

soup.find_all(has_class_but_no_id)          # [<p class="title"><b>The Dormouse's story</b></p>,
                                            #  <p class="story">Once upon a time there were...</p>,
                                            #  <p class="story">...</p>,
                                            # ...]
'''
If you pass in a function to filter on a specific attribute like href, 
the argument passed into the function will be the attribute value, not the whole tag. 
Here’s a function that finds all a tags whose href attribute does not match a regular expression
'''
def not_lacie(href):
    return href and not re.compile("lacie").search(href)

soup.find_all(href=not_lacie)               # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                            #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

'''
The function can be as complicated as you need it to be. 
Here’s a function that returns True if a tag is surrounded by string objects:
'''
from bs4 import NavigableString
def surrounded_by_strings(tag):
    return (isinstance(tag.next_element, NavigableString)
            and isinstance(tag.previous_element, NavigableString))

for tag in soup.find_all(surrounded_by_strings):
    tag.name                                # body, p, a, a, a, p

#endregion - A function

#region find_all()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-all

Signature: find_all(name, attrs, recursive, string, limit, **kwargs)

The find_all() method looks through a tag’s descendants and retrieves all descendants that match your filters.
'''
soup.find_all("title")                      # [<title>The Dormouse's story</title>]

soup.find_all("p", "title")                 # [<p class="title"><b>The Dormouse's story</b></p>]

soup.find_all("a")                          # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                            #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                            #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.find_all(id="link2")                   # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

import re
soup.find(string=re.compile("sisters"))     # u'Once upon a time there were three little sisters; and their names were\n'

#region The name argument
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-name-argument

Pass in a value for name and you’ll tell Beautiful Soup to only consider tags with certain names. 
Text strings will be ignored, as will tags whose names that don’t match.

Simplest form below, recall from 'Kinds of filters' (above) that you can pass in a string,
regex, list, function or the value 'true'
'''
soup.find_all("title")                      # [<title>The Dormouse's story</title>]

#endregion - The name argument

#region The keyword arguments
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-keyword-arguments

Any argument that’s not recognized will be turned into a filter on one of a tag’s attributes. 
If you pass in a value for an argument called id, Beautiful Soup will filter against each tag’s ‘id’ attribute
'''
soup.find_all(id='link2')                   # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

'''
If you pass in a value for href, Beautiful Soup will filter against each tag’s ‘href’ attribute
'''
soup.find_all(href=re.compile("elsie"))     # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

'''
You can filter an attribute based on a string, a regular expression, a list, a function, or the value True.
The following finds all tags whose id attribute has a value, regardless of what the value is
'''
soup.find_all(id=True)                      # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                            #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                            #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

'''
You can filter multiple attributes at once by passing in more than one keyword argument:
'''
soup.find_all(href=re.compile("elsie"), id='link1')
                                            # [<a class="sister" href="http://example.com/elsie" id="link1">three</a>]

'''
Some attributes, like the data-* attributes in HTML 5, have names that can’t be used as the names of keyword arguments
'''
data_soup = BeautifulSoup('<div data-foo="value">foo!</div>', features="lxml")
# data_soup.find_all(data-foo="value")      # SyntaxError: keyword can't be an expression

'''
You can use these attributes in searches by putting them into a dictionary and passing the dictionary into find_all() 
as the attrs argument:
'''
data_soup.find_all(attrs={"data-foo": "value"})
                                            # [<div data-foo="value">foo!</div>]

'''
You can’t use a keyword argument to search for HTML’s ‘name’ element, because Beautiful Soup uses the name argument 
to contain the name of the tag itself. Instead, you can give a value to ‘name’ in the attrs argument:
'''
name_soup = BeautifulSoup('<input name="email"/>', features="lxml")

name_soup.find_all(name="email")                # []
name_soup.find_all(attrs={"name": "email"})     # [<input name="email"/>]

#endregion - The keyword arguments

#region Searching by CSS class
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-by-css-class

It’s very useful to search for a tag that has a certain CSS class, but the name of the CSS attribute, “class”, 
is a reserved word in Python. 

Using class as a keyword argument will give you a syntax error. As of Beautiful Soup 4.1.2, you can search by 
CSS class using the keyword argument class_
'''
soup.find_all("a", class_="sister")         # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                            #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                            #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

'''
As with any keyword argument, you can pass class_ a string, a regular expression, a function, or True
'''
soup.find_all(class_=re.compile("itl"))     # [<p class="title"><b>The Dormouse's story</b></p>]

def has_six_characters(css_class):
    return css_class is not None and len(css_class) == 6

soup.find_all(class_=has_six_characters)    # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                            #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                            #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
'''
Remember that a single tag can have multiple values for its “class” attribute. 
When you search for a tag that matches a certain CSS class, you’re matching against any of its CSS classes
'''
css_soup = BeautifulSoup('<p class="body strikeout"></p>', features="lxml")
css_soup.find_all("p", class_="strikeout")          # [<p class="body strikeout"></p>]
css_soup.find_all("p", class_="body")               # [<p class="body strikeout"></p>]

'''
You can also search for the exact string value of the class attribute
'''
css_soup.find_all("p", class_="body strikeout")     # [<p class="body strikeout"></p>]

'''
But searching for variants of the string value won’t work:
'''
css_soup.find_all("p", class_="strikeout body")     # []

'''
If you want to search for tags that match two or more CSS classes, you should use a CSS selector
'''
css_soup.select("p.strikeout.body")                 # [<p class="body strikeout"></p>]

'''
In older versions of Beautiful Soup, which don’t have the class_ shortcut, you can use the attrs trick mentioned above.
Create a dictionary whose value for “class” is the string (or regular expression, or whatever) you want to search for
'''
soup.find_all("a", attrs={"class": "sister"})       # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
#endregion - Searching by CSS class

#region - The string argument
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-string-argument

With string you can search for strings instead of tags. As with name and the keyword arguments, 
you can pass in a string, a regular expression, a list, a function, or the value True. 
'''
soup.find_all(string="Elsie")                           # [u'Elsie']


soup.find_all(string=["Tillie", "Elsie", "Lacie"])      # [u'Elsie', u'Lacie', u'Tillie']


soup.find_all(string=re.compile("Dormouse"))            # [u"The Dormouse's story", u"The Dormouse's story"]


def is_the_only_string_within_a_tag(s):
    """Return True if this string is the only child of its parent tag."""
    return (s == s.parent.string)

soup.find_all(string=is_the_only_string_within_a_tag)   # [u"The Dormouse's story", u"The Dormouse's story", 
                                                        #  u'Elsie', u'Lacie', u'Tillie', u'...']

'''
Although string is for finding strings, you can combine it with arguments that find tags: 
Beautiful Soup will find all tags whose .string matches your value for string. 

This code finds the <a> tags whose .string is “Elsie”:
'''
soup.find_all("a", string="Elsie")              # [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]

'''
The string argument is new in Beautiful Soup 4.4.0. In earlier versions it was called text
'''
soup.find_all("a", text="Elsie")                # [<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>]

#endregion - The string argument

#region The limit argument
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-limit-argument

find_all() returns all the tags and strings that match your filters. This can take a while if the document is large. 
If you don’t need all the results, you can pass in a number for limit. 
This works just like the LIMIT keyword in SQL. 
It tells Beautiful Soup to stop gathering results after it’s found a certain number.

There are three links in the “three sisters” document, but this code only finds the first two
'''
soup.find_all("a", limit=2)                     # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                                #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]
#endregion - The limit argument

#region The recursive argument
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-recursive-argument

If you call mytag.find_all(), Beautiful Soup will examine all the descendants of mytag: 
its children, its children’s children, and so on. If you only want Beautiful Soup to consider direct children, 
you can pass in recursive=False. See the difference here:
'''
soup.html.find_all("title")                     # [<title>The Dormouse's story</title>]
soup.html.find_all("title", recursive=False)    # []

'''
The <title> tag is beneath the <html> tag, but it’s not directly beneath the <html> tag: the <head> tag is in the way. 
Beautiful Soup finds the <title> tag when it’s allowed to look at all descendants of the <html> tag, but 
when recursive=False restricts it to the <html> tag’s immediate children, it finds nothing.

Beautiful Soup offers a lot of tree-searching methods (covered below), and they mostly take the same arguments 
as find_all(): name, attrs, string, limit, and the keyword arguments. But the recursive argument is different: 
find_all() and find() are the only methods that support it. Passing recursive=False into a method like find_parents() 
wouldn’t be very useful.
'''
#endregion - The recursive argument

#region - Calling a tag is like calling find_all()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#calling-a-tag-is-like-calling-find-all

Because find_all() is the most popular method in the Beautiful Soup search API, you can use a shortcut for it. 
If you treat the BeautifulSoup object or a Tag object as though it were a function, then it’s the same as 
calling find_all() on that object. These two lines of code are equivalent:
'''
soup.find_all("a")
soup("a")

'''
These two lines are also equivalent:
'''
soup.title.find_all(string=True)
soup.title(string=True)

#endregion - Calling a tag is like calling find_all()

#endregion - find_all()

#region find()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find

Signature: find(name, attrs, recursive, string, **kwargs)

The find_all() method scans the entire document looking for results, but sometimes you only want to find one result. 
If you know a document only has one <body> tag, it’s a waste of time to scan the entire document looking for more. 
Rather than passing in limit=1 every time you call find_all, you can use the find() method. These two lines of code 
are nearly equivalent:
'''
soup.find_all('title', limit=1)             # [<title>The Dormouse's story</title>]
soup.find('title')                          # <title>The Dormouse's story</title>

'''
The only difference is that find_all() returns a list containing the single result, and find() just returns the result.

If find_all() can’t find anything, it returns an empty list. If find() can’t find anything, it returns None:
'''
soup.find("nosuchtag")                     # None

'''
Remember the soup.head.title trick from Navigating using tag names? That trick works by repeatedly calling find():
'''
soup.head.title                             # <title>The Dormouse's story</title>
soup.find("head").find("title")             # <title>The Dormouse's story</title>

#endregion - find()

#region find_parents() and find_parent()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-parents-and-find-parent

Signature: find_parents(name, attrs, string, limit, **kwargs)

Signature: find_parent(name, attrs, string, **kwargs)

find_parents() and find_parent(). Remember that find_all() and find() work their way down the tree, 
looking at tag’s descendants. These methods do the opposite: they work their way up the tree, 
looking at a tag’s (or a string’s) parents. Let’s try them out, starting from a string buried deep in 
the “three daughters” document:
'''
a_string = soup.find(string="Lacie")
a_string                                # u'Lacie'

a_string.find_parents("a")              # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

a_string.find_parent("p")       # <p class="story">Once upon a time there were three little sisters; and their names were
                                #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
                                #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
                                #  and they lived at the bottom of a well.</p>

a_string.find_parents("p", class_="title")   # []

'''
One of the three <a> tags is the direct parent of the string in question, so our search finds it. 
One of the three <p> tags is an indirect parent of the string, and our search finds that as well. 
There’s a <p> tag with the CSS class “title” somewhere in the document, but it’s not one of this string’s parents, 
so we can’t find it with find_parents().

You may have made the connection between find_parent() and find_parents(), and the .parent and .parents attributes 
mentioned earlier. The connection is very strong. These search methods actually use .parents to iterate over all the 
parents, and check each one against the provided filter to see if it matches.
'''
#endregion - find_parents() and find_parent()

#region find_next_siblings() and find_next_sibling()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-next-siblings-and-find-next-sibling

Signature: find_next_siblings(name, attrs, string, limit, **kwargs)

Signature: find_next_sibling(name, attrs, string, **kwargs)

These methods use .next_siblings to iterate over the rest of an element’s siblings in the tree. 
The find_next_siblings() method returns all the siblings that match, and find_next_sibling() only returns the first one:
'''
first_link = soup.a
first_link                              # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_next_siblings("a")      # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                        #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

first_story_paragraph = soup.find("p", "story")
first_story_paragraph.find_next_sibling("p")        # <p class="story">...</p>

#endregion - find_next_siblings() and find_next_sibling()

#region find_previous_siblings() and find_previous_sibling()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-previous-siblings-and-find-previous-sibling

Signature: find_previous_siblings(name, attrs, string, limit, **kwargs)

Signature: find_previous_sibling(name, attrs, string, **kwargs)

These methods use .previous_siblings to iterate over an element’s siblings that precede it in the tree. 
The find_previous_siblings() method returns all the siblings that match, and find_previous_sibling() only returns 
the first one:
'''
last_link = soup.find("a", id="link3")
last_link                                   # <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

last_link.find_previous_siblings("a")       # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                            #  <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

first_story_paragraph = soup.find("p", "story")
first_story_paragraph.find_previous_sibling("p")    # <p class="title"><b>The Dormouse's story</b></p>

#endregion - find_previous_siblings() and find_previous_sibling()

#region find_all_next() and find_next()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-all-next-and-find-next

Signature: find_all_next(name, attrs, string, limit, **kwargs)

Signature: find_next(name, attrs, string, **kwargs)

These methods use .next_elements to iterate over whatever tags and strings that come after it in the document. 
The find_all_next() method returns all matches, and find_next() only returns the first match:
'''

first_link = soup.a
first_link                              # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_all_next(string=True)   # [u'Elsie', u',\n', u'Lacie', u' and\n', u'Tillie',
                                        #  u';\nand they lived at the bottom of a well.', u'\n\n', u'...', u'\n']

first_link.find_next("p")               # <p class="story">...</p>

'''
In the first example, the string “Elsie” showed up, even though it was contained within the <a> tag we started from. 
In the second example, the last <p> tag in the document showed up, even though it’s not in the same part of the tree 
as the <a> tag we started from. 

For these methods, all that matters is that an element match the filter, and show up later in the document than the 
starting element.
'''
#endregion - find_all_next() and find_next()

#region find_all_previous() and find_previous()
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find-all-previous-and-find-previous

Signature: find_all_previous(name, attrs, string, limit, **kwargs)

Signature: find_previous(name, attrs, string, **kwargs)

These methods use .previous_elements to iterate over the tags and strings that came before it in the document. 
The find_all_previous() method returns all matches, and find_previous() only returns the first match:
'''
first_link = soup.a
first_link                              # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

first_link.find_all_previous("p")       # [<p class="story">Once upon a time there were three little sisters; ...</p>,
                                        #  <p class="title"><b>The Dormouse's story</b></p>]

first_link.find_previous("title")       # <title>The Dormouse's story</title>

'''
The call to find_all_previous("p") found the first paragraph in the document (the one with class=”title”), 
but it also finds the second paragraph, the <p> tag that contains the <a> tag we started with. 

This shouldn’t be too surprising: we’re looking at all the tags that show up earlier in the document than the one 
we started with. A <p> tag that contains an <a> tag must have shown up before the <a> tag it contains.
'''
#endregion - find_all_previous() and find_previous()

#region CSS selectors
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors

As of version 4.7.0, Beautiful Soup supports most CSS4 selectors via the SoupSieve project. 
If you installed Beautiful Soup through pip, SoupSieve was installed at the same time, so you don’t have 
to do anything extra.

BeautifulSoup has a .select() method which uses SoupSieve to run a CSS selector against a parsed document and 
return all the matching elements. Tag has a similar method which runs a CSS selector against the contents of 
a single tag.

(Earlier versions of Beautiful Soup also have the .select() method, but only the most commonly-used CSS selectors 
are supported.)

The SoupSieve documentation lists all the currently supported CSS selectors, but here are some of the basics:

You can find tags:
'''
soup.select("title")                # [<title>The Dormouse's story</title>]
soup.select("p:nth-of-type(3)")     # [<p class="story">...</p>]

'''
Find tags beneath other tags:
'''
soup.select("body a")               # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("html head title")      # [<title>The Dormouse's story</title>]

'''
Find tags directly beneath other tags:
'''
soup.select("head > title")         # [<title>The Dormouse's story</title>]

soup.select("p > a")                # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie"  id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("p > a:nth-of-type(2)") # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

soup.select("p > #link1")           # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select("body > a")             # []

'''
Find the siblings of tags:
'''
soup.select("#link1 ~ .sister")     # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie"  id="link3">Tillie</a>]

soup.select("#link1 + .sister")     # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

'''
Find tags by CSS class:
'''
soup.select(".sister")              # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select("[class~=sister]")      # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

'''
Find tags by ID:
'''
soup.select("#link1")               # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]
soup.select("a#link2")              # [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

'''
Find tags that match any selector from a list of selectors:
'''
soup.select("#link1,#link2")        # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>]

'''
Test for the existence of an attribute:
'''
soup.select('a[href]')              # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

'''
Find tags by attribute value:
'''
soup.select('a[href="http://example.com/elsie"]')   # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

soup.select('a[href^="http://example.com/"]')       # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
                                                    #  <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
                                                    #  <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

soup.select('a[href$="tillie"]')                    # [<a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]
soup.select('a[href*=".com/el"]')                   # [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]

'''
There’s also a method called select_one(), which finds only the first tag that matches a selector:
'''
soup.select_one(".sister")                          # <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

'''
If you’ve parsed XML that defines namespaces, you can use them in CSS selectors.:
'''
from bs4 import BeautifulSoup

xml = """<tag xmlns:ns1="http://namespace1/" xmlns:ns2="http://namespace2/">
            <ns1:child>I'm in namespace 1</ns1:child>
            <ns2:child>I'm in namespace 2</ns2:child>
        </tag> """

soup = BeautifulSoup(xml, "xml")
soup.select("child")                            # [<ns1:child>I'm in namespace 1</ns1:child>, <ns2:child>I'm in namespace 2</ns2:child>]

namespaces = dict(first="http://namespace1/", second="http://namespace2/")

soup.select("ns1|child", namespaces=namespaces)         # [<ns1:child>I'm in namespace 1</ns1:child>]
'''
NOTE: The above line isn't returning what the documentation states it should
      Changing 'ns1|child' to 'ns1:child' throws an error
'''

'''
When handling a CSS selector that uses namespaces, Beautiful Soup uses the namespace abbreviations it found when parsing the document. 
You can override this by passing in your own dictionary of abbreviations:
'''
soup.select("second|child", namespaces=namespaces)       # [<ns1:child>I'm in namespace 2</ns1:child>]

'''
All this CSS selector stuff is a convenience for people who already know the CSS selector syntax. You can do all of this with 
the Beautiful Soup API. And if CSS selectors are all you need, you should parse the document with lxml: it’s a lot faster. 
But this lets you combine CSS selectors with the Beautiful Soup API.
'''
#endregion - CSS selectors

#endregion - Kinds of filters

#endregion - Searching the tree


#region Modifying the tree
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#modifying-the-tree
# TODO: This is lower priority (28/06/2019), skip for now.
#endregion - Modifying the tree


#region Output
# https://www.crummy.com/software/BeautifulSoup/bs4/doc/#output
# TODO: This is lower priority (28/06/2019), skip for now.
#endregion - Output


#region Specifying the parser to use
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#specifying-the-parser-to-use
'''
#region Intro
'''
    If you just need to parse some HTML, you can dump the markup into the BeautifulSoup constructor, and it’ll probably be fine. 
    Beautiful Soup will pick a parser for you and parse the data. But there are a few additional arguments you can pass in to the 
    constructor to change which parser is used.

    The first argument to the BeautifulSoup constructor is a string or an open filehandle–the markup you want parsed.
    The second argument is how you’d like the markup parsed.

    If you don’t specify anything, you’ll get the best HTML parser that’s installed. 
    Beautiful Soup ranks lxml’s parser as being the best, then html5lib’s, then Python’s built-in parser. 
    You can override this by specifying one of the following:

        - What type of markup you want to parse. Currently supported are “html”, “xml”, and “html5”.
        - The name of the parser library you want to use. Currently supported options are “lxml”, “html5lib”, and 
          “html.parser” (Python’s built-in HTML parser).

    The section Installing a parser contrasts the supported parsers.

    If you don’t have an appropriate parser installed, Beautiful Soup will ignore your request and pick a different parser. 
    Right now, the only supported XML parser is lxml. If you don’t have lxml installed, asking for an XML parser won’t give you one, 
    and asking for “lxml” won’t work either.
'''
#endregion - Intro

#region Differences between parsers
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#differences-between-parsers

Beautiful Soup presents the same interface to a number of different parsers, but each parser is different. 
Different parsers will create different parse trees from the same document. The biggest differences are between 
the HTML parsers and the XML parsers. Here’s a short document, parsed as HTML:
'''
BeautifulSoup("<a><b /></a>", "lxml")       # <html><head></head><body><a><b></b></a></body></html>

'''
Since an empty <b /> tag is not valid HTML, the parser turns it into a <b></b> tag pair.

Here’s the same document parsed as XML (running this requires that you have lxml installed). 
Note that the empty <b /> tag is left alone, and that the document is given an XML declaration 
instead of being put into an <html> tag.:
'''
BeautifulSoup("<a><b /></a>", "xml")        # <?xml version="1.0" encoding="utf-8"?>
                                            # <a><b/></a>

'''
There are also differences between HTML parsers. If you give Beautiful Soup a perfectly-formed HTML document, 
these differences won’t matter. One parser will be faster than another, but they’ll all give you a data 
structure that looks exactly like the original HTML document.

But if the document is not perfectly-formed, different parsers will give different results. Here’s a short, 
invalid document parsed using lxml’s HTML parser. Note that the dangling </p> tag is simply ignored:
'''
BeautifulSoup("<a></p>", "lxml")            # <html><body><a></a></body></html>

'''
Here’s the same document parsed using html5lib:
'''
BeautifulSoup("<a></p>", "html5lib")        # <html><head></head><body><a><p></p></a></body></html>

'''
Instead of ignoring the dangling </p> tag, html5lib pairs it with an opening <p> tag. 
This parser also adds an empty <head> tag to the document.

Here’s the same document parsed with Python’s built-in HTML parser:
'''
BeautifulSoup("<a></p>", "html.parser")     # <a></a>

'''
Like html5lib, this parser ignores the closing </p> tag. Unlike html5lib, this parser makes no attempt 
to create a well-formed HTML document by adding a <body> tag. Unlike lxml, it doesn’t even bother to 
add an <html> tag.

Since the document “<a></p>” is invalid, none of these techniques is the “correct” way to handle it. 
The html5lib parser uses techniques that are part of the HTML5 standard, so it has the best claim on 
being the “correct” way, but all three techniques are legitimate.

Differences between parsers can affect your script. If you’re planning on distributing your script to 
other people, or running it on multiple machines, you should specify a parser in the BeautifulSoup 
constructor. That will reduce the chances that your users parse a document differently from the way 
you parse it.
'''
#endregion - Differences between parsers

#endregion - Specifying the parser to use


#region Encodings
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#encodings
'''
#region Intro
'''
Any HTML or XML document is written in a specific encoding like ASCII or UTF-8. But when you load that document 
into Beautiful Soup, you’ll discover it’s been converted to Unicode:
'''
markup = '<h1>Sacr\xc3\xa9 bleu!</h1>'
'''
NOTE: On Mac, this is instantly being set as '<h1>SacrÃ© bleu!</h1>'
      the encoding of this should be UTF8, but VSCode/OSX are setting this differently
'''

soup = BeautifulSoup(markup, features="lxml")
soup.h1                         # <h1>Sacré bleu!</h1>
soup.h1.string                  # u'Sacr\xe9 bleu!'

'''
It’s not magic. (That sure would be nice.) Beautiful Soup uses a sub-library called Unicode, Dammit to detect a 
document’s encoding and convert it to Unicode. The autodetected encoding is available as the .original_encoding 
attribute of the BeautifulSoup object:
'''
soup.original_encoding              # 'utf-8'
'''
NOTE: On Mac, this isn't set - we are getting None
'''

'''
Unicode, Dammit guesses correctly most of the time, but sometimes it makes mistakes. 
Sometimes it guesses correctly, but only after a byte-by-byte search of the document that takes a very long time. 
If you happen to know a document’s encoding ahead of time, you can avoid mistakes and delays by passing it to 
the BeautifulSoup constructor as from_encoding.

Here’s a document written in ISO-8859-8. The document is so short that Unicode, Dammit can’t get a lock on it, 
and misidentifies it as ISO-8859-7:
'''
markup = b"<h1>\xed\xe5\xec\xf9</h1>"
soup = BeautifulSoup(markup)
soup.h1                             # <h1>νεμω</h1>
soup.original_encoding              # 'ISO-8859-7'

'''
We can fix this by passing in the correct from_encoding:
'''
soup = BeautifulSoup(markup, from_encoding="iso-8859-8")
soup.h1                             # <h1>םולש</h1>
soup.original_encoding              # 'iso8859-8'

'''
If you don’t know what the correct encoding is, but you know that Unicode, Dammit is guessing wrong, 
you can pass the wrong guesses in as exclude_encodings:
'''
soup = BeautifulSoup(markup, exclude_encodings=["ISO-8859-7"])
soup.h1                             # <h1>םולש</h1>  - on Mac this is <h1>íåìù</h1> from encoding 'WINDOWS-1252'
soup.original_encoding              # 'WINDOWS-1255' - on Mac this is 'WINDOWS-1252'

'''
Windows-1255 isn’t 100% correct, but that encoding is a compatible superset of ISO-8859-8, so it’s 
close enough. (exclude_encodings is a new feature in Beautiful Soup 4.4.0.)

In rare cases (usually when a UTF-8 document contains text written in a completely different encoding), 
the only way to get Unicode may be to replace some characters with the special Unicode character 
“REPLACEMENT CHARACTER” (U+FFFD, �). If Unicode, Dammit needs to do this, it will set the 
.contains_replacement_characters attribute to True on the UnicodeDammit or BeautifulSoup object. 
This lets you know that the Unicode representation is not an exact representation of the original–some 
data was lost. If a document contains �, but .contains_replacement_characters is False, you’ll know that 
the � was there originally (as it is in this paragraph) and doesn’t stand in for missing data.
'''
#endregion - Intro

#region Output encoding
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#output-encoding

When you write out a document from Beautiful Soup, you get a UTF-8 document, even if the document wasn’t 
in UTF-8 to begin with. Here’s a document written in the Latin-1 encoding:
'''
markup = b'''
 <html>
  <head>
   <meta content="text/html; charset=ISO-Latin-1" http-equiv="Content-type" />
  </head>
  <body>
   <p>Sacr\xe9 bleu!</p>
  </body>
 </html>
'''

soup = BeautifulSoup(markup)
soup.prettify()                     # <html>
                                    #  <head>
                                    #   <meta content="text/html; charset=utf-8" http-equiv="Content-type" />
                                    #  </head>
                                    #  <body>
                                    #   <p>
                                    #    Sacré bleu!
                                    #   </p>
                                    #  </body>
                                    # </html>
'''
Note that the <meta> tag has been rewritten to reflect the fact that the document is now in UTF-8.

If you don’t want UTF-8, you can pass an encoding into prettify():
'''
soup.prettify("latin-1")

'''
You can also call encode() on the BeautifulSoup object, or any element in the soup, just as if it 
were a Python string:
'''
soup.p.encode("latin-1")            # '<p>Sacr\xe9 bleu!</p>'
soup.p.encode("utf-8")              # '<p>Sacr\xc3\xa9 bleu!</p>'

'''
Any characters that can’t be represented in your chosen encoding will be converted into numeric XML 
entity references. Here’s a document that includes the Unicode character SNOWMAN:
'''
markup = u"<b>\N{SNOWMAN}</b>"
snowman_soup = BeautifulSoup(markup)
tag = snowman_soup.b                    # NOTE: Without changing encoding in VSCode/OSX this is already
                                        #       <b>☃</b>

'''
The SNOWMAN character can be part of a UTF-8 document (it looks like ☃), but there’s no representation 
for that character in ISO-Latin-1 or ASCII, so it’s converted into “&#9731” for those encodings:
'''
tag.encode("utf-8")                     # <b>☃</b> - in VSCode/OSX this is b'<b>\xe2\x98\x83</b>'
tag.encode("latin-1")                   # <b>&#9731;</b>
tag.encode("ascii")                     # <b>&#9731;</b>
#endregion - Output encoding

#region Unicode, Dammit
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#unicode-dammit

You can use Unicode, Dammit without using Beautiful Soup. It’s useful whenever you have data in an 
unknown encoding and you just want it to become Unicode:
'''
from bs4 import UnicodeDammit
dammit = UnicodeDammit("Sacr\xc3\xa9 bleu!")

dammit.unicode_markup               # Sacré bleu! - this isn't utf-8 on VSCode/OSX
dammit.original_encoding            # 'utf-8'

'''
Unicode, Dammit’s guesses will get a lot more accurate if you install the chardet or cchardet Python 
libraries. The more data you give Unicode, Dammit, the more accurately it will guess. If you have 
your own suspicions as to what the encoding might be, you can pass them in as a list:
'''
dammit = UnicodeDammit("Sacr\xe9 bleu!", ["latin-1", "iso-8859-1"])
dammit.unicode_markup               # Sacré bleu!
dammit.original_encoding            # 'latin-1' - this is None on VSCode/OSX

'''
Unicode, Dammit has two special features that Beautiful Soup doesn’t use.
'''
#endregion - Unicode, Dammit

#region Unicode, Dammit - Smart quotes
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#smart-quotes

You can use Unicode, Dammit to convert Microsoft smart quotes to HTML or XML entities:
'''
markup = b"<p>I just \x93love\x94 Microsoft Word\x92s smart quotes</p>"

UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="html").unicode_markup
                                # u'<p>I just &ldquo;love&rdquo; Microsoft Word&rsquo;s smart quotes</p>'

UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="xml").unicode_markup
                                # u'<p>I just &#x201C;love&#x201D; Microsoft Word&#x2019;s smart quotes</p>'

'''
You can also convert Microsoft smart quotes to ASCII quotes:
'''
UnicodeDammit(markup, ["windows-1252"], smart_quotes_to="ascii").unicode_markup
                                # u'<p>I just "love" Microsoft Word\'s smart quotes</p>'

'''
Hopefully you’ll find this feature useful, but Beautiful Soup doesn’t use it. Beautiful Soup prefers the 
default behavior, which is to convert Microsoft smart quotes to Unicode characters along with everything else:
'''
UnicodeDammit(markup, ["windows-1252"]).unicode_markup
                                # u'<p>I just \u201clove\u201d Microsoft Word\u2019s smart quotes</p>'
                                # On VSCode/OSX: '<p>I just “love” Microsoft Word’s smart quotes</p>'

#endregion - Unicode, Dammit - Smart quotes

#region Unicode, Dammit - Inconsistent encodings
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#inconsistent-encodings

Sometimes a document is mostly in UTF-8, but contains Windows-1252 characters such as (again) Microsoft 
smart quotes. This can happen when a website includes data from multiple sources. You can use 
UnicodeDammit.detwingle() to turn such a document into pure UTF-8. Here’s a simple example:
'''
snowmen = (u"\N{SNOWMAN}" * 3)
quote = (u"\N{LEFT DOUBLE QUOTATION MARK}I like snowmen!\N{RIGHT DOUBLE QUOTATION MARK}")
doc = snowmen.encode("utf8") + quote.encode("windows_1252")

'''
This document is a mess. The snowmen are in UTF-8 and the quotes are in Windows-1252. 
You can display the snowmen or the quotes, but not both:
'''
doc                              # ☃☃☃�I like snowmen!�          - different on VSCode/OSX
doc.decode("windows-1252")       # â˜ƒâ˜ƒâ˜ƒ“I like snowmen!”

'''
Decoding the document as UTF-8 raises a UnicodeDecodeError, and decoding it as Windows-1252 gives 
you gibberish. Fortunately, UnicodeDammit.detwingle() will convert the string to pure UTF-8, allowing 
you to decode it to Unicode and display the snowmen and quote marks simultaneously:
'''
new_doc = UnicodeDammit.detwingle(doc)
new_doc.decode("utf8")           # ☃☃☃“I like snowmen!”

'''
UnicodeDammit.detwingle() only knows how to handle Windows-1252 embedded in UTF-8 (or vice versa, 
I suppose), but this is the most common case.

Note that you must know to call UnicodeDammit.detwingle() on your data before passing it into 
BeautifulSoup or the UnicodeDammit constructor. Beautiful Soup assumes that a document has a single 
encoding, whatever it might be. If you pass it a document that contains both UTF-8 and Windows-1252, 
it’s likely to think the whole document is Windows-1252, and the document will come out looking 
like â˜ƒâ˜ƒâ˜ƒ“I like snowmen!”.

UnicodeDammit.detwingle() is new in Beautiful Soup 4.1.0.
'''
#endregion - Unicode, Dammit - Inconsistent encodings

#endregion - Encodings


#region Comparing objects for equality
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#comparing-objects-for-equality

Beautiful Soup says that two NavigableString or Tag objects are equal when they represent the same HTML 
or XML markup. In this example, the two <b> tags are treated as equal, even though they live in different 
parts of the object tree, because they both look like “<b>pizza</b>”:
'''
markup = "<p>I want <b>pizza</b> and more <b>pizza</b>!</p>"
soup = BeautifulSoup(markup, 'html.parser')

first_b, second_b = soup.find_all('b')

first_b == second_b                                      # True
first_b.previous_element == second_b.previous_element    # False

'''
If you want to see whether two variables refer to exactly the same object, use is:
'''
first_b is second_b                                      # False

#endregion - Comparing objects for equality


#region Copying Beautiful Soup objects
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#copying-beautiful-soup-objects

You can use copy.copy() to create a copy of any Tag or NavigableString:
'''
import copy
p_copy = copy.copy(soup.p)
p_copy                      # <p>I want <b>pizza</b> and more <b>pizza</b>!</p>

'''
The copy is considered equal to the original, since it represents the same markup as the original, 
but it’s not the same object:
'''
soup.p == p_copy            # True
soup.p is p_copy            # False

'''
The only real difference is that the copy is completely detached from the original Beautiful Soup 
object tree, just as if extract() had been called on it:
'''
p_copy.parent               # None

'''
This is because two different Tag objects can’t occupy the same space at the same time.
'''
#endregion - Copying Beautiful Soup objects


#region Parsing only part of a document
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#parsing-only-part-of-a-document

Let’s say you want to use Beautiful Soup look at a document’s <a> tags. It’s a waste of time and memory 
to parse the entire document and then go over it again looking for <a> tags. It would be much faster to 
ignore everything that wasn’t an <a> tag in the first place. The SoupStrainer class allows you to choose 
which parts of an incoming document are parsed. You just create a SoupStrainer and pass it in to the 
BeautifulSoup constructor as the parse_only argument.

(Note that this feature won’t work if you’re using the html5lib parser. If you use html5lib, the whole 
document will be parsed, no matter what. This is because html5lib constantly rearranges the parse tree 
as it works, and if some part of the document didn’t actually make it into the parse tree, it’ll crash. 
To avoid confusion, in the examples below I’ll be forcing Beautiful Soup to use Python’s built-in parser.)
'''

#region SoupStrainer
'''
https://www.crummy.com/software/BeautifulSoup/bs4/doc/#soupstrainer

The SoupStrainer class takes the same arguments as a typical method from Searching the tree: 
name, attrs, string, and **kwargs. Here are three SoupStrainer objects:
'''
from bs4 import SoupStrainer

only_a_tags = SoupStrainer("a")

only_tags_with_id_link2 = SoupStrainer(id="link2")


def is_short_string(s):
    return len(s) < 10

only_short_strings = SoupStrainer(string=is_short_string)

'''
I’m going to bring back the “three sisters” document one more time, and we’ll see what the document 
looks like when it’s parsed with these three SoupStrainer objects:
'''
BeautifulSoup(html_doc, "html.parser", parse_only=only_a_tags).prettify()
BeautifulSoup(html_doc, "html.parser", parse_only=only_tags_with_id_link2).prettify()

'''
NOTE: This doesn't work - looking at the parameter values is_short_strings, the values are None
'''
# BeautifulSoup(html_doc, "html.parser", parse_only=only_short_strings).prettify()

'''
You can also pass a SoupStrainer into any of the methods covered in Searching the tree. 
This probably isn’t terribly useful, but I thought I’d mention it:

NOTE: Same as above, is_short_strings is only taking in None values
WORKAROUND: https://python-forum.io/Thread-SoupStrainer-example
'''
# soup = BeautifulSoup(html_doc)
# soup.find_all(only_short_strings)


#endregion - SoupStrainer

#endregion - Parsing only part of a document

#region Troubleshooting
#endregion - Troubleshooting




# entries_identifier = {
#     "element": "li",
#     "attributes": { "class": "o-teaser-collection__item" }
# }

# all_elements = content.find_all(content_identifier["element"], attrs=content_identifier["attributes"])
