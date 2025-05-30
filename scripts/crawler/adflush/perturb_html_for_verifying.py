#!/usr/bin/env python3
import random
import re
import string
from urllib.parse import parse_qs, urlencode, urlunparse
from urllib.parse import urlparse, urljoin
import copy
import os

import bs4
from fuzzywuzzy import fuzz

# process features
deltaNodes = 0  # # of edges should be equal to # of nodes
soup = None
requestURL = None
baseURL = None
modifiedRequestURL = None
tagCount = 0

adKeyWord = ["ad", "ads", "advert", "popup", "banner", "sponsor", "iframe", "googlead", "adsys", "adser",
             "advertise", "redirect", "popunder", "punder", "popout", "click", "track", "play", "pop", "prebid", "bid",
             "pb.min", "affiliate", "ban", "delivery", "promo", "tag", "zoneid", "siteid", "pageid", "size", "viewid",
             "zone_id", "google_afc", "google_afs"]
adKeyChar = [".", "/", "&", "=", ";", "-", "_", "/", "*", "^", "?", ";", "|", ","]
screenResolution = ["screenheight", "screenwidth", "browserheight", "browserwidth", "screendensity", "screen_res",
                    "screen_param", "screenresolution", "browsertimeoffset"]
# We won't add empty nodes under parents nodes with these name since it may cause error
blackListedTags = ["script", "html", "head", "body"]


def parse_url(url):
    components = urlparse(url)
    scheme = components[0]
    netloc = components[1]
    path = components[2]
    params = components[3]
    query = components[4]
    fragment = components[5]
    return scheme, netloc, path, params, query, fragment


def is_same_request(url1, url2, strict=True):
    scheme1, netloc1, path1, params1, query1, fragment1 = parse_url(url1)
    scheme2, netloc2, path2, params2, query2, fragment2 = parse_url(url2)

    if strict:
        if scheme1 == scheme2 \
                and netloc1 == netloc2 \
                and path1 == path2 \
                and params1 == params2 \
                and query1 == query2 \
                and fragment1 == fragment2:
            return True
        else:
            return False
    else:
        if netloc1 == netloc2 \
                and path1 == path2:
            return True
        else:
            return False


def BSFilterByAnyString(tag):
    global requestURL
    scheme, netloc, path, params, query, fragment = parse_url(requestURL)
    fuzzy_url = '/'.join([netloc, path])

    # Following 2 ways are for attributes
    for attr_val in list(tag.attrs.values()):
        if not isinstance(attr_val, str):
            continue
        if attr_val.startswith('/'):
            attr_val = urljoin(baseURL, attr_val)

        # Level2: URL component-level matching
        if is_same_request(attr_val, requestURL, strict=True):
            return tag

        # Level3: URL fuzzy matching
        if fuzzy_url in attr_val:
            return tag

        # # Level4: Fuzzy matching
        # if fuzz.ratio(attr_val, requestURL) > 95:
        #     return tag
        
        # # shine: handle edge cases
        # if tag.name == "script":
        #     if requestURL in tag.text:
        #         return tag



def getAttrByURL(tag):
    global requestURL

    scheme, netloc, path, params, query, fragment = parse_url(requestURL)
    fuzzy_url = '/'.join([netloc, path])

    for attr_key, attr_val in tag.attrs.items():
        if not isinstance(attr_val, str):
            continue
        if attr_val.startswith('/'):
            attr_val = urljoin(baseURL, attr_val)

        # Level1: URL component-level matching
        if is_same_request(attr_val, requestURL, strict=True):
            return attr_key

        # Level2: URL fuzzy matching
        if fuzzy_url in attr_val:
            return attr_key

        # Level3: Fuzzy matching
        if fuzz.ratio(attr_val, requestURL) > 95:
            return attr_key
    
    # # shine: handling requested url is in the function parameter.
    # script_content = tag.string
    # if requestURL in script_content:
    #     tag.attrs["href"] = requestURL
    #     return "href"


def ModifyURLLength(delta):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    
    if delta < 0:
        current_qs = urlparse(tag.attrs[attr])
        qs_len = len(current_qs.query)
        # print("qs_len:{}".format(qs_len))
        if (-1 * qs_len) <= delta:
            pass
        else:
            delta = (-1 * qs_len)
        # print("delta:{}".format(delta))
        schemaDomainStr = splitHostAndPath(tag.attrs[attr])[0] + "/"
        targetLen = len(tag.attrs[attr]) + delta
        if targetLen < len(schemaDomainStr) + delta:
            targetLen = len(schemaDomainStr) + delta
        tag.attrs[attr] = tag.attrs[attr][:targetLen]           

    else:
        for i in range(delta):
            tag.attrs[attr] += "#"
    modifiedRequestURL = tag.attrs[attr]


def splitHostAndPath(str):
    parsed_url = urlparse(url)
    domain_parts = parsed_url.netloc.split('.')
    if len(domain_parts) > 2:
        domain = '.'.join(domain_parts[-2:])
    else:
        domain = parsed_url.netloc
    return domain, parsed_url.path


# All of the following "Replacing" would replace with random chars. The server may parse the URL dynamically
def ModifyADKeyword(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    hostPathArray = splitHostAndPath(tag.attrs[attr])
    if add == -1:
        for adWord in adKeyWord:
            randStr = ""
            for _ in adWord:
                randStr += random.choice(string.ascii_lowercase)
            # print(hostPathArray[1].replace(adWord, randStr))
            hostPathArray[1] = hostPathArray[1].replace(adWord, randStr)
            tag.attrs[attr] = hostPathArray[0] + hostPathArray[1].replace(adWord, randStr)
    elif add == 1:
        tag.attrs[attr] += "#" + random.choice(adKeyWord)
    modifiedRequestURL = tag.attrs[attr]


# Possibly not usable since it replaces the special chars in the URL, but the server may parse the URL dynamically
def ModifyADKeyChar(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    hostPathArray = splitHostAndPath(tag.attrs[attr])
    if add == -1:
        for adWord in adKeyWord:
            for adChar in adKeyChar:
                combinedStr = adChar + adWord
                tag.attrs[attr] = hostPathArray[0] + hostPathArray[1].replace(combinedStr, random.choice(
                    string.ascii_lowercase) + adWord)
    elif add == 1:
        tag.attrs[attr] += "#?" + random.choice(adKeyWord)
    modifiedRequestURL = tag.attrs[attr]


# Not tested
def ModifySemicolon(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        tag.attrs[attr] = tag.attrs[attr].replace(";", "&")
    elif add == 1:
        tag.attrs[attr] += "#;"
    modifiedRequestURL = tag.attrs[attr]


def ModifyQueryString(removeQS):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if removeQS == 1:
        tag.attrs[attr] = tag.attrs[attr].replace("?", "#")
    elif removeQS == -1:
        tag.attrs[attr] = tag.attrs[attr].replace("#", "?")
    modifiedRequestURL = tag.attrs[attr]


def GetQSDictFromURL(url):
    urlParts = urlparse(url)
    qsdict = parse_qs(urlParts.query)
    for key in qsdict.keys():
        qsdict[key] = qsdict[key][0]
    return qsdict


def GenerateURLWithQS(url, qsdict):
    urlParts = urlparse(url)
    urlPartsList = list(urlParts)
    urlPartsList[4] = urlencode(qsdict)
    return urlunparse(urlPartsList)


def ModifyBaseDomainQueryString(add, domain):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    # Trim the domain
    if domain.startswith("www."):
        domain = domain[4:]
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        # Find the key/val pair
        for key in qsdict.keys():
            if domain == qsdict[key]:
                # Remove only if we found, prevent crash
                del qsdict[key]
                break
        # Write back
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)
    elif add == 1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        qsdict[str(random.randint(0, 999999))] = domain
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)
    modifiedRequestURL = tag.attrs[attr]


def ModifyScreenDimensionInBaseURL(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    hostPathArray = splitHostAndPath(tag.attrs[attr])
    if add == -1:
        tag.attrs[attr] = hostPathArray[0] + ReplaceX(hostPathArray[1])
    elif add == 1:
        tag.attrs[attr] += "#" + str(random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))
    modifiedRequestURL = tag.attrs[attr]


def ModifyADDimensionInQS(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        for key in qsdict.keys():
            qsdict[key] = ReplaceX(qsdict[key])
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)

    elif add == 1:
        if "?" in tag.attrs[attr]:
            tag.attrs[attr] += "&" + random.choice(string.ascii_lowercase) + "=" + str(
                random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))
        else:
            tag.attrs[attr] += "?" + random.choice(string.ascii_lowercase) + "=" + str(
                random.randint(0, 10000)) + "x" + str(random.randint(0, 10000))
    modifiedRequestURL = tag.attrs[attr]


def ModifyScreenDimensionKeywordInQS(add):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if add == -1:
        qsdict = GetQSDictFromURL(tag.attrs[attr])
        for key in qsdict.keys():
            if key in screenResolution:
                del qsdict[key]
                break
        tag.attrs[attr] = GenerateURLWithQS(tag.attrs[attr], qsdict)

    elif add == 1:
        if "?" in tag.attrs[attr]:
            tag.attrs[attr] += "&" + random.choice(screenResolution) + "=" + random.choice(string.ascii_lowercase)
        else:
            tag.attrs[attr] += "?" + random.choice(screenResolution) + "=" + random.choice(string.ascii_lowercase)
    modifiedRequestURL = tag.attrs[attr]


def ReplaceX(text):
    splitedStr = re.split("\\d{2,4}[xX_-]\\d{2,4}", text)
    matchedStr = re.findall("\\d{2,4}[xX_-]\\d{2,4}", text)
    text = ""
    randomstr = 'x'
    while randomstr == 'x':
        randomstr = random.choice(string.ascii_lowercase)
    for i in range(len(matchedStr)):
        replacingStr = re.sub("[xX_-]", randomstr, matchedStr[i])
        text += splitedStr[i] + replacingStr
    # if len(matchedStr) > 0:
    text += splitedStr[len(splitedStr) - 1]
    return text


def ModifyHostName(setToBaseDomain, baseDomain):
    global modifiedRequestURL
    tag = soup.find(BSFilterByAnyString)
    attr = getAttrByURL(tag)
    if setToBaseDomain == 1:
        # urlParts = urlparse(tag.attrs[attr])
        # urlPartsList = list(urlParts)
        # urlPartsList[1] = baseDomain
        # tag.attrs[attr] = urlunparse(urlPartsList)
        print("Will not modify domain name now, though requested.")
    elif setToBaseDomain == -1:
        urlParts = urlparse(tag.attrs[attr])
        urlPartsList = list(urlParts)
        urlPartsList[1] = urlPartsList[1] + "."
        tag.attrs[attr] = urlunparse(urlPartsList)
    modifiedRequestURL = tag.attrs[attr]


def ModifyHostNameWithSubDomain(addSubdomain, baseDomain):
    # tag = soup.find(BSFilterByAnyString)
    # attr = getAttrByURL(tag)
    # if addSubdomain == 1:
    #     urlParts = urlparse(tag.attrs[attr])
    #     urlPartsList = list(urlParts)
    #     urlPartsList[1] = baseDomain + "." + urlPartsList[1]
    #     tag.attrs[attr] = urlunparse(urlPartsList)
    # elif addSubdomain == -1:
    #     urlParts = urlparse(tag.attrs[attr])
    #     urlPartsList = list(urlParts)
    #     # Trim the domain
    #     if baseDomain.startswith("www."):
    #         baseDomain = baseDomain[4:]
    #     urlPartsList[1] = urlPartsList[1].replace(baseDomain + ".", "").replace("." + baseDomain, "").replace(
    #         baseDomain, "")
    #     tag.attrs[attr] = urlunparse(urlPartsList)
    print("Will not modify subdomain name now, though requested.")


def getTagCount():
    global tagCount
    tagCount = 0
    t = soup.head
    while t:
        if isinstance(t, bs4.element.Tag) and t.name not in blackListedTags:
            tagCount += 1
        t = t.next_element


def getRandomNode():
    getTagCount()
    targetTag = int(tagCount * random.random())
    t = soup.head
    for i in range(targetTag):
        t = t.next_element
        while t.name in blackListedTags:
            t = t.next_element
    return t


def addNodes(delta, mandatory=True, strategy="Centralized"):
    global deltaNodes, soup
    if delta < 0:
        print("changeNodes with delta < 0!")
        return
    if deltaNodes >= delta and mandatory:
        # We have done this
        return
    tag = soup.find(BSFilterByAnyString)
    for i in range(delta):
        deltaNodes += 1
        # newTag = random.choice(candidate_tags)
        newTag = soup.new_tag('p')
        try:
            newTag.attrs["hidden"] = ""
        except:
            pass
        if strategy == "Centralized":
            tag.insert_before(newTag)
        elif strategy == "Distributed":
            getRandomNode().insert_before(newTag)

def addChildNodes(delta):
    global deltaNodes, soup
    if delta < 0:
        print("changeNodes with delta < 0!")
        return
    if deltaNodes >= delta and mandatory:
        # We have done this
        return
    tag = soup.find(BSFilterByAnyString)
    for i in range(delta):
        deltaNodes += 1
        newTag = random.choice(candidate_tags)
        # newTag = soup.new_tag('p')
        try:
            newTag.attrs["hidden"] = ""
        except:
            pass
            tag.append(newTag)


def wrapNodes(layers, tagName="span"):
    global deltaNodes, soup
    if layers < 0:
        print("wrapNodes with layers < 0!")
        return
    for i in range(layers):
        tag = soup.find(BSFilterByAnyString)
        if tag is None:
            continue
        deltaNodes += 1
        newTag = soup.new_tag(tagName)
        tag.wrap(newTag)

def wrapNodes_ver2(layers, tagName="div"):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)

    parent_tag = tag.find_parent()
    children_tags = parent_tag.find_all(recursive=False)

    for i in range(layers):
        if i == layers - 1:
            tagName = "div"
        newTag = soup.new_tag(tagName)
        for child_tag in children_tags:
            child_tag.wrap(newTag)


def IncreaseFirstNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    # print("@ tag in IncreaseFirstNumberOfSiblings: {}".format(parent_tag))
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.insert_before(newTag)


def DecreaseFirstNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    orig_sibling_num = -1
    for sibling in tag.parent.children:
        orig_sibling_num += 1
    print("orig_sibling_num: %d" % orig_sibling_num)
    newTag = soup.new_tag('span')
    deltaNodes += 1
    tag.wrap(newTag)
    for i in range(orig_sibling_num - delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.insert_after(newTag)


def IncreaseFirstParentNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    # shine: get parent's parent
    try:
        parent_tag = tag.parent
    except:
        return
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent_tag.insert_before(newTag)


def DecreaseFirstParentNumberOfSiblings(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    orig_sibling_num = -1
    for sibling in parent.parent.children:
        orig_sibling_num += 1
    print("orig_sibling_num: %d" % orig_sibling_num)
    newTag = soup.new_tag('span')
    deltaNodes += 1
    parent.wrap(newTag)
    for i in range(orig_sibling_num - delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent.insert_before(newTag)


def SetFirstParentSiblingTagName(tagName):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag(tagName)
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.insert_before(newTag)


def RemoveFirstParentSiblingAdAttribute():
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag('p')
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.insert_before(newTag)


def IncreaseFirstParentInboundConnections(delta):
    global deltaNodes, soup
    tag = soup.find("body")
    for i in range(delta):
        script_tag = soup.new_tag('script')
        script_tag.append('var x = document.querySelector(\'[src="' + requestURL +
                          '"]\'); x.classList.add(\'eirqghjuijsiodvpasfwegg\'); x.classList.remove(\'eirqghjuijsiodvpasfwegg\');')
        deltaNodes += 1
        tag.append(script_tag)


def IncreaseFirstParentOutboundConnections(delta):
    # add children to first parent
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    try:
        parent_tag = tag.parent
    except:
        return
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent_tag.append(newTag)


def IncreaseFirstParentInboundOutboundConnections(delta):
    # add children to first parent
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    try:
        parent_tag = tag.parent
    except:
        return
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        parent_tag.append(newTag)


def IncreaseFirstParentAverageDegreeConnectivity(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    parent = tag.parent
    newTag = soup.new_tag('div')
    newTag.attrs["hidden"] = ""
    deltaNodes += 1
    parent.insert_before(newTag)
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        newTag.append(newTag)


def DecreaseFirstParentAverageDegreeConnectivity(delta):
    global deltaNodes, soup
    tag = soup.find("body")
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.append(newTag)

#shine
def duplicate_tag(delta):
    # duplicate target nodes
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    if delta > 0:
        for i in range(delta):
            newTag = copy.copy(tag)
            newTag.attrs["hidden"] = ""
            deltaNodes += 1
            tag.insert_after(newTag)

# shine
def IncreaseInboundConnections(delta):
    global deltaNodes, soup
    tag = soup.find("body")
    for i in range(delta):
        script_tag = soup.new_tag('script')
        script_tag.append('var x = document.querySelector(\'[src="' + requestURL +
                          '"]\'); x.classList.add(\'eirqghjuijsiodvpasfwegg\'); x.classList.remove(\'eirqghjuijsiodvpasfwegg\');')
        deltaNodes += 1
        tag.append(script_tag)

# shine
def IncreaseOutboundConnections(delta):
    # add children to first parent
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    for i in range(delta):
        newTag = soup.new_tag('p')
        newTag.attrs["hidden"] = ""
        deltaNodes += 1
        tag.append(newTag)


def IncreaseURLLength(delta):
    global deltaNodes, soup
    tag = soup.find(BSFilterByAnyString)
    tag.attrs["src"] = tag.attrs["src"] + '*' * delta

# shine: flow featuers
def IncreaseGetCookieOrStorage(delta, js_fname):
    global soup
    if delta == 0:
        return
    
    with open(js_fname, 'r') as file:
        js_content = file.read()
    
    additional_code = """
function setCookie(cookieName, cookieValue, expirationDays) {
const date = new Date();
date.setTime(date.getTime() + (expirationDays * 24 * 60 * 60 * 1000));
const expires = "expires=" + date.toUTCString();
document.cookie = `${cookieName}=${cookieValue}; ${expires}; path=/`;
}

function getCookie(cookieName) {
    const name = `${cookieName}=`;
    const decodedCookie = decodeURIComponent(document.cookie);
    const cookieArray = decodedCookie.split(';');

    for (let i = 1; i <= """+f"""{delta}"""+"""; i++) {
        const cookieName = `yourCookieName${i}`;
        setCookie(cookieName, "yourCookieValue", 7);
}
const cookieName2 = `yourCookieName0`;
getCookie(cookieName2);
    """
    
    modified_js_content = js_content + additional_code
    
    with open(js_fname, 'w') as file:
        file.write(modified_js_content)
        print(f"Additional code added to {js_fname}")

def IncreaseSetCookieOrStorage(delta, js_fname):
    global soup
    if delta == 0:
        return
    
    print("1")
    with open(js_fname, 'r') as file:
        js_content = file.read()
    print("2")
        
    additional_code = """
function setCookie(cookieName, cookieValue, expirationDays) {
    const date = new Date();
    date.setTime(date.getTime() + (expirationDays * 24 * 60 * 60 * 1000));
    const expires = "expires=" + date.toUTCString();
    document.cookie = `${cookieName}=${cookieValue}; ${expires}; path=/`;
}

for (let i = 1; i <= """+f"""{delta}"""+"""; i++) {
    const cookieName = `yourCookieName${i}`;
    setCookie(cookieName, "yourCookieValue", 7);
}
    """
    
    modified_js_content = js_content + additional_code
    
    with open(js_fname, 'w') as file:
        file.write(modified_js_content)
        print(f"Additional code added to {js_fname}")

def IncreaseSetRequestSent(delta, js_fname):
    global soup
    if delta == 0:
        return
    
    with open(js_fname, 'r') as file:
        js_content = file.read()
        
    additional_code = """
for (let i = 1; i <= """+f"""{delta}"""+"""; i++) {
    fetch("https://jsonplaceholder.typicode.com/todos/1")
    .then((response) => response.json())
    .then((json) => console.log(json));
}
    """
    
    modified_js_content = js_content + additional_code
    
    with open(js_fname, 'w') as file:
        file.write(modified_js_content)
        print(f"Additional code added to {js_fname}")

def featureMapbacks(name, html, url, js_fname, delta=None, domain=None, strategy="Centralized", candidate=None, qs=""):
    global deltaNodes, soup, requestURL, baseURL, modifiedRequestURL, candidate_tags, query_strings
    deltaNodes = 0
    soup = html
    requestURL = url
    modifiedRequestURL = url
    baseURL = domain
    candidate_tags = candidate
    query_strings = qs
    
    before_mapback = str(soup)
    # print("URL to find: %s" % url)
    print("Feature name: %s | %s" % (name, str(delta)))

    if name == "num_nodes" or name == "num_edges":
        try:
            addNodes(delta, True, strategy)
        except Exception as err:
            print(err)
            pass
    
    if name == "indirect_all_out_degree":
        try:
            addChildNodes(delta)
        except:
            pass
    
    elif name == "in_out_degree":
        try:
            if delta != 0:
                addNodes(delta, False)
        except:
            pass

    elif name == "ascendant_has_ad_keyword":  # can only turn this feature from true to false
        # wrapNodes(3)
        wrapNodes_ver2(3)

    elif name == "FEATURE_FIRST_NUMBER_OF_SIBLINGS":
        try:
            if delta > 0:
                IncreaseFirstNumberOfSiblings(delta)
            elif delta < 0:
                DecreaseFirstNumberOfSiblings(-delta)
        except:
            pass

    elif name == "FEATURE_FIRST_PARENT_NUMBER_OF_SIBLINGS":
        if delta > 0:
            try:
                IncreaseFirstParentNumberOfSiblings(delta)
            except:
                pass
        elif delta < 0:
            try:
                DecreaseFirstParentNumberOfSiblings(-delta)
            except:
                pass
            
    elif "FEATURE_FIRST_PARENT_SIBLING_TAG_NAME" in name:
        first_parent_sibling = name.split("FEATURE_FIRST_PARENT_SIBLING_TAG_NAME_")[-1]
        try:
            SetFirstParentSiblingTagName(first_parent_sibling)
        except:
            pass
        
    elif name == "FEATURE_FIRST_PARENT_SIBLING_AD_ATTRIBUTE":
        if delta > 0:
            print("Invalid delta parameter.")
        elif delta < 0:
            try:
                RemoveFirstParentSiblingAdAttribute()
            except:
                pass
            
    elif name == "FEATURE_FIRST_PARENT_INBOUND_CONNECTIONS":
        if delta > 0:
            try:
                IncreaseFirstParentInboundConnections(delta)
            except:
                pass
        elif delta < 0:
            print("Invalid delta parameter.")
            
    elif name == "FEATURE_FIRST_PARENT_OUTBOUND_CONNECTIONS":
        if delta > 0:
            try:
                IncreaseFirstParentOutboundConnections(delta)
            except:
                pass
        elif delta < 0:
            print("Invalid delta parameter.")
            
    elif name == "FEATURE_FIRST_PARENT_INBOUND_OUTBOUND_CONNECTIONS":
        if delta > 0:
            try:
                IncreaseFirstParentInboundOutboundConnections(delta)
            except:
                pass
        elif delta < 0:
            print("Invalid delta parameter.")
            
    elif name == "FEATURE_FIRST_PARENT_AVERAGE_DEGREE_CONNECTIVITY":
        if delta > 0:
            try:
                IncreaseFirstParentAverageDegreeConnectivity(delta)
            except:
                pass
        elif delta < 0:
            try:
                DecreaseFirstParentAverageDegreeConnectivity(-1 * delta)
            except:
                pass
    
    # flow features
    elif name == "num_set_cookie" or name == "num_set_storage":
        try:
            IncreaseSetCookieOrStorage(delta, js_fname)
        except Exception as e:
            print(e)
            pass
        
    elif name == "num_get_cookie" or name == "num_get_storage":
        try:
            IncreaseGetCookieOrStorage(delta, js_fname)
        except Exception as e:
            print(e)
            pass

    elif name == "num_requests_received":
        try:
            duplicate_tag(delta)
        except Exception as e:
            print(e)
            pass

    elif name == "num_requests_sent" or name == "out_degree" or name == "indirect_all_out_degree":
        try:
            IncreaseSetRequestSent(delta, js_fname)
        except Exception as e:
            print(e)
            pass


    # URL Features
    # TODO: lack ctx info for QS and URL length. We can add it if we know we are modifying the features of the same webpage. We know for sure some features would change if modifying other features
    
    elif name == "url_length":
        try:
            ModifyURLLength(delta)
        except Exception as e:
            pass

    elif name == "keyword_raw_present":
        try:
            ModifyADKeyword(delta)
        except:
            pass
    elif name == "keyword_char_present":
        try:
            ModifyADKeyChar(delta)
        except:
            pass
    elif name == "semicolon_in_query":
        try:
            ModifySemicolon(delta)
        except:
            pass
    elif name == "is_valid_qs":
        try:
            ModifyQueryString(delta)
        except:
            pass
    elif name == "base_domain_in_query":
        try:
            ModifyBaseDomainQueryString(delta, domain)
        except:
            pass
    elif name == "ad_size_in_qs_present":
        try:
            ModifyADDimensionInQS(delta)
        except:
            pass
    elif name == "screen_size_present":
        try:
            ModifyScreenDimensionKeywordInQS(delta)
        except:
            pass
    elif name == "ad_size_present":
        try:
            ModifyScreenDimensionInBaseURL(delta)
        except:
            pass
    elif name == "FEATURE_DOMAIN_PARTY":
        try:
            ModifyHostName(delta, domain)
        except:
            pass
    elif name == "FEATURE_SUB_DOMAIN_CHECK":
        try:
            ModifyHostNameWithSubDomain(delta, domain)
        except:
            pass

    after_mapback = str(soup)
    if before_mapback == after_mapback:
        return None, None
    else:
        print("Modification occured!")

    if requestURL != modifiedRequestURL:
        print("Some modifications occured in URL:")
        print("Before: %s | After: %s" % (requestURL, modifiedRequestURL))
    
    return soup, modifiedRequestURL