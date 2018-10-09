
# coding: utf-8

# In[3]:


import time
import random
from selenium import webdriver
import requests
import re
import csv
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from selenium.common.exceptions import NoSuchElementException


# In[4]:



    
def fetching_links(soup):
    
    soup1 = soup
    a_tag = soup1.findAll('a')

    str_a_tag = []
    #converting each entry to string for ease of operation
    for a in a_tag:
        str_a_tag.append(str(a))

    #reomving href
    m = [(a[9:]) for a in str_a_tag]

    #split the resulting string using " to get the link
    s = [a.split('"') for a in m]
    #appending all the links in a new list
    links=[]
    for i in range(len(s)):
        links.append(s[i][0])
    
    return links

def get_data(url):
    

    #for link in links:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(executable_path ='C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/660 Web/chromedriver.exe',                          chrome_options=chrome_options)
    driver.get(url)
    time.sleep(10)
    page_link = requests.get(url)
    time.sleep(10)
    soup_link = BeautifulSoup(page_link.content, 'html.parser')
    data = soup_link.findAll('p')
    time.sleep(10)
    driver.close()
    news_artilce=[]
    news_artilce_str = []
    for a in data:
        news_artilce.append(a.text)
    
    #news_artilce.remove("12 Please verify you're not a robot by clicking the box." or "Invalid email address. Please re-enter."\
    #                    or "You must select a newsletter to subscribe to." or "View all New York Times newsletters")
    news_article_str = " ".join(news_artilce[0:])
    news_article_str1 = news_article_str.replace("Please verify you're not a robot by clicking the box. Invalid email address. Please re-enter. You must select a newsletter to subscribe to. View all New York Times newsletters.",".")
    #print(news_artilce)
    #print(news_article_str)
    return news_article_str1

if __name__ == "__main__": 



    url = ['https://query.nytimes.com/search/sitesearch/#/amazon/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/microsoft/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Bank of America/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Netflix/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Yahoo/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Google/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/JP Morgan/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Apple/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Boeing/30days/allresults/1/allauthors/relevance/Business/'           'https://query.nytimes.com/search/sitesearch/#/Facebook/30days/allresults/1/allauthors/relevance/Business/']        ]


    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(executable_path ='C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/660 Web/chromedriver.exe',                              chrome_options=chrome_options)
    #url = ['https://query.nytimes.com/search/sitesearch/#/google/7days/allresults/1/allauthors/relevance/Business/']

    for a in range(len(url)):
        driver.get(url[a])
        page = requests.get(url[a])

    #creating a list to get links    
    links = []

    soup = BeautifulSoup(page.content, 'html.parser')
    data_div = driver.find_element_by_id('searchResults')
    data_html = data_div.get_attribute('innerHTML')
    soup1 = BeautifulSoup(data_html, 'html5lib')

    links.append(fetching_links(soup1))


    try:
        next_new = driver.find_element_by_class_name('next')

        while  next_new != "":
            next_new = driver.find_element_by_class_name('next')
            next_new.click()
            url_next = driver.current_url
            #print(url_next)
            page = requests.get(url_next)
            time.sleep(10)
            soup2 = BeautifulSoup(page.content, 'html.parser')
            data_div = driver.find_element_by_id('searchResults')
            data_html = data_div.get_attribute('innerHTML')
            soup3 = BeautifulSoup(data_html, 'html5lib')
            time.sleep(10)
            links.append(fetching_links(soup3))
            time.sleep(30)
    except NoSuchElementException: 
        pass
    driver.close()
    print(links)
    links_final = []
    for link in links:
        for l1 in link:
            if l1 not in links_final:
                links_final.append(l1)
    print(links_final)

    #getting the news articles in this list

    news = []    
    for link in links_final:
        print(link)
        news.append(get_data(link))
        print(news)
    #csvfile = 'C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/Project/News.csv'
    #with open(csvfile, "w") as output:
    #    writer = csv.writer(output, lineterminator='\n')
    #    for val in news:
    #        writer.writerow([val]) 
    #print(news)

    my_df = pd.DataFrame(news)
    print(my_df)
    my_df.to_csv ('C:/Users/shrey/OneDrive/Desktop/seaktop/660 Web/Project/news_amazon.csv')


    #print(url[0])
    #start_chrome(url[0])

