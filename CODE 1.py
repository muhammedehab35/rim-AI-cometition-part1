#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 1 ere partie 
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from hashlib import sha1
def telecharger_photo(photo, dossier):
    try:
        response = requests.get(photo, stream=True)
        response.raise_for_status()
        cn = response.headers.get('content-type')
        ext = '.jpg' if 'image/jpeg' in cn else '.png'
        nom = generate_unique_filename(photo, dossier) + ext
        with open(nom, 'wb') as f:
            for x in response.iter_content(1024):
                f.write(x)
def scrape(lien, dossier='images'):
    try:
        response = requests.get(lien)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tous_les_images = soup.find_all('img')
        for img in tous_les_images:
            photo = img.get('src')
            if lien:
                photo = urljoin(lien, photo)
                telecharger_photo(photo, dossier)
lien1='https://www.voursa.com/index.cfm?gct=1&sct=11&gv=13'
lien2='https://www.voursa.com/index.cfm?PN=2&gct=1&sct=11&gv=13'
lien3='https://www.voursa.com/index.cfm?PN=4&gct=1&sct=11&gv=13'
lien4='https://www.voursa.com/index.cfm?PN=3&gct=1&sct=11&gv=13'
lien5='https://www.voursa.com/index.cfm?PN=6&gct=1&sct=11&gv=13'
lien6='https://www.voursa.com/index.cfm?PN=5&gct=1&sct=11&gv=13'
lien7='https://www.voursa.com/index.cfm?PN=7&gct=1&sct=11&gv=13'
lien8='https://www.voursa.com/index.cfm?PN=8&gct=1&sct=11&gv=13'
lie19='https://www.voursa.com/index.cfm?PN=9&gct=1&sct=11&gv=13'
lien10='https://www.voursa.com/index.cfm?PN=10&gct=1&sct=11&gv=13'
liens=[lien1,lien2,lien3,lien4,lien5,lien6,lien7,lien8,lien9,lien10]
for lien in liens:
    scrape(lien)


# In[ ]:




