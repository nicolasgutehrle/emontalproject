#!/usr/bin/env python
# coding: utf-8

# In[13]:


import requests
from bs4 import BeautifulSoup
import re

import json
from multiprocessing.dummy import Pool
from calmjs.parse import es5
from calmjs.parse.asttypes import VarStatement



# In[2]:


# regex pour trouver resultats totaux
re_max_results = re.compile(r'<b>(\d*)') 
re_script = re.compile(r'</?script[^>]*>')
re_param_url = re.compile(r'(startRecord)=\d+(&maximumRecords)=\d+(&page)=\d+')


# In[3]:


def send_request(url):
    """
    Envoie la requete pour recuperer le html.
    Controle le code retourné.
    Retourne le contenu de la requete. Sinon, retourne l'erreur.
    """
    results = requests.get(url, stream=True)
    if results.status_code == 200: # si on obtient 
        return results
    else:
        raise Exception(f"Erreur {results.status_codes_code}")


# In[4]:


def get_script_tag(soup, filters):
    """
    Retourne tous les tags scripts du html
    """
    return soup.find_all('script', filters)
    


# In[5]:


def get_number_results(url):
    """
    Permet de récupérer le nombre total de résultats associés
    à l'url de Gallia donnée.
    Retourne le nombre total de resultats en str
    """
    request = send_request(url) # get request content
    soup = BeautifulSoup(request.content, 'html') # parse html content
    # on recupere tous les tags scripts, en precisant les attributs type et src
    script_tags = get_script_tag(soup, {'type' : 'text/javascript', 'src': False})
    # on recherche le tag <b> dans le premier script, qui contient les resultats
    search = re.search(re_max_results, str(script_tags[0]))
    return search.group(1)    


# In[6]:


def retrieve_json_from_script(scripttag):
    """
    Parse une variable JS pour recuperer le contenu en JSON
    """
#     parser = Parser()
    # on supprime le tag script du second element obtenus pour pouvoir la parser avec slimit
    javascript = re.sub(re_script, '', scripttag) 
    
    program = es5(javascript)
    json_list = []
    for node in program:
        if isinstance(node, VarStatement):
            children = node.children()[0].children()[1]
            if children:
                # permet de recuperer les variables et JSON dans balise script
                # node.value[1:-1] pour supprimer les '' en debut et fin de script
                json_data = str(children.children()[1].children()[0])[1:-1]
                json_list.append(json_data)
#                 json_list.append(children.children()[1].children()[0])

#     data = str(json_list[0])[1:-1]
    
    
    
#     tree = parser.parse(javascript)

#     json_list = [node.value[1:-1] for node in nodevisitor.visit(tree) 
#                    if isinstance(node, ast.String)]
    
    return json_list


# In[7]:


def get_num_page(total_results):
    """
    Retourne le nombre total de page a parser
    """
    total_results = int(total_results)
    reste = total_results % 50 # 50 est le maximum de resultat que l'on peut obtenir
    if reste != 0:
        return int(total_results / 50) + 1
    return int(total_results / 50)


# In[8]:


def prepare_url_by_page(url, num_page):
    """
    
    """
    startRecord, maximumRecords = 0, 50
    list_url = []
    for i in range(num_page):
        page = i + 1
        new_url = re.sub(re_param_url,r'\1={}\2={}\3={}'.format(startRecord, maximumRecords, page),url)
        list_url.append(new_url)
        startRecord += 50
        maximumRecords += 50
    return list_url
        


# In[9]:


def parse_results(url):

    new_request = send_request(url) # get request content

    soup = BeautifulSoup(new_request.content, 'html') # parse html content

    # on recupere tous les tags scripts, en precisant les attributs type et src
    script_tags = get_script_tag(soup, {'type' : 'text/javascript', 'src': False})

    json_list = retrieve_json_from_script(str(script_tags[1]))

    data = json.loads(json_list[0].encode().decode('unicode-escape'))

    # liste des resultats de recherche
    list_results = data['contenu']['SearchResultsFragment']['contenu']['ResultsFragment']['contenu']
#     # liste ark 
    data_ark = [data['thumb']['url'] for data in list_results]
    
    return data_ark


# In[ ]:


def get_ark_from_url(url):
    """
    """
    ark_position = url.find('12148')
    return url[ark_position + len('12148') + 1 :]


# In[10]:


def get_arks(url):
    # get total results from url request
    total_results = get_number_results(url)
    # nombre de pages de resultats / nombre de fois qu'il faudra faire des requetes
    num_page = get_num_page(total_results)
    
    # list des url: chacune contient 1 page avec au maximum 50 resultats
    list_url = prepare_url_by_page(url, num_page)
    
    pool = Pool(4)
    results = pool.map(parse_results, list_url)
    pool.close()
    pool.join()
    return [get_ark_from_url(data) for result in results for data in result]

