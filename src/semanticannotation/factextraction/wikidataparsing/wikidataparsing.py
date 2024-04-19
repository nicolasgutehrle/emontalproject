import sys 

sys.path.append('..')
import requests
import re
from bs4 import BeautifulSoup
from typing import List, Callable, TypeVar
from multiprocessing.dummy import Pool
import os
from segmenter.segmenter import Segmenter

from datetime import datetime
from babel.dates import format_date
from functools import partial
from rapidfuzz import process, fuzz
import pandas as pd
from glob import glob
import json
from itertools import groupby
from utils.utils import saveWhatLinksHere, saveWikidataLinks, loadWhatLinksHereLinks, saveEntitiesData

class WhatLinksHere:

    re_wikiHref = re.compile(r'/wiki/Q.*')
    url = f'https://www.wikidata.org/w/index.php?title=Special:WhatLinksHere'

    def __getNextPageURL(self, entity_type:str, url: str, results: List[str], limit: int = 50, m_size: int = 0, folderpath: str = "", save_step: int = 0 ) -> List[str]:
        """
        NOT TO USE DIRECTLY  
        Finds the URL leading the next limit number of entities on WhatLinksHere pages recursively.

        :param url: URL to parse
        :type url: str
        :param results: list of already processed urls
        :type results: List[str]
        :param limit: limit of URL to process at the same time, defaults to 50
        :type limit: int, optional
        :param m_size: maximum of url to process, defaults to 0
        :type m_size: int, optional
        :param folderpath: path to folder containing 'whatlinkshere.txt' txt file where results are save , defaults to None
        :type folderpath: str, optional
        :param save_step: specify at which every step to save results, defaults to 0
        :type save_step: int, optional
        :return: list of URL pages which contain sets of link to Wikidata entity pages
        :rtype: List[str]
        """
        if len(results) % save_step == 0 and save_step != 0:
            if folderpath:
                saveWhatLinksHere(entity_type=entity_type,savepath=folderpath, list_urls=results)
                # with open(f"{folderpath}", 'w', encoding='utf-8') as f:
                #     f.write('\n'.join(results))
            else:
                raise Exception('A path to a file with a .txt extension must be given')

        if len(results) == m_size and m_size != 0:
            return results
        else:
            req = requests.get(url)
            soup = BeautifulSoup(req.content, 'lxml')
            next_tag = soup.find('a', string=f"next {limit}")

            if next_tag:
                next_url = f"https://www.wikidata.org{next_tag['href']}"
                results.append(next_url)
                results = self.__getNextPageURL(entity_type=entity_type, url = next_url, results=results, limit=limit, m_size=m_size, folderpath=folderpath, save_step=save_step)
                return results
            
            else:
                
                return results


    def getWhatLinksHere(self, entitytype:str, limit: int = 100, m_size: int = 0, save_step: int = 10, folderpath: str="") -> List[str]:
        """
        Wrapper to retrieve and save on disk WhatLinksHere pages

        :param entitytype: type of entity to find
        :type entitytype: str
        :param limit: limit of URL to process at the same time, defaults to 50
        :type limit: int, optional
        :param m_size: maximum of url to process, defaults to 0
        :type m_size: int, optional
        :param save_step: specify at which every step to save results, defaults to 0
        :type save_step: int, optional
        :param folderpath: path to folder containing 'whatlinkshere.txt' file where results are saved , defaults to None
        :type folderpath: str, optional
        :return: list of URL pages which contain sets of link to Wikidata entity pages
        :rtype: str
        """

        def process(ent:str):
            print(f'Processing {ent} type...')

            data = loadWhatLinksHereLinks(entity_type=ent, folderpath=folderpath)
            if data['urls']:
                print('Starting from last parsed url...')
                url = data['urls'][-1]
                data['urls'] = self.__getNextPageURL(entity_type=ent, url=url, results=data['urls'], limit=limit, m_size = m_size, folderpath=folderpath, save_step=save_step)
            else:
                print('Starting from first url')
                url = f'{self.url}/{ent}&namespace=0&limit={limit}'
                data['urls'].append(url)
                data['urls'] = self.__getNextPageURL(entity_type=ent, url=url, results=data['urls'], limit=limit, m_size = m_size, folderpath=folderpath, save_step=save_step)

            print(f'Done processing {ent} type')
            if folderpath:
                saveWhatLinksHere(entity_type=ent,savepath=folderpath, list_urls=data['urls'])

            return data

        results = []

        if isinstance(entitytype, list):
            for ent in entitytype:
                results.append(process(ent=ent))
        else:
            results.append(process(ent=entitytype))

        return results


    def getWikidataLinks(self, dict_url: dict, savepath:str="") -> List[str]:
        """
        Parses a WhatLinksHere page to get links to Wikidata entry

        :param url: URL to parse
        :type url: str
        :param savepath: Path to save folder
        :type savepath: str
        :return: List of string containing the ID of each entity in the URL
        :rtype: List[str]
        """
        print(f'Processing {dict_url["type"]} type...')
        data = {
            "type": dict_url['type'],
            "ent_id": [] 
        }
        for url in dict_url['urls']:
            req = requests.get(url)
            soup = BeautifulSoup(req.content, 'lxml')
            tag_ul = soup.find('ul', {'id' : 'mw-whatlinkshere-list'})
            list_anchor = tag_ul.find_all('a', {'href' : self.re_wikiHref})

            list_anchor = [a['href'] for a in list_anchor]
            # TODO : faire en sorte de retirer Q...?response=no
            list_anchor = [a.replace('/wiki/', '') for a in list_anchor]
            list_anchor = list(filter(lambda x: not '?' in x, list_anchor))
            data['ent_id'].extend(list_anchor)
        
        if savepath:
            saveWikidataLinks(savepath=savepath, data=data)
        print(f'Processing {dict_url["type"]} done')

        return data

    def multi_getWikidataLinks(self, list_urls: List[dict], n_core:int=4, savepath:str = "") -> List[str]:
        """
        Applies getWikidataLinks to multiple URL at once, using parallel processing

        :param list_urls: List of URL to process
        :type list_urls: List[str]
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Path to save folder
        :type savepath: str
        :return: _description_
        :rtype: List[str]
        """
        # list_data = []
        with Pool(n_core) as p:
            list_data = p.map(self.getWikidataLinks, list_urls)

        # list_data = [x for y in list_data for x in y]
        if savepath:
            saveWikidataLinks(savepath=savepath, data=list_data)

        return list_data


# %%

class WikidataParser:
    
    base_url_entity = "https://www.wikidata.org/wiki/Special:EntityData"

    def __init__(self, lg:str) -> None:
        """
        WikidataParser, to parse and collect data about entities on Wikidata

        :param lg: Language in which to process data, especially for the sentence Segmenter
        :type lg: str
        """
        self.lg = lg
        self.seg = Segmenter(self.lg)


    def getEntityData(self, entityID: str, entityType: str = '', format: str = 'json', savepath : str = "") -> dict:
        """
        Retrieves data about given entity and returns it with given format (default: json)

        :param entityID: ID of entity to process
        :type entityID: str
        :param entityType: Type of the entity selected, e.g. Q5
        :type entityType: str
        :param format: File format under which results are saved, defaults to 'json'
        :type format: str, optional
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Data about the entity from its corresponding Wikidata page
        :rtype: dict
        """
        req = requests.get(f"{self.base_url_entity}/{entityID}.{format}")
        # the key "property" is to be filled up later
        entityData = {'id': entityID, 'type': entityType, 'data': req.json(), 'properties': []}

        if savepath:
            saveEntitiesData(savepath=savepath, entityData=entityData)
            # with open(f"{savepath}/{entityID}.json", 'w', encoding='utf-8') as f:
            #     json.dump(data, f, indent=4)

        return entityData

    def multi_getEntityData(self, list_entities:List[dict], n_core:int=4, savepath:str='')-> List[dict]:
        """
        Applies getEntityData in a parallel processing manner 

        :param list_entities: List of entity ids to process
        :type list_entities: List[str]
        :param entityType: Type of the entity selected, e.g. Q5
        :type entityType: str
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Folder where to save data, defaults to ''
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities data
        :rtype: List[dict]
        """
        list_entities_data = []
        for dict_data in list_entities:
            print(f'Processing {dict_data["type"]} type...')
            with Pool(n_core) as p:
                partial_func = partial(self.getEntityData, entityType=dict_data['type'], savepath=savepath)
                list_entities_data.extend(p.map(partial_func, dict_data['ent_id']))
            print(f"Processing {dict_data['type']} done")
        return list_entities_data

    def getEntityLabels(self, entityData: dict, default_lg: str = 'en', savepath : str = '') -> dict:
        """
        Returns list of labels and aliases of given entity in given language. If selected language is not available, will select default language

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param lg: language in which to retrieve the entity labels and aliases
        :type lg: str
        :param default_lg: backup language if selected language is not available, defaults to 'en'
        :type default_lg: str, optional
        :param savepath: path to file to save results, defaults to None
        :type savepath: str, optional
        :return: Updated entityData dictionary with labels
        :rtype: dict
        """

        def getAliases(labels: List[str]):

            aliases = data['aliases'][self.lg]
            func_labels_append = labels.append
            for i in range(len(aliases)):
                func_labels_append(aliases[i]['value'])
            return labels

        entityID = entityData['id']
        # print(entityID)

        try:
            data = entityData['data']['entities'][entityID]
            # first gets its main label
            if self.lg in data['labels']:
                labels = [data['labels'][self.lg]['value']]

            else:
                # by default, selects text in English if 
                # selected language is not available
                labels = [data['labels'][default_lg]['value']]

            # adds aliases if they exist in the given language
            # and if there are any alias
            if len(data['aliases']) != 0:
                try:
                    if self.lg in data['aliases']:
                        labels = getAliases(labels)
                    # else:
                    #     labels = getAliases(labels, default_lg)
                # if neither selected or default language are avaible
                except KeyError:
                    entityData['labels'] = labels
                    return entityData    

            entityData['labels'] = labels

            if savepath:
                saveEntitiesData(savepath=savepath, entityData=entityData)

                # with open(f"{savepath}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
                #     json.dump(entityData, f, indent=4)
                    
            return entityData

        # if neither selected or default language are avaible
        except KeyError:
            if savepath:
                os.system(f"rm {savepath}/entity_data/{entityData['type']}/{entityData['id']}.json")    

    def multi_getEntityLabels(self, list_entities_data: List[dict], n_core:int =4 , savepath:str='') -> List[dict]:
        """
        Applies getEntityLabels in parallel processing

        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Folder where to save data, defaults to ''
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.getEntityLabels, savepath=savepath)
            list_entities_data = p.map(partial_func, list_entities_data)
            list_entities_data = [x for x in list_entities_data if x]

        return list_entities_data


    def getEntityWikipediaContent(self, entityData: dict, toSent= True, savepath : str = '') -> dict:
        """
        Parses Wikipedia page associated with an entity to retrieve text in p tags

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param lg: Language in which to process data
        :type lg: str
        :param toSent: Post process p tags into sentences, defaults to True
        :type toSent: bool, optional
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Updated entityData dictionary with Wikipedia content
        :rtype: dict
        """

        entityID = entityData['id']
        data = entityData['data']['entities'][entityID]['sitelinks']
        
        try:
            url = data[f'{self.lg}wiki']['url']
            req = requests.get(url)
            wikipedia_page = BeautifulSoup(req.content, 'lxml')
            entityData['wikipedia'] = {'url': url, 'doc': wikipedia_page}
            
            p_tags = wikipedia_page.find_all('p')

            p_tags = [p.text for p in p_tags]

            if toSent:
                all_sents = []
                for doc in p_tags:
                    all_sents.extend(self.seg.getSentences(doc))
                entityData['wikipedia']['content'] = all_sents

            else:
                entityData['wikipedia']['content'] = p_tags

            del entityData['wikipedia']['doc']

            if savepath:
                saveEntitiesData(savepath=savepath, entityData=entityData)

                # with open(f"{savepath}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
                #     json.dump(entityData, f, indent=4)
                    
            return entityData

        except:
            if savepath:
                os.system(f"rm {savepath}/entity_data/{entityData['type']}/{entityData['id']}.json")    

    def multi_getEntityWikipediaContent(self, list_entities_data: List[dict], n_core:int =4 , savepath:str='') -> List[dict]:
        """
        Applies getEntityWikipediaContent in parallel processing

        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Folder where to save data, defaults to ''
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.getEntityWikipediaContent, savepath=savepath)
            list_entities_data = p.map(partial_func, list_entities_data)
            list_entities_data = [x for x in list_entities_data if x]

        return list_entities_data            

    def translateDate(self, datevalue: str) -> str:
        """
        Translate datetime into full text in given language

        :param datevalue: Date to translate in date str format
        :type datevalue: str
        :return: Translated date
        :rtype: str
        """
        # parse standard format
        try:
            str2datetime = datetime.strptime(datevalue, "+%Y-%m-%dT%H:%M:%SZ")
            time2date = str2datetime.date()
            return format_date(time2date, format='long', locale=self.lg)
        except:
            # error usually when there's only a year, without month or day
            # in that case, keeps only the year

            year = datevalue[1:5]
            zero = re.match(r'^0{1,3}', year[:4])

            if zero:
                span = zero.end()
                year = year[span:]

            if datevalue.startswith('-'):
                year = f"-{year}"
            
            return year


    def getProperty4Entity(self, entityData: dict, dict_prop: List[dict], savepath : str = '') -> dict:
        """
        Associate each selected properties with possible label mentions 

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param lg: Language in which the entity labels and aliases are written
        :type lg: str
        :param dict_prop: Dictionnary of properties to collect, where keys are the property id and values are a chosen label (e.g. {"P19": "placeOfBirth"})
        :type dict_prop: dict
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Updated entity data dictionary with properties and their possible expressing labels
        :rtype: dict
        """

        entityID = entityData['id']
        data = entityData['data']['entities'][entityID]['claims']

        ent_prop = [x for x in dict_prop if x['type'] == entityData['type']][0]
        # print(entityID, entityData['type'], ent_prop)
        # if propertyID in data:
        for propertyID in ent_prop['props'].keys():
            list_propvalues = []

            if propertyID in data:

                func_propvalues_append = list_propvalues.append

                for i in range(len(data[propertyID])):
                    try:
                        valueData = data[propertyID][i]['mainsnak']['datavalue']

                        if valueData['type'] == 'wikibase-entityid':
                            value_entityId = valueData['value']['id']
                            value_entityData = self.getEntityData(value_entityId)

                            ent_labels = self.getEntityLabels(value_entityData, self.lg)

                            if ent_labels:
                                value = [{
                                    "value": x,
                                    "type": value_entityId
                                } for x in ent_labels['labels']]
                            else:
                                value = []
                                    
                        elif valueData['type'] == 'time':
                            time_value = self.translateDate(valueData['value']['time'])
                            value = {
                                "value": time_value,
                                "type": "time"
                            }

                        elif valueData['type'] == 'quantity':
                            q_value = valueData['value']['amount']
                            value = {
                                "value": q_value,
                                "type": 'quantity'
                            }

                        elif valueData['type'] == 'string':
                            s_value = valueData['value']
                            value = {
                                "value": s_value,
                                "type": "string"
                            }
                        
                    # the value for this property is empty
                    except: 
                        continue

                    if isinstance(value, list):
                        list_propvalues.extend(value)
                    else:
                        func_propvalues_append(value)

                entityData['properties'].append(
                {
                    "propertyID": propertyID,
                    "values": list_propvalues,
                }
                )
                # entityData['properties'][propertyID] = list_propvalues

        if savepath:
            saveEntitiesData(savepath=savepath, entityData=entityData)

            # with open(f"{savepath}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
            #     json.dump(entityData, f, indent=4)
                
        return entityData
    
    def multi_getProperty4Entity(self, list_entities_data:List[dict], dict_prop:dict, n_core:int=4, savepath:str='') -> List[dict]:
        """
        Applies getProperty4Entity in parrallel processing
        
        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param dict_prop: Dictionnary of properties to collect, where keys are the property id and values are a chosen label (e.g. {"P19": "placeOfBirth"})
        :type dict_prop: dict
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Folder where to save data, defaults to ''
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """
        
        with Pool(n_core) as p:
            partial_func = partial(self.getProperty4Entity, dict_prop = dict_prop, savepath=savepath)
            list_entities_data = p.map(partial_func, list_entities_data)

        return list_entities_data


    def __measureSim(self, sentences: List[str], values: List[str], scorer: Callable[[str], float], score_cutoff : float, filter: bool = True) -> pd.DataFrame:
        """
        NOT TO USE DIRECTLY
        Uses rapidfuzz to measure similarity between sentences and set of values. Returns DataFrame with only values different than 0

        :param sentences: List of sentence to check
        :type sentences: List[str]
        :param values: Values to find in sentences
        :type values: List[str]
        :param scorer: Scorer function to use for fuzzy matching between sentences and labels
        :type scorer: Callable
        :param score_cutoff: Minimum similarity threshold to reach
        :type score_cutoff: float
        :param filter: Option to remove empty cells, defaults to True
        :type filter: bool, optional
        :return: Dataframe of sentence - labels similarity scores
        :rtype: pd.DataFrame
        """

        sim_matrix = process.cdist(sentences, values, scorer=scorer, score_cutoff = score_cutoff)
        
        # outputs a matrix of N sentences by M properties
        df = pd.DataFrame(sim_matrix)

        # only keeps row where there at least one value not NaN
        if filter:
            df = df[df.iloc[:, :] > 0]
            df.dropna(inplace=True, how='all')

        return df

    def __findSourceinSent(self, labels: List[str], sentences: List[str]) -> List[str]:
        """
        NOT TO USE DIRECTLY
        Regex to find which entity label appears in each sentence

        :param labels: Entity labels or aliases to search in sentences. Must be exact match
        :type labels: List[str]
        :param sentences: Sentences in which to search for entity mentions
        :type sentences: List[str]
        :return: List of sentences where there was a match
        :rtype: List[str]
        """

        search_labels = [re.escape(x) for x in labels]
        search_labels = "|".join(search_labels)
        re_labels = re.compile(f'({search_labels})')
        # print(re_labels)
        list_match = []
        func_list_match = list_match.append
        for sent in sentences:
            search_re = re_labels.search(sent)
            if search_re:
                func_list_match(search_re.group(0))
            else:
                # adds a NO-MATCH "sentence" if there's no match
                func_list_match('NO-MATCH')
        return list_match


    def findSentences(self, entityData: dict, source_doc: str = 'wikipedia', scorer:Callable = fuzz.partial_ratio,  score_cutoff: int = 90, savepath : str = '') -> dict:
        """
        Finds most similar sentences in source document to the Source and Target values of each properties selected for this entity

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param source_doc: Document where to find sentences, defaults to 'wikipedia'
        :type source_doc: str, optional
        :param scorer: Scorer function to use for fuzzy matching between sentences and labels
        :type scorer: function
        :param score_cutoff: Minimum similarity threshold to reach
        :type score_cutoff: int
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Updated entity data dictionary with sentences matching properties
        :rtype: dict
        """

        all_sents = entityData[source_doc]['content']
        # finds entity labels appearing in each sentence
        list_match = self.__findSourceinSent(entityData['labels'], all_sents)

        for prop_data in entityData['properties']:
            # finds sentences where properties are mentioned 
            df_target = self.__measureSim(all_sents, [x['value'] for x in prop_data['values']], scorer=scorer, score_cutoff=score_cutoff)

            selected_sents = []
            func_append = selected_sents.append

            # iterates over the matrix to retrieve sentences, source
            # and targets
            for c in df_target.columns:
                target = prop_data['values'][c]
                target_label = target['value']
                target_type = target['type']

                for i in df_target.index:
                    sent = all_sents[i]
                    sim = df_target.loc[i][c]
                    source_value = list_match[i]

                    if sim > 0:
                        func_append(
                            {
                                "prop": prop_data['propertyID'],
                                "sent": sent,
                                'source': source_value,
                                "source_type": entityData['type'],
                                "target": target_label,
                                "target_type": target_type
                                # "sim": sim
                            }
                            )

            prop_data['sents'] = selected_sents

        if savepath:
            saveEntitiesData(savepath=savepath, entityData=entityData)

            # with open(f"{savepath}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
            #     json.dump(entityData, f, indent=4)
                
        return entityData

    def multi_findSentences(self, list_entities_data:List[dict], source_doc: str = 'wikipedia', n_core:int=4, scorer:Callable = fuzz.partial_ratio,  score_cutoff: int = 90, savepath : str = '') -> List[dict]:
        """
        Applies findSentences in parallel processing

        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param source_doc: Document where to find sentences, defaults to 'wikipedia'
        :type source_doc: str, optional
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param scorer: Scorer function to use for fuzzy matching between sentences and labels
        :type scorer: function
        :param score_cutoff: Minimum similarity threshold to reach
        :type score_cutoff: float
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.findSentences, source_doc = source_doc, scorer = scorer, score_cutoff=score_cutoff, savepath=savepath)

            list_entities_data = p.map(partial_func, list_entities_data)

        return list_entities_data

    def getOtherSentences(self, entityData: dict, maxsizesent:False, maxsize:int=0, savepath : str = '') -> dict:
        """
        Labels as Other any sentence not labelled already with another category. Only use this function if you're processing an evaluation dataset

        :param entityData: Dictionary containing data about an entity
        :type entityData: dict
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Updated entity data dict with sentences labelled as Other
        :rtype: dict
        """

        all_labeled_sents = set([sent['sent'] for prop in entityData['properties'] for sent in prop['sents'] ])
        all_unlabeled_sent = set([sent for sent in entityData['wikipedia']['content']])
        all_unlabeled_sent = all_unlabeled_sent - all_labeled_sents

        if maxsize and maxsizesent:
            raise Exception('Please only give as argument either maxsizesent or maxsize')

        if maxsize:
            all_unlabeled_sent = list(all_unlabeled_sent)
            all_unlabeled_sent = all_unlabeled_sent[:maxsize]
        
        if maxsizesent:
            all_unlabeled_sent = list(all_unlabeled_sent)
            all_unlabeled_sent = all_unlabeled_sent[:len(all_labeled_sents)]

        entityData['properties'].append(
            {
                'propertyID': 'Other',
                'sents': [
                    {
                        'prop': 'Other',
                        'sent': x,
                        'source': 'NO-SOURCE',
                        'target': 'NO-TARGET'
                        
                    } for x in all_unlabeled_sent
                ]
            }
        )


        if savepath:
            saveEntitiesData(savepath=savepath, entityData=entityData)

            # with open(f"{savepath}/{entityData['id']}.json", 'w', encoding='utf-8') as f:
            #     json.dump(entityData, f, indent=4)

        return entityData
    
    def multi_getOtherSentences(self,list_entities_data:List[dict], maxsizesent:bool=False, maxsize:int = 0, n_core:int = 4,savepath:str='') -> List[dict]:
        """
        Applies getOtherSentences in parallel processing

        :param list_entities_data: List of dictionnaries containing entities data
        :type list_entities_data: List[dict]
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: _description_, defaults to ''
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Return list of dictionnaries containing entities with updated data
        :rtype: List[dict]
        """

        with Pool(n_core) as p:
            partial_func = partial(self.getOtherSentences, maxsizesent=maxsizesent, maxsize = maxsize,savepath=savepath)
            list_entities_data = p.map(partial_func, list_entities_data)
        return list_entities_data
 
    def processListEntities(self, list_entities: List[str], dict_prop:dict, scorer:Callable = fuzz.partial_ratio, source_doc:str='wikipedia', score_cutoff:int=90, getOther:bool = False, maxsizesent:bool=False, maxsize:int = 0,n_core:int=4, savepath:str='') -> List[dict]:
        """
        Wrapper function to process list of entities ID obtained from WikidataLinks obtained with WhatLinksHere

        :param list_entities: List of entity ID from Wikidata to process
        :type list_entities: List[str]
        :param entityType: Type of the entity selected, e.g. Q5
        :type entityType: str
        :param dict_prop: Dictionnary of properties to collect, where keys are the property id and values are a chosen label (e.g. {"P19": "placeOfBirth"})
        :type dict_prop: dict
        :param scorer: Scorer function to use for fuzzy matching between sentences and labels
        :type scorer: Callable[[str], float]
        :param source_doc: Document where to find sentences, defaults to 'wikipedia'
        :type source_doc: str, optional
        :param score_cutoff: Minimum similarity threshold to reach, defaults to 90
        :type score_cutoff: int, optional
        :param n_core: Number of cores to use for parallel processing, defaults to 4
        :type n_core: int, optional
        :param savepath: Path to save file, defaults to None
        :type savepath: str, optional
        :return: Processed entity with their data
        :rtype: dict
        """
        maxstep = 5
        if getOther:
            maxstep = 6

        print(f'Step 1/{maxstep}')
        print('Collecting Entity data...')
        list_entities_data = self.multi_getEntityData(list_entities=list_entities, n_core=n_core, savepath=savepath)
        print('Entity data collected')
        
        print(f'Step 2/{maxstep}')
        print('Collecting content...')
        list_entities_data = self.multi_getEntityWikipediaContent(list_entities_data=list_entities_data,n_core=n_core, savepath=savepath)
        print('Content collected')
        
        print(f'Step 3/{maxstep}')
        print('Collecting Entity labels...')
        list_entities_data = self.multi_getEntityLabels(list_entities_data=list_entities_data,n_core=n_core, savepath=savepath)
        print('Entity labels collected')


        print(f'Step 4/{maxstep}')
        print('Collecting properties...')
        list_entities_data = self.multi_getProperty4Entity(list_entities_data=list_entities_data, dict_prop=dict_prop, n_core=n_core, savepath=savepath)
        print('Properties collected')

        print(f'Step 5/{maxstep}')
        print('Collecting sentences...')
        list_entities_data = self.multi_findSentences(list_entities_data=list_entities_data, source_doc=source_doc, n_core=n_core, scorer=scorer, score_cutoff=score_cutoff, savepath=savepath)
        print('Sentences collected')

        if getOther:
            print(f'Step {maxstep}/{maxstep}')
            print('Collecting Other sentences...')
            list_entities_data = self.multi_getOtherSentences(list_entities_data=list_entities_data, maxsizesent=maxsizesent, maxsize=maxsize, n_core=n_core, savepath=savepath)
            print('Other sentences collected')


        return list_entities_data