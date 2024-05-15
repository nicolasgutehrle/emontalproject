#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append('..')

from bs4 import BeautifulSoup
from bs4.element import Tag
import re

import json
from multiprocessing.dummy import Pool
from calmjs.parse import es5
from calmjs.parse.asttypes import VarStatement

from connector.xmlconnector import XMLConnector
from connector.folderconnector import FolderConnector
from document import Gallica_Document

from time import sleep

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import subprocess
from typing import List 


class GallicaInterface:

    re_max_results = re.compile(r'<b>(\d*)')
    # to get script tags in HTML files
    re_script = re.compile(r'</?script[^>]*>')
    re_param_url = re.compile(r'(startRecord)=\d+(&maximumRecords)=\d+(&page)=\d+')
    xpath_ocr = '//ark[text() = "{document_ark}"]/following-sibling::ocr'

    def __init__(self, database_name, database_type, new_db=False):
        """
        Constructor

        :param database_name: name of database or folder where data will be saved
        :param database_type: Must be either (folder, xml)
        """
        # choose type of databasem either in XMLdatabase or in folder
        if database_type == 'xml':
            self.connector = XMLConnector('localhost', 1984, 'admin', 'admin', database_name, new_db=new_db)
        elif database_type == 'folder':
            self.connector = FolderConnector(database_name, new_db=new_db)

        self.request_session = self.init_request_session()

    def init_request_session(self) -> requests.Session:
        """
        Initiate request session, to be used to collect data

        :return: The session
        :rtype: requests.Session
        """

        session = requests.Session()
        retry = Retry(connect=5, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def send_request(self, url:str) -> str:
        """
        Sends the request to retrieve the html. Checks the returned code. Returns the contents of the request. Otherwise, returns the error.


        :param url: Url to request
        :type url: str
        :return: Content of the returned response
        :rtype: str
        """

        results = self.request_session.get(url, stream=True)
        results.encoding = 'utf-8'

        if results.status_code == 200: # si on obtient
            return results.content
        else:
            print(url)
            print(results)
            sleep(60)
            return self.send_request(url)
            # raise Exception(f"Erreur {results.status_code}")
            # return None

    def _get_script_tag(self, soup:BeautifulSoup, filters:dict) -> List[Tag]:
        """
        Find all script tags in HTML, based on filters such as id attributes

        :param soup: BeautifulSoup object to search
        :type soup: BeautifulSoup
        :param filters: Dictionary of key:values to filter script tags
        :type filters: dict
        :return: list of script tags found in document
        :rtype: List[Tag]
        """
        return soup.find_all('script', filters)

    def _get_number_results(self, url:str) -> str:
        """
        Retrieves the total number of results associated with the given Gallia url. Returns the total number of results in str

        :param url: Url to search
        :type url: str
        :return: Number of document found in a URL, as a str
        :rtype: str
        """
        request = self.send_request(url) # get request content
        soup = BeautifulSoup(request, 'html') # parse html content
        # we retrieve all the script tags, specifying the type and src attributes
        script_tags = self._get_script_tag(soup, {'type' : 'text/javascript', 'src': False})
        # search for the <b> tag in the first script, which contains the results
        search = re.search(self.re_max_results, str(script_tags[0]))
        return search.group(1)

    def _retrieve_json_from_script(self, scripttag:Tag) -> List[Tag]:
        """
        Parse a JS variable to retrieve JSON content

        :param scripttag: Script tag to parse
        :type scripttag: Tag
        :return: JSON content in script tag
        :rtype: List[Tag]
        """
        # remove the script tag from the second element obtained so that it can be parsed with slimit
        javascript = re.sub(self.re_script, '', scripttag)

        program = es5(javascript)
        json_list = []
        for node in program:
            if isinstance(node, VarStatement):
                children = node.children()[0].children()[1]
                if children:
                    # retrieves variables and JSON in script tags
                    # node.value[1:-1] to remove the '' at the beginning and end of the script
                    json_data = str(children.children()[1].children()[0])[1:-1]
                    json_list.append(json_data)

        return json_list

    def _get_num_page(self, total_results:str) -> int:
        """
        Returns number of pages from document to parse

        :param total_results: Number od documents, as returned by _get_number_results
        :type total_results: str
        :return: Total number of pages in documents
        :rtype: int
        """
        total_results = int(total_results)
        reste = total_results % 50 # 50 est le maximum de resultat que l'on peut obtenir
        if reste != 0:
            return int(total_results / 50) + 1
        return int(total_results / 50)

    def _prepare_url_by_page(self, url:str, num_page:int) -> List[str]:
        """
        Prepares list of URL to parse, where URL is the base URL of the document, followed by the page number

        :param url: Base url of the document to parse
        :type url: str
        :param num_page: Total number of pages in the document
        :type num_page: int
        :return: List of prepared URLs
        :rtype: List[str]
        """
        startRecord, maximumRecords = 0, 50
        list_url = []
        for i in range(num_page):
            page = i + 1
            new_url = re.sub(self.re_param_url,r'\1={}\2={}\3={}'.format(startRecord,
                                                                         maximumRecords,
                                                                         page),url)
            list_url.append(new_url)
            startRecord += 50
            maximumRecords += 50
        return list_url

    def _parse_results(self, url:str) -> List[str]:
        """
        Collects list of ark ids from base URL of the collection in Gallica

        :param url: Base url of the collection in Gallica
        :type url: str
        :return: List of ark ids found in the collection
        :rtype: List[str]
        """

        new_request = self.send_request(url) # get request content

        soup = BeautifulSoup(new_request, 'html') # parse html content

        # we retrieve all the script tags, specifying the type and src attributes
        script_tags = self._get_script_tag(soup, {'type' : 'text/javascript', 'src': False})

        json_list = self._retrieve_json_from_script(str(script_tags[1]))

        data = json.loads(json_list[0].encode().decode('unicode-escape'))

        # list of search results
        list_results = data['contenu']['SearchResultsFragment']['contenu']['ResultsFragment']['contenu']
    #     # list of ark ids
        data_ark = [data['thumb']['url'] for data in list_results]

        return data_ark

    def _get_ark_from_url(self, url:str) -> str:
        """
        Find ark id in URL

        :param url: URL containing an ark id
        :type url: str
        :return: Ark id as found in the url
        :rtype: str
        """
        ark_position = url.find('12148')
        return url[ark_position + len('12148') + 1 :]

    def get_arks(self, url:str) -> List[str]:
        """
        Main function to collect ark ids from initial URL of document collection in Gallica

        :param url: Base URL of the collection to parse
        :type url: str
        :return: List of ark ids found in collection
        :rtype: List[str]
        """
        # get total results from url request
        total_results = self._get_number_results(url)
        # number of pages of results / number of times queries will need to be made
        num_page = self._get_num_page(total_results)

        # url list: each url contains 1 page with a maximum of 50 results
        list_url = self._prepare_url_by_page(url, num_page)

        with Pool(4) as p:
            results = p.map(self._parse_results, list_url)

        return [self._get_ark_from_url(data) for result in results for data in result]

    def _save_document(self, docname:str, document:BeautifulSoup) -> None:
        """
        Helper function to save document, either as a file on disk or in an XMLBase database

        :param docname: Filename
        :type docname: str
        :param document: Document to save
        :type document: BeautifulSoup
        """
        self.connector.make_dir(docname)
        if not document.sub_docs:
            # self.connector.save(f"{docname}.xml", document.prettify())
            self.connector.save(f"{docname}/{docname}.xml", document.prettify())

        else:
            # delete subd_docs tags from doc
            subdocs = document.sub_docs.extract()
            # self.connector.save(f"{docname}.xml", document.prettify())
            self.connector.save(f"{docname}/{docname}.xml", document.prettify())

            for subdoc in subdocs:
                subark = subdoc.ark.text.strip()
                subfolder = f"{docname}/{subark}"
                self.connector.make_dir(subfolder)
                self.connector.save(f"{subfolder}/{subark}.xml", subdoc.prettify())

    def launch_subprocess(self, command:str) -> None:
        """
        Helper function to launch subprocess. The command-line is given as str and must have every parameter completed.

        :param command: Command line to launch
        :type command: str
        """
        subprocess.Popen(command, shell=True)

    def _download_ocr(self, document:BeautifulSoup) -> bool:
        """
        By default, the collected document don't have the OCR tag. This function downloads this tag for a given document.

        :param document: Document to collect OCR for
        :type document: BeautifulSoup
        :return: Either True if OCR has been collected, else False
        :rtype: bool
        """

        print('Retrieving XML')
        # retrieves each OCR page
        ocr_tag = document.find('ocr')
        if ocr_tag:
            ocr_page = ocr_tag.find_all('page', recursive=False)
            # retrieves each url directing to OCR
            list_ocrlink = [page.ocrlink.text.strip() for page in ocr_page if not page.alto]
            # download OCR XML,
            print('Downloading OCR from source')
            with Pool(4) as p:
                results = p.map(self.send_request, list_ocrlink)
            print('Updating XML tree')
            # results = [self.send_request(url) for url in list_ocrlink]
            for old, content in zip(ocr_page, results):
                soup_content = BeautifulSoup(content.decode('utf-8'), features='lxml')
                old.append(soup_content.alto)
            return True
        else:
            return False

    def _filter_documents(self, filter:str) -> List[str]:
        """
        Return document ID where document doesn't have given filter. For instance, if filter is 'ocr', return any document that doesnt have an ocr attribute

        :param filter: Attribute to use to filter tags
        :type filter: str
        :return: List of ark ids that has been kept after filtering
        :rtype: List[str]
        """
        list_to_do = []
        list_doc = [doc for doc in self.connector.stat_file.get_collection().find_all('document')
                    if not doc.has_attr(filter)]
        for doc in list_doc:
            if doc.subdocument:
                list_ark = [subdoc['id'] for subdoc in doc.find_all('subdocument')
                            if not subdoc.has_attr(filter)]
            else:
                list_ark = [doc['id']]

            if list_ark:
                list_to_do.append(dict(document=doc['id'], list_ark=list_ark))

        return list_to_do

    def retrieve_ocr(self) -> None:
        """
        Main function to download OCR XML from Gallica for each page in Gallica Document.
        """
        list_to_do = self._filter_documents('ocr')

        for i, dict_doc in enumerate(list_to_do):
            doc_ark = dict_doc['document']
            list_ark = dict_doc['list_ark']
            print(f"{i} --- {doc_ark}")

            # if document contains sub documents
            if list_ark[0] != doc_ark:
                return_status = False

                for j, subark in enumerate(list_ark):

                    doc = self.connector.open(f"{doc_ark}/{subark}/{subark}.xml")
                    print(f"{j} --- {subark}")

                    return_status = self._download_ocr(doc)

                    # saves file each time full ocr downloaded

                    if return_status:
                        self.connector.stat_file.update(subark, ocr=return_status)
                    else:
                        self.connector.stat_file.update(subark, ocr=return_status)

                    print(f'Updating {subark} OCR...')
                    self.connector.save(f"{doc_ark}/{subark}/{subark}.xml", doc.prettify())

                    print(f'Updating {subark} Stat File')
                    self.connector.stat_file.save(self.connector.stat_file_path)

                # saves file each time full ocr downloaded
                print(f'Updating {doc_ark} Stat File')
                self.connector.stat_file.update(doc_ark, ocr=return_status)
                self.connector.stat_file.save(self.connector.stat_file_path)

            else:
                doc = self.connector.open(f"{doc_ark}/{doc_ark}.xml")
                return_status = self._download_ocr(doc)

                # saves file each time full ocr downloaded
                if return_status:
                    self.connector.stat_file.update(doc_ark, ocr=return_status)
                else:
                    self.connector.stat_file.update(doc_ark, ocr=return_status)

                print(f'Updating {doc_ark} OCR...')
                self.connector.save(f"{doc_ark}/{doc_ark}.xml", doc.prettify())
                print(f'Updating {doc_ark} Stat File')

                self.connector.stat_file.save(self.connector.stat_file_path)

    def _download_image(self, document:BeautifulSoup, filepath:str) -> None:
        """
        Collects image corresponding to document, and saves it on disk

        :param document: Document to collect image for
        :type document: BeautifulSoup
        :param filepath: Filepath where to save img
        :type filepath: str
        """
        imgurl = document.image_url.text.strip()
        imageres = document.imageres.text.strip()
        splitindex = imgurl.find(f"/{imageres}")
        baseurl = imgurl[:splitindex]
        resurl = imgurl[splitindex:]
        print(baseurl, resurl)
        list_command = []
        num_pages = int(document.num_pages.text.strip())
        for i in range(num_pages):
            page = i + 1
            filename = f"{filepath}_page{page}.jpeg"
            url = f"{baseurl}/f{page}{resurl}"
            command = f'wget -q -O {filename} {url}'
            list_command.append(command)

        with Pool(4) as p:
            p.map(self.launch_subprocess, list_command)


    def retrieve_images(self) -> None:
        """
        Main function to collect image corresponding to a document.
        """

        list_to_do = self._filter_documents('image')

        for i, dict_doc in enumerate(list_to_do):
            doc_ark = dict_doc['document']
            list_ark = dict_doc['list_ark']
            print(f"{i} --- {doc_ark}")

            # if document contains sub documents
            if list_ark[0] != doc_ark:
                for j, subark in enumerate(list_ark):
                    docpath = f"{doc_ark}/{subark}/{subark}"
                    doc = self.connector.open(f"{docpath}.xml")
                    print(f"{j} --- {subark}")
                    filepath = f"{self.connector.folder_path}/{docpath}"
                    # print(filepath)
                    self._download_image(doc, filepath)

                    print(f'Updating {subark} Stat File')
                    self.connector.stat_file.update(subark, image=True)
                    self.connector.stat_file.save(self.connector.stat_file_path)

                print(f'Updating {doc_ark} Stat File')
                self.connector.stat_file.update(doc_ark, image=True)

            else:
                docpath = f"{doc_ark}/{doc_ark}"
                doc = self.connector.open(f"{docpath}.xml")
                filepath = f"{self.connector.folder_path}/{docpath}"

                self._download_image(doc, filepath)

                print(f'Updating {doc_ark} Stat File')
                self.connector.stat_file.update(doc_ark, image=True)
                self.connector.stat_file.save(self.connector.stat_file_path)


    def retrieve_documents(self, url:str, save_to_db:bool = True) -> List[Gallica_Document]:
        """
        Main function to retrieve document from collection on Gallica.

        :param url: Url of collection to collect documents from
        :type url: str
        :param save_to_db: Either to save the documents to a database (folder or XML), defaults to True
        :type save_to_db: bool, optional
        :return: List of collected documents 
        :rtype: List[Gallica_Document]
        """
        print('Retrieving arks from url')
        search_arks = self.get_arks(url)
        print('Filtering data ...')
        filter_search = set([file['id'] for file in self.connector.stat_file.get_collection().find_all('document')])
        print('Number doc filtered :', len(filter_search))
        print('Filtered doc :', filter_search)
        search_arks = self.connector.filter_search_arks(search_arks, filter_search)
        if save_to_db:
            list_doc = []
            for i, ark in enumerate(search_arks):
                print(f" i: {i} --- Ark: {ark}")
                doc = Gallica_Document(ark)
                list_doc.append(doc)
                doc_ark = doc.ark
                if '/' in doc_ark:
                    doc_ark = doc_ark[:doc_ark.find('/')]
                converted_doc = doc.convert_to()
                print('Saving document ...')
                # self.connector.save(f"{doc_ark}.xml", xmldoc.prettify())
                self._save_document(doc_ark, converted_doc)
                # update stat document
                print('Updating stat file')
                self.connector.stat_file.add('document', id=doc_ark)

                if doc.sub_docs:
                    # ark_subdocs = [sub.ark for sub in ]
                    for subdoc in doc.sub_docs:
                        # print(subdoc.ark)
                        self.connector.stat_file.add('subdocument', parent_tag_id=doc_ark,
                                                     id=subdoc.ark)
                    
                self.connector.stat_file.save(self.connector.stat_file_path)
            return list_doc
        else:
            return [Gallica_Document(ark) for ark in search_arks]
