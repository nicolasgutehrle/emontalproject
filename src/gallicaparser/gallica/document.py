# from .api import  Gallica_API
# from ..interface.parser import get_arks
# from parser import get_arks

from .api import Gallica_API

import re
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from bs4 import BeautifulSoup, Doctype
import copy
import xmltodict
import json
import requests


class Gallica_Document:

    """
    Class representing document retrieved from Gallica.
    Contains metadata about the document, the link to its scan,
    its OCR (if available) and its subdocuments (if any) """

    num_core = cpu_count() - 1

    def __init__(self, ark, data_type='xml', imageres='highres'):
        self.ark = ark
        self.data_type = data_type
        self.imageres = imageres
        self.oai = Gallica_API.oairecords(ark= self.ark, data_type=self.data_type)
        self.image_url = Gallica_API.simpleimage(ark=self.ark, res=self.imageres)
        self.toc = Gallica_API.toc(ark=self.ark, data_type=self.data_type)
        self.pagination = self.get_pagination()

        if self.pagination:
            self.num_pages = int(self.pagination.find('nbVueImages').text)
        else:
            self.num_pages = None

        self.ocr = self.get_ocr_link()
        # self.ocr = self.get_ocr()

        self.remove_doctypes()
        self.sub_docs = self.get_sub_documents()

    def remove_doctypes(self):
        """
        Remove any Doctype from retrieved xml / html document
        in metadata, toc, pagination and ocr
        :return:
        """
        for tree in (self.oai, self.toc, self.pagination):
            if tree:
                for item in tree.contents:
                    if isinstance(item, Doctype):

                        item.extract()

    def get_pagination(self):
        """

        :return:
        """
        if not self.is_periodical():
            return Gallica_API.pagination(ark=self.ark, data_type=self.data_type)
        else:
            return None
        
    def send_request(self, url):
        """
        Envoie la requete pour recuperer le html.
        Controle le code retourné.
        Retourne le contenu de la requete. Sinon, retourne l'erreur.
        """

        results = requests.get(url, stream=True)
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
        
    def get_ocr(self):
        print('Retrieving XML')
        # retrieves each OCR page
        
        
        ocrlinkstags = self.ocr.find_all('ocrlink')
        ocrlinks = [x.text for x in ocrlinkstags]
        print(len(ocrlinks))
        print('Downloading OCR from source')
        with Pool(self.num_core) as p:
            results = p.map(self.send_request, ocrlinks)
        print('Updating XML tree')    
        ocr_page = self.ocr.find_all('page')
        for old, content in zip(ocr_page, results):
            soup_content = BeautifulSoup(content.decode('utf-8'), features='lxml')
            old.append(soup_content.alto)        
    def get_ocr_link(self):
        """

        :return:
        """

        # if the document is not a periodical
        if not self.is_periodical():
            # if OCR is available
            if self.is_ocr():

                ocrsoup = BeautifulSoup(features='xml')
                ocrsoup.append(ocrsoup.new_tag('ocr'))
                ocrtag = ocrsoup.find('ocr')
                for i in range(self.num_pages):
                    # adding page tag
                    ocrtag.append(ocrsoup.new_tag('page'))
                    last_page_tag = ocrsoup.find_all('page')[-1]

                    # adding num and ocrlink tags to page tag
                    last_page_tag.append(ocrsoup.new_tag('num'))
                    last_page_tag.append(ocrsoup.new_tag('ocrlink'))

                    num = str(i + 1)
                    ocrlink = Gallica_API.format_url('ocr', ark=self.ark, page=num)

                    last_page_tag.num.append(num)
                    last_page_tag.ocrlink.append(ocrlink)

                return ocrsoup
            else:
                return None
        else:
            return None

    def is_ocr(self):
        """
        Checks if OCR is available for this document

        :return:
        """
        nqamoyen = float(self.oai.nqamoyen.text)
        if nqamoyen > 0:
            return True
        return False

    def is_periodical(self):
        """
        Identifies if document is either a periodical or not,
        by looking at url's end

        :param ark:
        :return:
        """
        if self.ark.endswith('/date'):
            return True
        return False

    def get_sub_documents(self):
        """
        Recursively retrieves sub-document from current document
        :return:
        """

        def get_issues_by_ark(date):
            """
            Obtient l'ark de chaque numéro pour la date donnée
            """

            date = date.text
            doc_issue_date = Gallica_API.issuesdate(self.ark, date)
            issues_ark = [issue['ark'] for issue in doc_issue_date.find_all('issue')]
            return issues_ark

        if self.is_periodical():

            # ne marche pas si metadata == json: comment faire ?
            # faire conversion en json plus tard ?
            list_date_tags = self.oai.find_all('date', {'nbIssue': re.compile(r'\d+')})

            with Pool(self.num_core) as p:
                # results = liste des arks de chaque revues contenues dans ark_revues
                # retourne une liste de liste

                results = p.map(get_issues_by_ark, list_date_tags)
                p.close()
                p.join()
                # aplatissement de results
                results = [ark for result in results for ark in result]

            # on peut etre eviter le deuxieme Pool ?
            with Pool(self.num_core) as p:
                # faire en sorte de supprimer la declaration xml
                # list_issues = p.map(Gallica_API.oairecords, results)
                list_issues = p.map(Gallica_Document, results)
                p.close()
                p.join()
            return list_issues

        else:
            return []

    def prepare_xml(self):
        """
        """
        xmlsoup = BeautifulSoup(features='xml')
        # create initial tag
        xmlsoup.append(xmlsoup.new_tag('gallica_document'))
        for attr, data in self.__dict__.items():
            # append current attribut as tag

            # xmlsoup.gallica_document.append(xmlsoup.new_tag(attr))
            # attr_tag = xmlsoup.find(attr)

            # recursively converts subdocument to given format if any
            if attr == 'sub_docs':
                if data:
                    data = [subdoc.prepare_xml() for subdoc in data]
                    xmlsoup.gallica_document.append(xmlsoup.new_tag(attr))
                    attr_tag = xmlsoup.find(attr)
                    attr_tag.extend(data)

            else:

                if data:
                    # tag must be copied to be added to new xml tree, otherwise, they are
                    # deleted from original tree
                    if isinstance(data, BeautifulSoup):
                        # change xmldoc first node name into attribute name
                        copy_data = copy.copy(data)
                        copy_data.find_all()[0].name = attr
                        xmlsoup.gallica_document.append(copy_data)
                    # convert any other type of data into string
                    else:
                        xmlsoup.gallica_document.append(xmlsoup.new_tag(attr))
                        attr_tag = xmlsoup.find(attr)
                        attr_tag.append(str(data))

        return xmlsoup

    def convert_to(self, data_format='xml'):
        """
        Converts instance of Document into either XML or JSON
        :param format:
        :return:
        """
        xmlsoup = self.prepare_xml()
        if data_format == 'json':
            xmldict = json.dumps(xmltodict.parse(xmlsoup.prettify().encode('UTF-8')))
            xmldict = json.loads(xmldict)
            xmldict = xmldict['gallica_document']
            xmldict['sub_docs'] = xmldict['sub_docs']['gallica_document']
            return xmldict
        else:
            return xmlsoup


if __name__ == "__main__":
    url_fc = 'https://gallica.bnf.fr/services/engine/search/sru?operation=searchRetrieve&version=1.2&startRecord=0&maximumRecords=15&page=1&exactSearch=false&query=%28colnum%20adj%20%22Appartient%20%C3%A0%20l%27ensemble%20documentaire%20%3A%20FrancComt1%22%29&filter=century%20all%20%2220%22%20and%20dc.type%20all%20%22fascicule%22'
    search_arks = get_arks(url_fc)
    ark = "cb43682678n/date"
    doc = Gallica_Document(search_arks[2])
    print(doc.convert_to())

