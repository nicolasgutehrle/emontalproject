import re

from BaseXClient import BaseXClient
from bs4 import BeautifulSoup
import subprocess

from .connector import Connector


class XMLConnector(Connector):

    def __init__(self, host, port, user, pswrd, database_name, new_db=False):
        super(XMLConnector, self).__init__(database_name, new_db)
        self.host = host
        self.port = port
        self.user = user
        self.pswrd = pswrd


        self.update_template = "let $x := {xmldoc} return insert node $x as last into {updated_node}"
        self.replace_template = "replace node {oldnode} with {newdoc}"
        self.re_xml_ext = re.compile(r'([^\.]*)\.xml')

        self.init_session()

    def init_session(self):
        """

        :return:
        """
        try:
            print('Connecting to database ...')

            self.session = BaseXClient.Session(self.host, self.port, self.user, self.pswrd)
            print('Connected to database')

        except ConnectionRefusedError:
            print('Starting BaseXServer ...')
            self.start_basexserver()
            print('Server started')
            print('Connecting to database ...')
            self.session = BaseXClient.Session(self.host, self.port, self.user, self.pswrd)
            print('Connected to database')

    def start_basexserver(self):
        """

        :return:
        """
        # subprocess.Popen(['basexserver'])
        self.proc = subprocess.Popen('basexserver', shell=True)
        self.proc.terminate()

    def stop_basexserver(self):
        """

        :return:
        """
        self.proc.kill()

    def connect_to(self):
        if self.new_db:
            self.session.execute(f'drop db {self.database_name}')
            self.new_db = False
        self.session.execute(f'check {self.database_name}')

    def query(self, path):
        """
        """
        self.init_session()
        self.connect_to()
        results = [item for idx, item in self.session.query(path).iter()]
        self.session.close()
        return self.results_to_soup(results)

    def get_collection(self):
        """
        """
        self.init_session()
        self.connect_to()
        results = [item for idx, item in self.session.query(f"collection('{self.database_name}')").iter()]
        self.session.close()
        return self.results_to_soup(results)

    def get_data_files(self):
        """
        Return files name in database
        :param session:
        :return:
        """
        self.init_session()
        self.connect_to()
        doc_list = []
        # return filename without extension
        for doc in self.session.execute(f'list {self.database_name}').split():
            search_ext = re.search(self.re_xml_ext, doc)
            if search_ext:
                doc_list.append(search_ext.groups()[0])
        self.session.close()
        return set(doc_list)

    def results_to_soup(self, results):
        """
        """
        resultsoup = BeautifulSoup(features='xml')
        resultsoup.append(resultsoup.new_tag('results'))
        resultstag = resultsoup.find('results')
        for r in results:
            soup = BeautifulSoup(r, features='xml')
            resultstag.append(soup)
        return resultsoup

    def add(self, name, xmldoc):
        """

        :param name:
        :param doc:
        :return:
        """
        self.init_session()
        self.connect_to()
        self.session.add(f"{name}.xml", xmldoc.prettify())
        self.session.close()

    def update(self, xmldoc, updated_node):
        """

        :return:
        """
        self.init_session()
        self.connect_to()
        dict_format = {
            "xmldoc": xmldoc,
            "updated_node": updated_node
        }
        query_input = self.update_template.format(**dict_format)
        query = self.session.query(query_input)

        query.execute()

        self.session.close()

    def save(self, docname, xmldoc):
        """

        :param docname:
        :param xmldoc:
        :return:
        """
        self.init_session()
        self.connect_to()
        self.session.add(f"{docname}", xmldoc)
        self.session.close()

    def replace(self, oldnode, newdoc):
        """
        """
        self.init_session()
        self.connect_to()

        newdoc = newdoc.replace('{', '{{')
        newdoc = newdoc.replace('}', '}}')

        dict_format = {
            "oldnode": oldnode,
            "newdoc": newdoc
        }
        query_input = self.replace_template.format(**dict_format)

        query = self.session.query(query_input)

        query.execute()

        self.session.close()



# if __name__ == '__main__':
#
#     database = "fond_bourguignon"
#     xmlconnector = XMLConnector('localhost', 1984, 'admin', 'admin', database)
#
#     # url_fc = 'https://gallica.bnf.fr/services/engine/search/sru?operation=searchRetrieve&version=1.2&startRecord=0&maximumRecords=15&page=1&exactSearch=false&query=%28colnum%20adj%20%22Appartient%20%C3%A0%20l%27ensemble%20documentaire%20%3A%20FrancComt1%22%29&filter=century%20all%20%2220%22%20and%20dc.type%20all%20%22fascicule%22'
#     url_b = 'https://gallica.bnf.fr/services/engine/search/sru?operation=searchRetrieve&version=1.2&startRecord=0&maximumRecords=50&page=1&exactSearch=false&query=%28colnum%20adj%20%22Appartient%20%C3%A0%20l%27ensemble%20documentaire%20%3A%20Bourgogn1%22%29&filter=century%20all%20%2220%22%20and%20dc.type%20all%20%22fascicule%22'
#
#     search_arks = get_arks(url_b)
#     filter_set = xmlconnector.get_data_files()
#
#     search_arks = filter_search_arks(search_arks, filter_set)
#     # num_core = cpu_count() - 2
#
#     # with Pool(num_core) as p:
#     #     list_gallica_doc = p.map(ark_to_doc, search_arks)
#     # for ark in search_arks:
#
#     for i, ark in enumerate(search_arks):
#         print(f" i: {i} --- Ark: {ark}")
#         doc = Gallica_Document(ark)
#         doc_ark = doc.ark
#         if '/' in doc_ark:
#             doc_ark = doc_ark[:doc_ark.find('/')]
#         xmldoc = doc.convert_to()
#         xmlconnector.add_doc(doc_ark, xmldoc)
