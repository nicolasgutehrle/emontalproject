from typing import List, Set

class Connector:

    def __init__(self, database_name: str, new_db:bool=False) -> None:
        """
        Constructor of class Connector. This is an abstract class, and should not be used.

        :param database_name: Name of the database where to save the data
        :type database_name: str
        :param new_db: Either create a new database or not, defaults to False
        :type new_db: bool, optional
        """
        self.database_name = database_name
        self.new_db = new_db

    def init_stat_file(self):
        """
        Either creates or load stat file about current database
        """
        pass

    def update_stat_file(self):
        """
        Must add id of document already downloaded and other attributes
        """
        pass

    def filter_search_arks(self, search_arks: List[str], filter_set:Set(str)) -> List[str]:
        """
        Filter list of ark ids

        :param search_arks: List of ark ids to search
        :type search_arks: List[str]
        :param filter_set: Set of ark ids to filter our the search ark ids
        :type filter_set: Set
        :return: List of ark ids
        :rtype: List[str]
        """

        ark_set = set([ark[:ark.find('/')] if '/' in ark else ark for ark in search_arks])
        left_arks = list(ark_set - filter_set)
        return [f"{ark}/date" if ark.startswith('cb') else ark for ark in left_arks]

    def init_session(self) -> None:
        """
        Must either initiatize session with database or connect to folder
        """
        pass

    def query(self, query:str) -> None:
        """
        Must return data corresponding to given query

        :param query: Query to search data in Gallica
        :type query: str
        """
        pass

    def connect_to(self):
        """
        Must either connect to database server or give access to folder containing data
        """
        pass

    def get_data_files(self):
        """
        Must return files in database
        """
        pass

    def save(self, docname:str, doc:str) -> None:
        """
        Must save doc under docname in database or folder

        :param docname: Name of document to save
        :type docname: str
        :param doc: Doc to save
        :type doc: str
        """
        pass

    def add(self, name, xmldoc):
        """
        Must add given doc to database or folder with given name
        :return:
        """
        pass

    def update(self, doc, updated_doc):
        """
        Must update given doc with updated_doc
        :param doc:
        :param updated_doc:
        :return:
        """

    def replace(self, old, new):
        """
        Must replace old data with new in data
        :param old:
        :param new:
        :return:
        """
        pass