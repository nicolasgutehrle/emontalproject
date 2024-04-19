

class Connector:

    def __init__(self, database_name, new_db=False):
        self.database_name = database_name
        self.new_db = new_db

    def init_stat_file(self):
        """
        Either creates or load stat file about current database
        :return:
        """
        pass

    def update_stat_file(self):
        """
        Must add id of document already downloaded and other attributes
        :return:
        """
        pass

    def filter_search_arks(self, search_arks, filter_set):
        """

        :param search_arks:
        :param filter_list:
        :return:
        """

        ark_set = set([ark[:ark.find('/')] if '/' in ark else ark for ark in search_arks])
        left_arks = list(ark_set - filter_set)
        return [f"{ark}/date" if ark.startswith('cb') else ark for ark in left_arks]

    def init_session(self):
        """
        Must either initiatize session with database
        or connect to folder
        :return:
        """
        pass

    def query(self, query):
        """
        Must return data corresponding to given query
        :param query:
        :return:
        """
        pass

    def connect_to(self):
        """
        Must either connect to database server
        or give access to folder containing data
        :return:
        """
        pass

    def get_data_files(self):
        """
        Must return files in database
        :return:
        """
        pass

    def save(self, docname, doc):
        """
        Must save doc under docname in database or folder
        :param docname:
        :param doc:
        :return:
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