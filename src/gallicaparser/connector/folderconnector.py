from .connector import Connector
# from connector import Connector
from bs4 import BeautifulSoup
import os
import shutil
from glob import glob
from .statfile.statfile import StatFile

class FolderConnector(Connector):

    def __init__(self, database_name, new_db=False):
        super(FolderConnector, self).__init__(database_name, new_db)
        # self.session = self.init_session()
        self.folder_path = self.connect_to(database_name)
        self.stat_file_path = f"{self.folder_path}/stat_file.xml"

        self.stat_file = self.init_stat_file()

    def init_session(self):
        """

        :return:
        """
        if not os.path.exists('databases'):
            os.mkdir('databases')
        return 'databases'

    def connect_to(self, database_name):
        """

        :return:
        """
        # folder_path = f"{self.session}/{self.database_name}"
        folder_path = database_name
        # creates new folder. Removes existing one if any

        if os.path.exists(folder_path):
            if self.new_db:
                shutil.rmtree(folder_path, ignore_errors=True)
                os.mkdir(folder_path)
        else:
            os.mkdir(folder_path)
        return folder_path

    def init_stat_file(self):
        """
        Either creates or load stat file about current database
        :return:
        """
        if os.path.exists(self.stat_file_path):
            print('Loading stat_file')
            stat_file = StatFile(doc_path=self.stat_file_path)
        else:
            # data_stat = BeautifulSoup("<datastat></datastat>", 'lxml-xml')
            print('Creating stat_file ...')
            stat_file = StatFile()
            stat_file.save(self.stat_file_path)
            print('Stat_file created')
        return stat_file

    def get_data_files(self):
        """

        :return:
        """
        list_files = [file['id'] for file in self.stat_file.get_collection()]
        return set(list_files)

    def save(self, docname, doc):
        """

        :param docname:
        :param doc:
        :return:
        """
        with open(f"{self.folder_path}/{docname}", 'w', encoding='utf-8') as f:
            f.write(doc)

    def make_dir(self, dirname):
        """

        :param dirname:
        :return:
        """
        if not os.path.exists(f"{self.folder_path}/{dirname}"):
            os.mkdir(f"{self.folder_path}/{dirname}")

    def open(self, docname):
        """

        :param docname:
        :param doc:
        :return:
        """
        with open(f"{self.folder_path}/{docname}", 'r', encoding='utf-8') as f:
            return BeautifulSoup(f, 'lxml-xml')




if __name__ == "__main__":
    connector = FolderConnector('test', new_db=False)
    # connector.get_data_files()