from bs4 import BeautifulSoup

class StatFile:

    def __init__(self, doc_path:str=None, document:str=None):
        """
        Constructor

        :param doc_path: Path where to save document, defaults to None
        :type doc_path: str, optional
        :param document: Document itself, defaults to None
        :type document: str, optional
        :raises Exception: Raised if both arguments are given (only one should be given)
        """
        if doc_path and document:
            raise Exception('Either "doc_path" or "document" can be given, not both')
        if doc_path:
            with open(doc_path) as f:
                self.stat_file = BeautifulSoup(f, 'lxml-xml')
        elif document:
            self.stat_file = BeautifulSoup(document, 'lxml-xml')
        else:
            self.stat_file = BeautifulSoup("<datastat></datastat>", 'lxml-xml')

    def add(self, tagname:str, parent_tag_id:str=None, **kwargs) -> None:
        """
        Add new tag to document, under the tag with parent_tag_id

        :param tagname: Tagname to build the new tag
        :type tagname: str
        :param parent_tag_id: Id the parent tag containing the new tag, defaults to None
        :type parent_tag_id: str, optional
        """

        new_tag = self.stat_file.new_tag(tagname)
        for k, v in kwargs.items():
            new_tag[k] = v
        if parent_tag_id:
            parent_tag = self.stat_file.find(id=parent_tag_id)
            parent_tag.append(new_tag)
        else:
            self.stat_file.datastat.append(new_tag)

    def add_sub_tag(self):
        pass

    def update(self, doc_id:str, **kwargs) -> None:
        """
        Replaces content of document with given id with content provided in **kwargs

        :param doc_id: _description_
        :type doc_id: str
        """
        doc = self.stat_file.find(id=doc_id)
        for k, v in kwargs.items():
            doc[k] = v
        # self.save()

    def get_collection(self) -> BeautifulSoup:
        """
        Returns stats about collection as a BeautifulSoup document

        :return: Stats about collection
        :rtype: BeautifulSoup
        """
        return self.stat_file.datastat

    def save(self, path:str) -> None:
        """

        :return:
        """
        with open(path, 'w') as f:
            f.write(self.stat_file.prettify())