from bs4 import BeautifulSoup

class StatFile:

    def __init__(self, doc_path=None, document=None):
        if doc_path and document:
            raise Exception('Either "doc_path" or "document" can be given, not both')
        if doc_path:
            with open(doc_path) as f:
                self.stat_file = BeautifulSoup(f, 'lxml-xml')
        elif document:
            self.stat_file = BeautifulSoup(document, 'lxml-xml')
        else:
            self.stat_file = BeautifulSoup("<datastat></datastat>", 'lxml-xml')

    def add(self, tagname, parent_tag_id=None, **kwargs):
        """

        :return:
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
        """

        :return:
        """
        pass

    def update(self, doc_id, **kwargs):
        """

        :return:
        """
        doc = self.stat_file.find(id=doc_id)
        for k, v in kwargs.items():
            doc[k] = v
        # self.save()

    def get_collection(self):
        """

        :return:
        """
        return self.stat_file.datastat

    def save(self, path):
        """

        :return:
        """
        with open(path, 'w') as f:
            f.write(self.stat_file.prettify())