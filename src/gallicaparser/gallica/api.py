import requests
from bs4 import BeautifulSoup
import xmltodict
import json
from multiprocessing import cpu_count
from time import sleep
import pkg_resources


class Gallica_API:

    # loads endpoint from disk
    endpoint_path = pkg_resources.resource_filename(__name__,"apis_endpoint.json" )
    with open(endpoint_path) as f:
        dict_apis_endpoint = json.load(f)

    add_bnf_identifier = lambda ark: f"12148/{ark}"
    page_res = ('thumbnail', 'lowres', 'medres', 'highres')
    # num_core = cpu_count() - 1

    @classmethod
    def format_url(cls, endpoint:str, **kwargs:str)-> str:
        """
        Formats endpoint with given values

        :param endpoint: Endpoint of the API
        :type endpoint: str
        :return: Formatted endpoint
        :rtype: str
        """

        # get url corresponding to that endpoint name
        url = cls.dict_apis_endpoint[endpoint]

        # replace each argument in url
        formatted_url = url.format(**kwargs)
        return formatted_url

    @classmethod
    def send_request(cls, url:str) -> str:
        """
        Send request to given URL in Gallica

        :param url: url to request
        :type url: str
        :return: Content of the response
        :rtype: str
        """

        # send get request to url. Returns the content if valid
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            return r.content
        else:
            # raise Exception(f'Request return status code {r.status_code}')
            return None

    @classmethod
    def return_content_as(cls, content:str, url:str, data_type='xml'):
        """
        Return a content as either XML or JSON

        :param content: Content to format
        :type content: str
        :param url: Url where the content comes from
        :type url: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :raises Exception: Raised if data_type is neither 'xml' or 'json'
        :return: Either BeautifulSoup or dictionary
        :rtype: _type_
        """
        if content:
            # parse request content
            soup = BeautifulSoup(content, "lxml-xml")

            # add request url to xml data first data
            first_tag = soup.find()
            first_tag.append(soup.new_tag('request_url'))
            soup.request_url.append(url)

            if data_type == 'xml':
                return soup
            elif data_type == 'json':
                return xmltodict.parse(soup.prettify().encode('UTF-8'))
            else:
                raise Exception(f"Please select one of the following data_type to return ['xml', 'json']")
        else:
            return content

    @classmethod
    def issues(cls, ark:str, data_type='xml'):
        """
        Collect an issue from Gallica by its ark id

        :param ark: Ark id of the issue to collect
        :type ark: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :raises Exception: Raised if document is not a periodical (does not end with /date)
        :return: Either BeautifulSoup or dictionary
        :rtype: _type_
        """
        if ark.endswith('/date'):
            ark = cls.add_bnf_identifier(ark)
            formatted_url = cls.format_url('issues', ark=ark)
            content = cls.send_request(formatted_url)
            data = cls.return_content_as(content, formatted_url, data_type)
            return data
        else:
            raise Exception('Document must be a periodical (ark must end with /date)')

    @classmethod
    def issuesdate(cls, ark, date, data_type='xml'):
        """

        :param ark:
        :param date:
        :param data_type:
        :return:
        """
        if ark.endswith('/date'):
            ark = cls.add_bnf_identifier(ark)
            formatted_url = cls.format_url('issuesdate', ark=ark, date=date)
            content = cls.send_request(formatted_url)
            data = cls.return_content_as(content, formatted_url, data_type)
            return data
        else:
            raise Exception('ark must end with /date')

    @classmethod
    def oairecords(cls, ark:str, data_type='xml'):
        """
        Collects metadata of a document from Gallica

        :param ark: Ark id of the issue to collect
        :type ark: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :raises Exception: Raised if document is not a periodical (does not end with /date)
        :return: Either BeautifulSoup or dictionary
        :rtype: _type_
        """

        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('oairecords', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def pagination(cls, ark, data_type='xml'):
        """
        Gets document's pagination

        :param ark: Ark id of the issue to collect
        :type ark: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :raises Exception: Raised if document is not a periodical (does not end with /date)
        :return: Either BeautifulSoup or dictionary
        :rtype: _type_
        """
        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('pagination', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def simpleimage(cls, ark:str, res:str) -> str:
        """
        Gets URL of scan of documents with the given ark id, in the resolution specified by res argument

        :param ark: Ark id of the document
        :type ark: str
        :param res: Resolution to get the document
        :type res: str
        :raises Exception: Raised if correct value for the resolution is not given
        :return: Url of the scanned document
        :rtype: str
        """
        if res in cls.page_res:
            if not ark.endswith('/date'):
                ark = cls.add_bnf_identifier(ark)
                formatted_url = cls.format_url('simpleimage', ark=ark, res=res)
                # content = cls.send_request(formatted_url)
                # data = cls.return_content_as(content, formatted_url, data_type)
                return formatted_url
            else:
                return None
                # raise Exception('Document must not be a periodical')
        else:
            raise Exception(f'res argument must be one of the following value : {cls.page_res}')

    @classmethod
    def contentsearch(cls, ark:str, query:str, data_type='xml'):
        """
        Returns list of occurrences within a document with given ark id

        :param ark: Ark id
        :type ark: str
        :param query: Term to search for
        :type query: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :return: Occurrences of the terms in the document
        :rtype: _type_
        """

        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('contentsearch', ark=ark, query=query)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def contentpage(cls, ark:str, query:str, page:int,data_type='xml'):
        """
        Returns list of occurrences within a document with given ark id in given page

        :param ark: Ark id
        :type ark: str
        :param query: Term to search for
        :type query: str
        :param page: Page where to search for
        :type page: int
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :return: Occurrences of the terms in the document
        :rtype: _type_
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('contentpage', ark=ark, query=query, page=page)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def toc(cls, ark:str, data_type='xml'):
        """
        Gets the table of content of the document

        :param ark: Ark id
        :type ark: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :return: Table of content of the document
        :rtype: _type_
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('toc', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def texteBrut(cls, ark:str, data_type='xml'):
        """
        Gets textual content of document with given ark id, without XML formatting

        :param ark: Ark id
        :type ark: str
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :return: Textual content of document
        :rtype: _type_
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('texteBrut', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def ocr(cls, ark:str, page:int, data_type='xml'):
        """
        Returns textual content of document in XML ALTO format for given page

        :param ark: Ark id
        :type ark: str
        :param page: Page to collect the transcription
        :type page: int
        :param data_type: Format in which to return the data, defaults to 'xml'
        :type data_type: str, optional
        :return: Textual content of document
        :rtype: _type_
        """
        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('ocr', ark=ark, page=page)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def iiif(cls, ark:str, region:str, size:str, rotation:str, quality:str, format:str, data_type='xml'):
        """
        TODO : Collects scan image of document

        :param ark: _description_
        :type ark: str
        :param region: _description_
        :type region: str
        :param size: _description_
        :type size: str
        :param rotation: _description_
        :type rotation: str
        :param quality: _description_
        :type quality: str
        :param format: _description_
        :type format: str
        :param data_type: _description_, defaults to 'xml'
        :type data_type: str, optional
        :return: _description_
        :rtype: _type_
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('iiif', ark=ark, region=region,
                                   size=size, rotation=rotation, quality=quality,
                                   format=format)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def iiifmetadata(cls, ark:str, data_type='xml'):
        """
        TODO : collects image metadata

        :param ark: _description_
        :type ark: str
        :param data_type: _description_, defaults to 'xml'
        :type data_type: str, optional
        :return: _description_
        :rtype: _type_
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('iiifmetadata', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def search(cls, *args, data_type='xml'):
        """

        :param args:
        :param data_type:
        :return:
        """

        query = ", ".join([f"{item}" for item in args])
        formatted_url = cls.format_url('search', query=query)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

if __name__ == "__main__":
    api = Gallica_API(

    )