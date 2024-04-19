import requests
from bs4 import BeautifulSoup
import xmltodict
import json
from multiprocessing import cpu_count
from time import sleep
import pkg_resources


class Gallica_API:
    # with open('apis_endpoint.json') as f:
    endpoint_path = pkg_resources.resource_filename(__name__,"apis_endpoint.json" )
    with open(endpoint_path) as f:
        dict_apis_endpoint = json.load(f)

    add_bnf_identifier = lambda ark: f"12148/{ark}"
    page_res = ('thumbnail', 'lowres', 'medres', 'highres')
    # num_core = cpu_count() - 1

    @classmethod
    def format_url(cls, endpoint, **kwargs):
        """

        :param endpoint:
        :param kwargs:
        :return:
        """

        # get url corresponding to that endpoint name
        url = cls.dict_apis_endpoint[endpoint]

        # replace each argument in url
        formatted_url = url.format(**kwargs)
        return formatted_url

    @classmethod
    def send_request(cls, url):
        """

        :param endpoint:
        :param args:
        :return:
        """

        # send get request to url. Returns the content if valid
        # print(url)
        # sleep(.5)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            return r.content
        else:
            # raise Exception(f'Request return status code {r.status_code}')
            return None

    @classmethod
    def return_content_as(cls, content, url, data_type='xml'):
        """

        :param data_type: String. Either xml, json
        :return:
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
    def issues(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
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
    def issuesdate(cls, ark, date, data_type='xml', return_url=False):
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
    def oairecords(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('oairecords', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def pagination(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
        """
        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('pagination', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def simpleimage(cls, ark, res):
        """

        :param ark:
        :param res:
        :param data_type:
        :return:
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
    def contentsearch(cls, ark, query, data_type='xml', return_url=False):
        """

        :param ark:
        :param query:
        :param data_type:
        :return:
        """
        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('contentsearch', ark=ark, query=query)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def contentpage(cls, ark, query, page,data_type='xml', return_url=False):
        """

        :param ark:
        :param query:
        :param page:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('contentpage', ark=ark, query=query, page=page)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def toc(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('toc', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def texteBrut(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('texteBrut', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def ocr(cls, ark, page, data_type='xml', return_url=False):
        """

        :param ark:
        :param page:
        :param data_type:
        :return:`
        """
        # ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('ocr', ark=ark, page=page)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def iiif(cls, ark, region, size, rotation, quality, format, data_type='xml', return_url=False):
        """

        :param ark:
        :param region:
        :param size:
        :param rotation:
        :param quality:
        :param format:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('iiif', ark=ark, region=region,
                                   size=size, rotation=rotation, quality=quality,
                                   format=format)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def iiifmetadata(cls, ark, data_type='xml', return_url=False):
        """

        :param ark:
        :param data_type:
        :return:
        """
        ark = cls.add_bnf_identifier(ark)
        formatted_url = cls.format_url('iiifmetadata', ark=ark)
        content = cls.send_request(formatted_url)
        data = cls.return_content_as(content, formatted_url, data_type)
        return data

    @classmethod
    def search(cls, *args, data_type='xml', return_url=False):
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