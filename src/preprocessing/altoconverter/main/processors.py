from typing import List
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Tag
import re
from textcleaner.main.ocrtextprocessor import OCRTextProcessor
from .postprocessors import Postprocessor
from .line_rules import classify_lines
from .block_rules import classify_block
from rapidfuzz import fuzz, process
from segmenter.segmenter.segmenter import Segmenter
import warnings
warnings.filterwarnings('ignore')

import unidecode

# from langdetect import DetectorFactory, detect_langs
# DetectorFactory.seed = 0

class Processor:
    """
    Processes ALTO document in order to add semantic annotations.
    These annotations are used for converting the ALTO document into another XML format
    """

    def __init__(self, lg: str):
        
        # TODO: add support for any model by Spacy and add it to constructor
#         self.nlp = nlp
        if lg not in ('fr', 'en'):
            raise Exception('Parameter lg must be one of the following: (fr, en)')
        self.seg = Segmenter(lg)
        self.ocrtextproc = OCRTextProcessor(lg)
        self.lg = lg

        self.re_capital = re.compile(r'([^a-zA-Z\d\s:]*[a-z]?)\s*[A-ZÈÉÊËÏÎÔÂÀ]')
        self.re_digit = re.compile(r'\d')
        self.re_punct = re.compile(r'[\.!?\)]$')

        self.re_month = re.compile(r'(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre)\s',
                              re.IGNORECASE)
        self.re_page = re.compile(r'([-—][^\d]*\d[^-—]*[-—]|Page\s*\d)', re.IGNORECASE)
        self.re_currency1 = re.compile(r'(centime)', re.IGNORECASE)
        self.re_currency2 = re.compile(r'(franc[s\s:\.,]|frs?[\.\s]|fr$)')

        self.re_total = re.compile(r'^tota(l|ux)', re.IGNORECASE)
        self.re_endsdigit = re.compile(r'\d$')
        self.re_address = re.compile(r'(rue|avenue|boulevard|impasse|faubourg|quai)', re.IGNORECASE)
        self.re_dash = re.compile(r'—')

        self.headerwords = ['Rubrique Locale', 'Gérant :', 'Publicité', 'Abonnement', 'Envoyez les fonds',
                   'Conservez chaque numéro',
                   'Rédacteur', 'Directeur', 'Numéro', 'Chèque postal', 'Dépôt', 'Achat — Vente — Echange', 'Annonce',
                   'Imprimerie',
                   'En vente partout', 'Paraissant']

    def round_value(self, x: int, base: int = 5) -> int:
        """
        Round value x by base
        For instance, if x = 62 and base is 5, then x becomes 60
        """
        return base * round(x/base)
    
    def capitalletters_proportion(self, text: str) -> int :
        """
        Return proportion of capital letter in text
        """
        try:
            capitals = re.findall(r'[A-Z]', text)
            proportion = len(capitals) / len(text)
            proportion = self.round_value(int(proportion * 100))
            return proportion
        except:
            pass

    def digits_proportion(self, text: str) -> int :
        """
        Return proportion of digits letter in text
        """
        try:
            capitals = re.findall(r'\d', text)
            proportion = len(capitals) / len(text)
            proportion = self.round_value(int(proportion * 100))
            return proportion
        except:
            pass

    def nonalpha_proportion(self, text: str) -> int:
        """
        Return proportion of non-alphanumeric characters in text
        """
        try:
            non_alpha = re.findall(r'[^a-zA-Z\d\s:]', text)
            proportion = len(non_alpha) / len(text)
            proportion = self.round_value(int(proportion * 100))
            return proportion
        except:
            pass

    def get_morphological_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts morphological data about each line
        these morphological data are:
        does the line starts with a capital letter, with a digit,
        ends with a punctuation. Also finds the proportion of capital letters and
        non-alphanumeric character in the line

        """

        # non alphanumeric characters, can be preceded and / or followed by one
        # alphanumeric character  than maybe spaces before Capital letter
        df['stw_capital'] = 0
        df.loc[df['text_line'].str.match(self.re_capital) == True, 'stw_capital'] = 1

        df['stw_digit'] = 0
        df.loc[df['text_line'].str.match(self.re_digit) == True, 'stw_digit'] = 1

        df['ends_digits'] = 0
        df.loc[df['text_line'].str.contains(self.re_endsdigit) == True, 'ends_digits'] = 1

        df['ends_punct'] = 0
        df.loc[df['text_line'].str.contains(self.re_punct) == True, 'ends_punct'] = 1

        df['capital_prop'] = [self.capitalletters_proportion(text) for text in df['text_line'].values]
        df['non_alpha_prop'] = [self.nonalpha_proportion(text) for text in df['text_line'].values]
        df['digits_prop'] = [self.digits_proportion(text) for text in df['text_line'].values]

        return df

    def get_semantic_data(self, title: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts semantic features from ALTO document such as presence of a keyword, similarity with a title, ...
        :param title: Title of the document, used to determine the similarity of a line with the title
        :type title: str
        :param df: TextLine or TextBlock Feature DataFrame
        :type df: pd.DataFrame
        :return: Updated TextLine or TextBlock Feature DataFrame
        :rtype: pd.DataFrame
        """
        # calculates the similarity between each line with the title of the document
        simtitle = [fuzz.token_set_ratio(title, line) for line in df['text_line']]
        df['simtitle'] = simtitle

        # calculates similarity of any word in the Header word set with each TextLine
        df['simheader'] = [process.extractOne(line, self.headerwords)[1] for line in df['text_line']]

        df['ctn_page'] = 0
        df.loc[df['text_line'].str.contains(self.re_page) == True, 'ctn_page'] = 1

        df['ctn_currency'] = 0
        df.loc[df['text_line'].str.contains(self.re_currency1) == True, 'ctn_currency'] = 1
        df.loc[df['text_line'].str.contains(self.re_currency2) == True, 'ctn_currency'] = 1

        df['ctn_month'] = 0
        df.loc[df['text_line'].str.contains(self.re_month) == True, 'ctn_month'] = 1

        df['ctn_total'] = 0
        df.loc[df['text_line'].str.contains(self.re_total) == True, 'ctn_total'] = 1

        df['ctn_address'] = 0
        df.loc[df['text_line'].str.contains(self.re_address) == True, 'ctn_address'] = 1

        df['ctn_dash'] = 0
        df.loc[df['text_line'].str.contains(self.re_dash) == True, 'ctn_dash'] = 1

        return df

    def postprocess_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess page_metadata file by identifying block types
        """

        # extracts block metadata
        df_block = self.get_block_data(df)
        df_block = df_block.reset_index()

        # updates block types with given set of rules
        df_block = classify_block(df_block, df)

        for block_id, block_type, block_rule, block_post_rule in zip(df_block['block_id'], df_block['block_type'].values, df_block['block_rule'].values, df_block['block_post_rule'].values):
            df.loc[df['block_id'] == block_id, 'block_type'] = block_type
            df.loc[df['block_id'] == block_id, 'block_rule'] = block_rule
            df.loc[df['block_id'] == block_id, 'block_post_rule'] = block_post_rule
        return df, df_block

    def extract_line_data(self, file) -> pd.DataFrame:
        """

        """
        with open(file, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml-xml')

        list_pages = soup.ocr.find_all('page', recursive=False)
        # merges every alto page into one document
        ocr_doc = BeautifulSoup('<ocr_doc></ocr_doc>', features='xml')
        ocr_tag = ocr_doc.find('ocr_doc')

        for i, ocr_page in enumerate(list_pages):

            alto = ocr_page.alto
            layout = alto.layout
            description = alto.description
            styles = alto.styles

            alto_page = layout.printspace
            if alto_page:
                ocr_tag.append(alto_page)

        # retrieves metadata for each tag textblock in the document
        list_textblocks = ocr_doc.find_all('textblock')
        list_block_data = []

        if list_textblocks:
            # collectcs data about lines in each textblock, plus some data about the block itself
            for num_block, textblock in enumerate(list_textblocks):

                block_data = self.get_line_data(textblock)
                if block_data:

                    list_block_data.append(block_data)

        # groups DataFrame for each textblock into one DF
        df = pd.concat([pd.DataFrame.from_dict(d) for d in list_block_data]).reset_index(drop=True)
        df = df[['page', 'block_id', 'block_type', 'line_id', 'text_line']]
        return df

    def get_page_metadata(self, alto_page: BeautifulSoup) -> pd.DataFrame:
        """

        """
        doctitle = alto_page.find('title').contents[0]
        # calculates linespace between each line for the whole document
        # then calculates the linespace mean
        all_lines = alto_page.find_all('textline')
        all_lines = [int(x['vpos']) for x in all_lines]
        dic_linespace = self.get_interline_space(all_lines, all_lines[1:])

        # retrieves metadata for each tag textblock in the document
        list_textblocks = alto_page.find_all('textblock')
        list_block_data = []

        ## TODO: MAYBE JUST LOOP OVER THE FILE TO GET ATTRIBUTES FROM XML
        ## TODO: THAN USE GROUPBY TO COLLECT DATA AT BLOCK LEVEL (MAYBE AT RULE TIMES ?)
        if list_textblocks:
            # collectcs data about lines in each textblock, plus some data about the block itself
            for num_block, textblock in enumerate(list_textblocks):

                block_data = self.get_geometric_data(textblock)
                if block_data:

                    list_block_data.append(block_data)

            # groups DataFrame for each textblock into one DF
            df = pd.concat([pd.DataFrame.from_dict(d) for d in list_block_data]).reset_index(drop=True)

            # adds dic_linespace data to DF TODO: CORRIGER CA
            for k, v in dic_linespace.items():
                df[k] = v

        else:
            df = pd.DataFrame()

        df['diff_hpos'] = df['line_hpos'] - df['hpos_median']

        # # retrieves morphological data about lines and blocks
        df = self.get_morphological_data(df)

        df = self.get_semantic_data(doctitle, df)

        doc_median = df.median()
        doc_median = doc_median[[index for index in doc_median.index if index.startswith('line_') or index in ('word_count', 'capital_prop',)]]
        df['doc_std_hpos'] = df['diff_hpos'].std()

        doc_median = doc_median.drop('line_following_space', axis  = 0)
        doc_median.index = ['doc_height', 'doc_width', 'doc_vpos', 'doc_hpos', 'doc_word', 'doc_space', 'doc_capital' ]
        for index, value in zip(doc_median.index, doc_median.values):
            df[index] = value

        # df['correct_hpos'] = self.correct_hpos(df)

        return df

    def get_interline_space(self, list_lines_1: List[int], list_lines_2: List[int]) -> dict:
        """
        Calculates space between each line by substracting VPOS values
        then calculates linespace mean
        """
        list_following_space = []
        # finding space between line and next one
        for main_line, next_line in zip(list_lines_1, list_lines_2):

            # calculates if next line if below main line
            # else main line is the last line of its block
            if int(next_line) > int(main_line):
                diff_linespace = self.round_value(int(next_line) - int(main_line))
                list_following_space.append(diff_linespace)
            else:
                # last line of a textblock.
                list_following_space.append(0)

        list_following_space.append(0)

        list_previous_space = [0] + list_following_space[:-1]


        dic_linespace = {
            "line_previous_space": list_previous_space,
            "line_following_space": list_following_space,
            "linespace_mean": self.round_value(int(np.mean(list_following_space))),
            "linespace_median": self.round_value(int(np.median(list_following_space)))


        }

        return dic_linespace

    def get_block_type(self, textblock: Tag) -> str:
        """
        Retrieves value of type attribute from given textblock.
        Is only concerned with 'table', 'advertisement', 'titre1' types
        If it has no type attribute or type is not in the list above,
        then its type is set to 'text'
        """
        
        if textblock.has_attr('type'):
            blocktype = textblock['type']
            if blocktype in ('table', 'advertisement', 'titre1'):
                return blocktype
            else:
                return 'No_type'
                # return 'text'
        else:
            textblock_parent = textblock.parent
            if textblock_parent.has_attr('type'):
                parentblocktype = textblock_parent['type']
                if parentblocktype == 'table':
                    return parentblocktype
                else:
                    return 'No_type'
                    # return 'text'
            else:
                return 'No_type'
                # return 'text'

    def get_block_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gathers metadata about each textblock in the document and metadata about
        the document they are in
        """

        # data about each block have already been collected before while
        # collecting data about lines. We can groupby block id to get about
        # data about the blocks
        block_id = data.groupby('block_id')
        block_page = block_id.first()['page']
        block_type = block_id.first()['block_type']
        first_hpos = block_id.first()['line_hpos']  # vpos / hpos of the first textline in the block
        first_vpos = block_id.first()['line_vpos']
        last_hpos = block_id.last()['line_hpos']  # vpos / hpos of the last textline in the block
        last_vpos = block_id.last()['line_vpos']
        word_count = block_id.sum()['word_count']  # number of word in block
        height = block_id.median()['line_height']  # median height of the block
        width = block_id.median()['line_width']  # median width of the blcok
        doc_width = block_id.first()['doc_width']
        doc_words = block_id.first()['doc_word']
        doc_height = block_id.first()['doc_height']
        ctn_total_word = block_id.sum()['ctn_total']
        capital_prop = block_id.median()['capital_prop']
        digits_prop = block_id.median()['digits_prop']
        block_counts = data['block_id'].value_counts()  # number of lines in the block

        # transforming data into DataFrame
        data_block = {
            'block_line': block_id.first()['text_line'],
            "block_page": block_page,
            "block_type": block_type,
            "block_first_hpos": first_hpos,
            "block_first_vpos": first_vpos,
            "block_last_hpos": last_hpos,
            "block_last_vpos": last_vpos,
            "block_word_count": word_count,
            "block_height": height,
            "block_width": width,
            "block_linecounts": block_counts.sort_index(ascending=True).values,
            "digits_prop": digits_prop,
            "capital_prop": capital_prop,
            "doc_width": doc_width,
            "doc_words": doc_words,
            "doc_height": doc_height,
            "ctn_total_word": ctn_total_word

        }
        df_block = pd.DataFrame.from_dict(data_block)

        # calculates for the whole document the space between each block
        main_line = df_block['block_last_vpos'].values
        next_line = df_block.iloc[1:]['block_first_vpos'].values

        interline = self.get_interline_space(main_line, next_line)
        df_block['block_previous_space'] = interline['line_previous_space']
        df_block['block_following_space'] = interline['line_following_space']

        # median height for the whole document
        df_block['block_height_med'] = df_block['block_height'].median()

        # word ratio by block for the whole document
        df_block['block_word_ratio'] = round(df_block['block_word_count'].sum() / len(df_block))

        # median interblock space for the whole document
        df_block['block_interline_space_med'] = df_block['block_previous_space'].median()

        # median number of line by block for the whole document
        df_block['block_count_med'] = df_block['block_linecounts'].median()

        return df_block

    def get_geometric_data(self, textblock: Tag) -> dict:
        """
        Retrieves metadata about textline tag and metadata about the textblock they are in
        Mainly extracts data that is in the textline attributes, but also calculates other data
        from data (number of word, mean and median in the whole block, ...)
        """
        # extracts each line in the block
        list_lines = textblock.find_all('textline')

        # extract the id and the type of the block if it is available
        block_type = self.get_block_type(textblock)
        block_id = textblock['id']

        # the block is ignore if it doesnt have lines
        if list_lines:
            line_id = [line['id'] for line in list_lines]
            page = [int(re.search(r'PAG_0*(\d[^_]*)', line).group(1)) for line in line_id]
            words = [line.find_all('string') for line in list_lines]
            # text_line = [" ".join([string['content'] for string in word]) for word in words]
            text_line = self.convert_line_to_text(list_lines)

            if len(list_lines) > 1:
                # if more than 1 line in textblock
                line_height = [self.round_value(int(line['height'])) for line in list_lines]
                line_width = [self.round_value(int(line['width'])) for line in list_lines]
                line_hpos = [self.round_value(int(line['hpos'])) for line in list_lines]
                line_vpos = [self.round_value(int(line['vpos'])) for line in list_lines]
                word_count = [len(word) for word in words]

            else:
                line = list_lines[0]
                line_height = [self.round_value(int(line['height']))]
                line_width = [self.round_value(int(line['width']))]
                line_hpos = [self.round_value(int(line['hpos']))]
                line_vpos = [self.round_value(int(line['vpos']))]
                word_count = [len(words[0])]

            # collect mean and median data at the block level for
            # the height, width, word count, hpos and vpos
            height_mean = self.round_value(int(np.mean(line_height)))
            width_mean = self.round_value(int(np.mean(line_width)))
            count_mean = self.round_value(int(np.mean(word_count)))
            hpos_mean = self.round_value(int(np.mean(line_hpos)))
            vpos_mean = self.round_value(int(np.mean(line_vpos)))
            # wordspace_mean = self.round_value(int(np.mean(space_sum)))

            height_median = self.round_value(int(np.median(line_height)))
            width_median = self.round_value(int(np.median(line_width)))
            count_median = self.round_value(int(np.median(word_count)))
            hpos_median = self.round_value(int(np.median(line_hpos)))
            vpos_median = self.round_value(int(np.median(line_vpos)))
            # wordspace_median = self.round_value(int(np.median(space_sum)))

            data = {
                "text_line": text_line,
                'page': page,
                "line_id": line_id,
                "block_id": block_id,
                "block_type": block_type,
                "line_height": line_height,
                "height_mean": height_mean,
                "height_median": height_median,
                "line_width": line_width,
                "width_mean": width_mean,
                "width_median": width_median,
                "line_hpos": line_hpos,
                "hpos_mean": hpos_mean,
                "hpos_median": hpos_median,
                "line_vpos": line_vpos,
                "vpos_mean": vpos_mean,
                "vpos_median": vpos_median,
                "word_count": word_count,
                "count_mean": count_mean,
                "count_median": count_median,
                # "wordspace_med": space_med,
                # "wordspace_mean": wordspace_mean,
                # "wordspace_median": wordspace_median
            }

            return data

    def convert_line_to_text(self, list_lines: List[Tag]) -> List[str]:
        """
        Convert each textline tags in list_line into into string
        """
        list_txtline = []
        for line in list_lines:
            # check each subtag in textline tag and adds it to list
            text_line = []
            for chartag in line.find_all():

                if chartag.has_attr('subs_content'):
                    text_line.append(chartag['subs_content'])

                elif chartag.name == 'string':
                    text_line.append(chartag['content'])

                elif chartag.name == 'sp':
                    text_line.append(" ")
                else:
                    text_line.append(chartag['content'])

                    # print(line)
                    # raise Exception('Tag unknown : ', chartag)
            text_line = "".join(text_line)
            list_txtline.append(text_line)
        return list_txtline
    
    def tag_sentences(self, doc_book: BeautifulSoup) -> None:
        """
        Tags sentence from text of every paragraph
        """

        list_para_tags = doc_book.find_all('para')
        list_text = [self.ocrtextproc.clean_text(para.text) for para in list_para_tags]
        list_doc = [self.seg.getSentences(text) for text in list_text]
#         list_doc = self.nlp.pipe(list_text)

        for para, text, doc in zip(list_para_tags, list_text, list_doc):
            para.clear()
            for sent in doc:

                sent = self.ocrtextproc.remove_hyphen(sent)
                para.append(doc_book.new_tag('sent'))
                sent_tag = para.find_all('sent')[-1]
                sent_tag.append(sent)

    def convert_to_docbook(self, filename:str, xml: BeautifulSoup, page_metadata: pd.DataFrame) -> BeautifulSoup:
        """
        Parse each line contained in page_metadata. Creates a XML doc.
        For each line, add tag according to line type:
        * title tag
        * subtitle tag
        * table tag
        * para tag (paragraph)
        * add text to last para tag if line has no type

        Call self.tag_sentences to tag sentences in para tag after the whole document is converted to XML
        """

        def add_line(tagname: str, last_tag_added: str, attr:dict, appendedtag: Tag) -> None:
            """
            Add tag with tagname to the last article tag create
            """
            if last_tag_added != tagname:
                appendedtag.append(doc_book.new_tag(tagname, attrs=attr))
            lasttag = doc_book.find_all(tagname)[-1]
            lasttag.append(text_line)

        def add_id(tagname: str) -> None:
            """
            Runs through all tag for given tagname and adds id
            id is formed as tagname_i
            """
            for i, tag in enumerate(content_tag.find_all(tagname)):
                tag['id'] = f"{tagname}_{i + 1}"

        def add_readability(tagname: str) -> None:
            """
            Runs OCRTextprocessor method to calculate text readability
            on every tagname in doc
            """
            for tag in content_tag.find_all(tagname):
                text = tag.text.strip()
                read_score, read_mark = self.ocrtextproc.text_readability(text)
                tag['readability_score'] = read_score
                tag['readability'] = read_mark

        def process_metadata(xml: BeautifulSoup) -> Tag:
            """
            Extract given set of tags in xml and adds them to a new
            metadata tag
            """
            metadata_tag = doc_book.new_tag('metadata')

            for tagname in ['ark', 'dc:identifier', 'dc:date', 'dc:title', 'dc:contributor',
                            'dc:publisher', 'dc:languague', 'dc:creator','dc:source', 'typedoc',
                            'nqamoyen', 'dewey', 'image_url']:

                # tag = xml.new_tag(tagname)

                try:
                    original_tag = xml.find(tagname)
                    tagname = tagname.replace('dc:', '')
                    tag = xml.new_tag(tagname)
                    tag.append(original_tag.text)
                except:
                    tagname = tagname.replace('dc:', '')
                    tag = xml.new_tag(tagname)

                # print(tag.name)
                metadata_tag.append(tag)

            return metadata_tag


#       creates initial empty docbook document with document tag
#         doc_book = BeautifulSoup(f'<document id="{filename}"></document>', features='xml')
        doc_book = BeautifulSoup(f'<document></document>', features='lxml-xml')
        document_tag = doc_book.find_all('document')[-1]

        # extracts given tags in metadata of original file
        metadata_tag = process_metadata(xml)
        document_tag.append(metadata_tag)

        content_tag = doc_book.new_tag('content')
        document_tag.append(content_tag)

        # precreates page tags with their header
        for pagenumber in page_metadata['page'].unique():
            pagenumber = int(pagenumber)
            pagetag = doc_book.new_tag('page', attrs={"id" : f"{pagenumber}"})

            headertag = doc_book.new_tag('header', attrs={"id" : f"header_{pagenumber}"})

            articlestag = doc_book.new_tag('articles')

            pagetag.append(headertag)
            pagetag.append(articlestag)

            content_tag.append(pagetag)

        dic_articletags = {}
        df_articles = page_metadata[
            page_metadata['article_id'] != 'header'

        ]
        # precreates article tags for each page with their title and text.
        # Each tag is stored in dic_articletags
        # in order to access them more easily
        setarticles = set(df_articles[['page', 'article_id']].to_records(index=False).tolist())
        setarticles = sorted(setarticles)
        for articles in setarticles:
            pagenumber = int(articles[0])
            article_id = articles[1]
            article_number = article_id[article_id.find('_') + 1:]
            pagetag = content_tag.find('page', {'id': f'{pagenumber}'})
            pagearticles = pagetag.findChildren('articles', recursive=False)[0]
            articletag = doc_book.new_tag('article', attrs={'id': article_id})

            articletag.append(doc_book.new_tag('title', attrs={'id': f"title_{article_number}"}))
            articletag.append(doc_book.new_tag('text', attrs={'id': f"text_{article_number}"}))


            pagearticles.append(articletag)

            dic_articletags[f"{pagenumber}_{article_id}"] = articletag

        last_tag_added = 'text'
        zip_columns = zip(page_metadata['text_line'],
                          page_metadata['block_id'],
                          page_metadata['page'],
                          page_metadata['line_class'],
                          page_metadata['article_id']
                          )
        for text_line, block_id, page, line_class, article_id in zip_columns:

            text_line = f"{text_line}\n"
            attr = {
                "block_id": block_id,
                # "page": page
            }
            pagenumber = int(page)

            # TODO: change firstline name into para in rules
            if line_class == 'firstline':
                line_class = 'para'

            if line_class == 'header':
                headertag = content_tag.find('header', {'id': f"header_{pagenumber}"})
                headertag['block_id'] = block_id
                headertag.append(text_line)
            else:
                article_tag = dic_articletags[f"{pagenumber}_{article_id}"]
                article_number = article_id[article_id.find('_') + 1:]
                text_tag = article_tag.find('text')

                # some article might be running across multiple pages,
                # thus, their mightbe multiple article_XX and title_XX tags
                # so we need to add the title to all these tags
                if line_class == 'title':
                    l_titles = doc_book.find_all('title', {'id': f'title_{article_number}'})
                    for t in l_titles:
                        t.append(text_line)

                elif line_class != 'text':
                    add_line(line_class, last_tag_added, attr, text_tag)

                else:
                    list_para_tags = doc_book.find_all('para')
                    last_para_tag = list_para_tags[-1]
                    last_para_tag.append(text_line)

                last_tag_added = line_class

        # extracts sentences from para tags, than adds id to sent tag
        self.tag_sentences(doc_book)

        # add readability to given tags
        add_readability('sent')
        add_readability('title')
        add_readability('header')
        add_readability('other')
        add_readability('para')
        add_readability('text')
        add_readability('article')

        # adds id to given tags
        list_tag = ['para', 'sent', 'other']
        for tagname in list_tag:
            add_id(tagname)
        return doc_book

    def article_segmentation(self, df_line: pd.DataFrame) -> pd.DataFrame:
        """
        Finds groups of titles and adds an article_id column where each
        row is assigned an article id

        :param df_line: TextLine Feature DataFrame
        :type df_line: pd.DataFrame
        :return: Updated TextLine Feature DataFrame
        :rtype: pd.DataFrame
        """

        df_line['article_id'] = np.nan
        # gets every line labelled as title
        df_line.loc[(df_line['line_class'] == 'title')
                          & (df_line['prev_title'] == 0), 'article_id'] = 'article'

        # sets a differt article id to each title line
        article_id = df_line[df_line['article_id'].notna()]['article_id']
        for i, v in enumerate(article_id.index):
            if i < 9:
                df_line.loc[v, 'article_id'] = f"article_0{i + 1}"
            else:
                df_line.loc[v, 'article_id'] = f"article_{i + 1}"

        # propagates article id to each line until reaching a different article id
        df_line['article_id'].fillna(method='ffill', inplace=True)

        # removes article_id from header row
        df_line.loc[
            df_line['line_class'] == 'header', 'article_id'
        ] = 'header'

        return df_line


    # def article_segmentation(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Proceeds to article segmentation by finding groups of titles
    #     Adds an article_id column where each row is assigned an article id
    #     """

    #     page_metadata['article_id'] = np.nan
    #     # gets every line labelled as title
    #     page_metadata.loc[(page_metadata['line_class'] == 'title')
    #                       & (page_metadata['prev_title'] == 0), 'article_id'] = 'article'

    #     # sets a differt article id to each title line
    #     article_id = page_metadata[page_metadata['article_id'].notna()]['article_id']
    #     for i, v in enumerate(article_id.index):
    #         if i < 9:
    #             page_metadata.loc[v, 'article_id'] = f"article_0{i + 1}"
    #         else:
    #             page_metadata.loc[v, 'article_id'] = f"article_{i + 1}"

    #     # propagates article id to each line until reaching a different article id
    #     page_metadata['article_id'].fillna(method='ffill', inplace=True)

    #     # removes article_id from header row
    #     page_metadata.loc[
    #         page_metadata['line_class'] == 'header', 'article_id'
    #     ] = 'header'

    #     return page_metadata

    def process(self, ocr_doc: BeautifulSoup):
        """
        Extracts metadata about TextBlock and TextLine tags
        then applies rules to extract logical structure
        """
        page_metadata = self.get_page_metadata(ocr_doc)
        # page_metadata = self.postprocess_metadata(page_metadata)
        page_metadata, df_block = self.postprocess_metadata(page_metadata)

        page_metadata = classify_lines(page_metadata)

        # article segmentation
        page_metadata = self.article_segmentation(page_metadata)

        return page_metadata, df_block

    def convert_doc(self, file: str, return_metadata: bool = True) -> dict:
        """
        Main interface to convert ALTO XML file into another format
        """
        with open(file, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml-xml')
        metadata_tag = soup.find('metadata')
        doctitle = metadata_tag.find('title').contents[0].strip()

        list_pages = soup.ocr.find_all('page', recursive=False)

        # merges every alto page into one document
        ocr_doc = BeautifulSoup('<ocr_doc><title></title></ocr_doc>', features='xml')
        title_tag = ocr_doc.find('title')
        title_tag.append(doctitle)

        ocr_tag = ocr_doc.find('ocr_doc')

        for i, ocr_page in enumerate(list_pages):

            alto = ocr_page.alto
            layout = alto.layout
            description = alto.description
            styles = alto.styles

            alto_page = layout.printspace
            if alto_page:
                ocr_tag.append(alto_page)

        page_metadata, df_block = self.process(ocr_doc)
        page_metadata = page_metadata.dropna()
        docbook = self.convert_to_docbook(file, soup, page_metadata)

        return {
            "line_metadata": page_metadata,
            "block_metadata": df_block,
            "docbook": docbook
        }


if __name__ == '__main__':
    proc = Processor()

