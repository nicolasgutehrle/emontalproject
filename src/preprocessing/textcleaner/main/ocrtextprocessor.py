from typing import List
import re
from bs4 import BeautifulSoup, Tag, SoupStrainer
import pandas as pd
from symspellpy import SymSpell, Verbosity
# import pkgutil
import pkg_resources
from glob import glob
import os
import unidecode
import cchardet
from itertools import starmap
import json
from nltk.util import ngrams
import pyphen

class OCRTextProcessor:
    """
    Class containing multiple method to process and clean text in ALTO document

    """

    def __init__(self, lg):

        self.lg = lg

        # find acronyms that are stuck with sentence stop
        self.re_acronym_sent = re.compile(r'([A-Z]\.)([A-Z])')
        self.re_hyphen = re.compile(r'(\w)- (\w)', re.UNICODE)
        self.re_notoneletterword = re.compile(r' [^a|à|y|ô|m]( |[^a-zA-Z]|\n|$)', re.IGNORECASE)
        self.re_nonalpha = re.compile(r'[^a-zA-Z\sàâäéèêëïîôöùûüÿç]')
        self.re_elision = re.compile(r"(c|l|d|m|n|s|t|qu)(’|')")
        self.re_initiale = re.compile(r"[A-Z]\. ")
        self.re_digit = re.compile(r'\d')
        self.re_acronyms = re.compile(r' ([A-Z]\.)+( |$)')
        self.list_punct = ('.', ',', ':', ';', '...', '!', '?', '-', '—' )


        self.sym_spell = SymSpell()
        # 0, 1 pour dic unigramme, 0,2 pour dic bigramme
        dictpath = pkg_resources.resource_filename(__name__,f"data/{lg}/{lg}_dic.txt" )
        self.sym_spell.load_dictionary(dictpath, 0, 1)

        dictpath = pkg_resources.resource_filename(__name__,f"data/{lg}/{lg}-100k.txt" )
        self.sym_spell.load_dictionary(dictpath, 0, 1)

#         self.setdictionnary = set(self.sym_spell.words.keys())

        stopwordpath = pkg_resources.resource_filename(__name__,f"data/{lg}/{lg}_stopwords.txt" )
        with open(stopwordpath, encoding='utf-8') as f:
            self.stopwords = f.read()

#         self.stopwords = self.stopwords.split('\n')
#         self.stopwords = set(self.stopwords)

#         self.df_firstnames = pd.read_csv(pkg_resources.resource_filename(__name__,"data/firstnames.csv" ), sep=';')
#         # one of the name in the list is "nan": it needs to be converted into str to avoid being considered NaN
#         self.df_firstnames['firstnames'] = self.df_firstnames['firstnames'].astype(str)


#         selected_names = self.df_firstnames[
#             # TODO: change language column in firstnames.csv to the ISO value
#             self.df_firstnames['language'] == 'french'

#         ]['firstnames']
#         self.normfirstnameset = set(map(unidecode.unidecode, selected_names))
#         self.normfirstnameset = set(map(str.lower, self.normfirstnameset))
#         # print(self.normfirstnameset)

#         self.normsetdictionnary = set(map(unidecode.unidecode, self.setdictionnary))
#         self.normsetdictionnary = set(map(str.lower, self.normsetdictionnary))
        
        # gramdicpath = pkg_resources.resource_filename(__name__,f"data/{lg}/3_gram_dic.json" )
        # with open(gramdicpath, encoding='utf-8') as f:
        #     gramdic = json.load(f)
        #     self.setgramdic = set(gramdic.keys())
        self.dichyphen = pyphen.Pyphen(lang=self.lg)
                
        sylldicpath = pkg_resources.resource_filename(__name__,f"data/{lg}/syll_dic.txt" )
        with open(sylldicpath, encoding='utf-8') as f:
            sylldic = f.read()
            self.sylldic = set(sylldic.split('\n'))


    def clean_text(self, text: str) -> str:
        """
        Cleans text by:
        * Adding space after stop that is stuck to text
        * Replaces newline by space
        """
        # print(text)
        text = re.sub(self.re_acronym_sent, r'\1 \2', text)
        # text = re.sub(self.re_hyphen, r'\1\2', text)
        text = text.replace('\n', " ")
        return text

    def remove_hyphen(self, text: str) -> str:
        text = re.sub(self.re_hyphen, r'\1\2', text)
        return text

    def process(self, file: str, return_str = True) -> str:
        """
        Preprocess whole document by:
        * removing hyphens
        * postcorrecting ocr
        By default returns a string representation of the file
        but can also return BeautifulSoup format of the file.
        """
#         only_textblock_tags = SoupStrainer("textblock")
#         # alto_page = BeautifulSoup(alto_page, 'lxml-xml', parse_only=only_string_tags)

#         with open(file, encoding='utf-8') as f:
#             # alto_page = BeautifulSoup(f, 'lxml-xml')
#             alto_page = BeautifulSoup(f, 'lxml-xml', parse_only=only_textblock_tags)
#         print(alto_page)    
#         list_textblock = alto_page.find_all('textblock')
#         if list_textblock:
#             self.correct_hyphen(list_textblock)
        with open(file, encoding='utf-8') as f:
            alto_page = BeautifulSoup(f, 'lxml-xml')
            
        ocr_tag = alto_page.find('ocr')
        ocr_tag.extract()
        
        print('Processing hyphens...')
        self.correct_hyphen(ocr_tag)
        # processed_alto = self.correct_hyphen(file, return_str=False)

        print('Hyphen processing done !')
        print('Processing OCR...')
        # processed_alto = self.correct_ocr(file, return_str = False, is_file = True)
        self.correct_ocr(ocr_tag, is_file = False)
        print('OCR processing done !')
        # self.correct_ocr(alto_page.prettify())
        
        gallica_tag = alto_page.find('gallica_document')
        gallica_tag.append(ocr_tag)
        
        if return_str:
            return alto_page.prettify()
        return alto_page

#         print('Processing hyphens...')
#         processed_alto = self.correct_hyphen(file)
#         # processed_alto = self.correct_hyphen(file, return_str=False)

#         print('Hyphen processing done !')
#         print('Processing OCR...')
#         # processed_alto = self.correct_ocr(file, return_str = False, is_file = True)
#         processed_alto = self.correct_ocr(processed_alto, return_str = False, is_file = False)
#         print('OCR processing done !')
#         # self.correct_ocr(alto_page.prettify())
        
#         if return_str:
#             return processed_alto.prettify()
#         return processed_alto

    def correct_hyphen(self, file:str, return_str = True, is_file = True) -> None:
        """
        Delete HYP tag from Alto document
        """

#         def preprocess_subscontent(previous_string_tag: Tag) -> bool:
#             """
#             Checks if text within subs_content attribute is valid (just text, no integer)
#             Returns False if it is not valid, True if else
#             """
#             re_subscontent = re.compile(r'^[a-zA-Z]+$')
#             subs_content = previous_string_tag['subs_content']
#             search_subscontent = re.search(re_subscontent, subs_content)
#             if search_subscontent:

#                 return True
#             else:
#                 return False

        # def correct_textblock(textblock: Tag, next_textblock: Tag) -> None:

        def correct_textblock(textblock: Tag) -> None:
            """            
            If HYP tag is found at the end of a TextBlock tag, then the
            word is running between two blocks, without the attribute 'subs_content',
            Deletes HYP tag from the XML tree

            :param textblock: TextBlock tag to process
            :type textblock: Tag
            """
            
            tb_hyp_tag = textblock.find('hyp')
            if tb_hyp_tag:

                tb_hyp_tag.extract()

                # previous_string_tag = tb_hyp_tag.find_previous('string')
                # next_string_tag = next_textblock.find('string')

                # if subs_content text is not valid, the attribut is deleted
                # else, its value is modified as the combination of the text from
                # both string tags. The next string tag is deleted

                # if previous_string_tag.has_attr('subs_content'):
                #     print('prev', previous_string_tag)
                #     subs_content = f"{previous_string_tag['content']}{next_string_tag['content']}"
                #     previous_string_tag['subs_content'] = subs_content

                #     next_string_tag.extract()
                    
#                     extract_next_string = preprocess_subscontent(previous_string_tag)
#                     if extract_next_string:

#                         # adds subs_content attribute, filled with text from both string tags
#                         subs_content = f"{previous_string_tag['content']}{next_string_tag['content']}"
#                         previous_string_tag['subs_content'] = subs_content

#                         next_string_tag.extract()
#                     else:
#                         del previous_string_tag['subs_content']
#                         del next_string_tag['subs_content']

        def correct_textline(line: Tag, next_line: Tag) -> None:
            """
            HYP tag appears in the first context : deletes HYP tag and the
            next String tag from the XML Tree

            :param line: TextLine tag to process
            :type line: Tag
            :param next_line: Next TextLine tag in the XML Tree
            :type next_line: Tag
            """
            hyp_tag = line.find('hyp')
            next_string_tag = next_line.find('string')

            if hyp_tag:

                previous_string_tag = hyp_tag.find_previous('string')
                if previous_string_tag.has_attr('subs_content'):
                    
                    hyp_tag.extract()
                    next_string_tag.extract()


                    # extract_next_string = preprocess_subscontent(previous_string_tag)

                    # if subs_content text is not valid, the attribute is deleted
                    # else, the next string tag is deleted
                    # if extract_next_string:
                    #     next_string_tag.extract()
                    # else:
                    #     del previous_string_tag['subs_content']
                    #     del next_string_tag['subs_content']
                    
        if isinstance(file, Tag):
            alto_page = file
        else:
            if is_file:
                with open(file, encoding='utf-8') as f:
                    # alto_page = BeautifulSoup(f, 'lxml-xml', parse_only=only_string_tags)
                    alto_page = BeautifulSoup(f, 'lxml-xml')

            else:
                # alto_page = BeautifulSoup(file, 'lxml-xml', parse_only=only_string_tags)
                alto_page = BeautifulSoup(file, 'lxml-xml')   

        list_textblock = alto_page.find_all('textblock')
        
        if list_textblock:
            for tb, next_tb in zip(list_textblock, list_textblock[1:]):

                list_lines = tb.find_all('textline')
                # deletes each HYP tag and String tag at end and beginning of both lines
                for line, next_line in zip(list_lines, list_lines[1:]):
                    correct_textline(line, next_line)

                # deletes HYP tag at TextBlock end
                correct_textblock(tb, next_tb)

            # same operation, but on the last TextBlock element
            last_tb = list_textblock[-1]
            list_lines = last_tb.find_all('textline')
            for line, next_line in zip(list_lines, list_lines[1:]):
                correct_textline(line, next_line)
                
    def extract_word_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts sets of metadata abouth each word (line)
        contained in df
        """
        def get_word_length(word):
            return len(word)

        def stw_capital(word):
            try:
                if word[0].isupper():
                    return 1
                return 0
            except:
                return 0

        def stw_elision(word):
            try:
                elision_search = re.match(self.re_elision, word)
                if elision_search:
                    return 1
                return 0
            except:
                return 0

        def is_capitalized(word):
            try:
                if word.isupper():
                    return 1
                return 0
            except:
                return 0

        def non_alpha_prop(word:str) -> int:
            """
            Proportion of non alphanumerical characters in word

            :param word: Word to process
            :type word: str
            :return: Proportion of non alphanumerical characters in word
            :rtype: int
            """
            try:
                norm_word = unidecode.unidecode(word)
                non_alpha = re.findall(r'[^a-zA-Z\d\s:]', norm_word)
                proportion = len(non_alpha) / len(word)
                proportion =  int(proportion * 100)
                return proportion
            except:
                return 100

        def ends_punct(word: str) -> int:
            """
            Determines if word ends with punctuation

            :param word: Word to process
            :type word: str
            :return: 1 if word ends with punctuation, else 0
            :rtype: int
            """
            try:
                punct_search = re.search(r'[\.,!?\)]$', word)
                if punct_search:
                    return 1
                return 0      
            except:
                return 1

        def is_hyphen(word):
            if word == '-':
                return 1
            return 0

        def is_punct(word):
            if word in self.list_punct:

                return 1
            return 0

        def is_digit(word):
            try:
                if re.search(r'\d/\d', word):
                    return 1

                int_word = int(word)
                return 1
            except:
                return 0
        #
        # def is_one_letter(word):
        #     if word in ('a', 'à', "y", 'ô'):
        #         return 1
        #     return 0

        def is_stopword(word):
            if word.lower() in self.stopwords:
                return 1
            return 0

        def is_in_dict(word):
            if word in self.sym_spell.words:
                return 1
            elif word.lower() in self.sym_spell.words:
                return 1
            return 0
        
        df['len'] = df['content'].apply(get_word_length)
        df['stw_capital'] = df['content'].apply(stw_capital)
        df['stw_elision'] = df['content'].apply(stw_elision)
        df['is_capitalized'] = df['content'].apply(is_capitalized)
        df['non_alpha_prop'] = df['content'].apply(non_alpha_prop)
        df['ends_punct'] = df['content'].apply(ends_punct)
        df['freq'] = df['content'].map(df['content'].value_counts())
        df['is_punct'] = df['content'].apply(is_punct)
        df['is_stopword'] = df['content'].apply(is_stopword)
        df['is_digit'] = df['content'].apply(is_digit)
        df['is_in_dict'] = df['content'].apply(is_in_dict)
        df['correction'] = df['content'] # by default, keeps the same word
        df['operation'] = 'keep' # by default, the operation is to keep the content as it is.
        # Other operations: substitute, delete
        
        return df
    
    def correct_ocr(self, file:str, return_str=True, is_file=True) -> None:
        """
        Sets of rules to process and correct OCR documents.
        Each string tag in the XML document is added to a DataFrame,
        along with metadata (length, case, ...). Each word is asigned
        an operation (either: keep, delete, substitute). The set of rules
        determines which operation is to be done on that word. The DF is
        then processed to apply the operations on the XML file.
        """
        
        def remove_nonalpha(df: pd.DataFrame, threshold = 75) -> None:
            """
            Adds delete operation to word with nonalpha 
            character above threshold and is not a punctuation

            :param df: Feature DataFrame
            :type df: pd.DataFrame
            :param threshold: Proportion threshold of non-alphanumeric characters, defaults to 75
            :type threshold: int, optional
            """
            candidates = df[
                (df['non_alpha_prop'] > threshold)
                & (df['is_punct'] == 0)
            ]
            
            df.loc[df.index.intersection(candidates.index), 'operation'] = 'delete'
            
        def remove_onechar(df: pd.DataFrame) -> None:
            """
            Adds delete operation to word of length 1, 
            not a punctuation, a digit and not in the SymSpell dictionnary

            :param df: Feature DataFrame
            :type df: pd.DataFrame
            """
            re_notoneletterword = re.compile(r'[^a|à|y|ô|m]', re.IGNORECASE)
            candidates = df[
                (df['content'].str.contains(re_notoneletterword))
                & (df['len'] == 1)
                & (df['is_punct'] == 0)
                & (df['is_digit'] == 0)

            ]
            df.loc[df.index.intersection(candidates.index), 'operation'] = 'delete'
            
        def spell_corrector(df: pd.DataFrame) -> None:
            """
            Adds substitute operation using the SymSpell algorithm 
            to correct words.

            :param df: Feature DataFrame
            :type df: pd.DataFrame
            """
            def correct(candidate: str) -> str:
#               removes characters that repeat themselves more than 2 times (eg: iiiii)
                re_samechar = re.compile(r'([a-z])\1{2,}')
                candidate = re.sub(re_samechar, r'\1', candidate)
#               suggests spelling correction for given candidate and given distance
                suggestions = self.sym_spell.lookup(candidate, Verbosity.CLOSEST,
                                              max_edit_distance=1, include_unknown=True,
                                              transfer_casing=True)
                newterm = suggestions[0].term

                return newterm
            
            candidates = df[
                (df['stw_capital'] == 0)
                & (df['ends_punct'] == 0)
                & (df['is_punct'] == 0)
                & (df['is_digit'] == 0)
                & (df['stw_elision'] == 0)
                & (df['operation'] != 'delete')
            ]
            corrections = candidates['content'].apply(correct)
            df['correction'].update(corrections)
            df.loc[df['content'] != df['correction'], 'operation'] = 'correct'


            # df.loc[df['content'] != df['correction'], 'operation'] = 'substitute'

        print('Collecting metadata ...')
#         creates DF with string tag from XML doc

        if isinstance(file, Tag):
            alto_page = file
        else:
            if is_file:
                with open(file, encoding='utf-8') as f:
                    alto_page = BeautifulSoup(f, 'lxml-xml')

            else:
                alto_page = BeautifulSoup(file, 'lxml-xml')   

        stringtags = alto_page.find_all('string', subs_content = None)
        ids = pd.Series([x['id'] for x in stringtags])
        contents = pd.Series([x['content'] for x in stringtags])
        stringdf = pd.concat([ids, contents], axis=1)
        stringdf['is_substring'] = 0
        
        
        substrings = alto_page.find_all('string', subs_content = True)
        subids = pd.Series([x['id'] for x in substrings])
        subcontents = pd.Series([x['subs_content'] for x in substrings])
        subdf = pd.concat([subids, subcontents], axis=1)
        subdf['is_substring'] = 1
        
        df = pd.concat([stringdf, subdf]).reset_index()
        del df['index']
        df.rename(columns={0:'id', 1:'content'}, inplace=True)
        
#         extracts metadata about each string tag
#       then process it by adding / changing operations
        df = self.extract_word_metadata(df)
        remove_nonalpha(df)
        remove_onechar(df)
        spell_corrector(df)
        
        all_strings = stringtags + substrings
        
        print('Cleaning doc')

        print('Deleting tags ...')
        df_delete = df[(df['operation'] == 'delete')]

        del_tag = filter(lambda x: (x if not df_delete.loc[df_delete['id'] == x['id']].empty else False), all_strings)

        for tag in del_tag:
            tag.extract()
        print('Tags deleted !')
        
        
        print('Correcting tags...')
        df_subs = df[(df['operation'] == 'substitute')]

        
        subs_tag = filter(lambda x: (x if not df_subs.loc[df_subs['id'] == x['id']].empty else False), all_strings)
        for tag in subs_tag:
            tagid = tag['id']
            df_row = df_subs.loc[df_subs['id'] == tagid]
            is_substring = df_row['is_substring'].values[0]
            correction = df_row['correction'].values[0]
            tag['operation'] = 'substitution'
            if is_substring == 0:
                tag['content'] = correction
            else:
                tag['subs_content'] = correction 
        

        print('Tags corrected !')        

    def text_readability(self, text: str) -> float:
        """
        TODO
        """

        def clean_text(text):
            # for rule in [self.re_acronyms, self.re_initiale, self.re_elision, self.re_digit, self.re_nonalpha]:
             # for rule in [self.re_acronyms, self.re_initiale, self.re_elision, self.re_digit, self.re_nonalpha]:
                # text = re.sub(rule, '', text)
            text = text.lower()
            text = re.sub(self.re_elision, '', text)
            # text = unidecode.unidecode(text)
            return text
        
        def to_syllables(tokentext):
            list_syllables = []
            for word in tokentext:
                syll_words = self.dichyphen.inserted(word)
                syll_words = syll_words.split('-')
                list_syllables.extend(syll_words)
            return list_syllables
        def get_readability_mark(score):
            score = score * 100
            if score in range(0, 20):
                return 'E'
            elif score in range(21, 40):
                return 'D'
            elif score in range(41, 60):
                return 'C'
            elif score in range(61, 80):
                return 'B'
            else:
                return 'A'
        
        text = clean_text(text)
        tokentext = text.split()
        syllable_text = to_syllables(tokentext)
        common_syll = [syll for syll in syllable_text if syll in self.sylldic]
        if not len(common_syll) == 0:
            score = len(common_syll) / len(syllable_text)
            score = float("{:.2f}".format(score))
        else:
            score = 0
        mark = get_readability_mark(score)
        return score, mark
        
#     def text_readability(self, text: str, n=3) -> float:
#         """
#         TODO
#         """
#         def clean_text(text):
#             # for rule in [self.re_acronyms, self.re_initiale, self.re_elision, self.re_digit, self.re_nonalpha]:
#              # for rule in [self.re_acronyms, self.re_initiale, self.re_elision, self.re_digit, self.re_nonalpha]:
#              #    text = re.sub(rule, '', text)
#             text = text.lower()
#             text = text.replace(' ', '') 
#             return text
        
#         def to_ngram(text, n=3):
#             grams = ngrams(text, n)
#             list_grams = [''.join(g) for g in grams]
#             return list_grams
        
#         text = clean_text(text)
#         text = unidecode.unidecode(text)
#         textgrams = to_ngram(text, n=n)
        
#         common_grams = list(filter( lambda x: (x if x in self.setgramdic else False), textgrams))
#         if not len(common_grams) == 0:
#             prop = len(common_grams) / len(textgrams)
#             prop = float("{:.2f}".format(prop))
#         else:
#             prop = 0
#         return prop



if __name__ == "__main__":

    
    pass
    # ocr_doc = BeautifulSoup('<ocr_doc></ocr_doc>', features='xml')
    #
    # pre = Preprocessor()
    # pre.preprocess(ocr_doc
    #                )