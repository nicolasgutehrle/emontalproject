import pandas as pd
from typing import List


def classify_block(df: pd.DataFrame, df_line: pd.DataFrame) -> pd.DataFrame:
    """
    Set of rules to classify textblock as either text, titles or other. This may be expanded later to add more classes (advertisement, tables, ...). Theses rules are only applied to block that don't have already a type (table, titre1, advertisement) from the OCR.

    :param df: Block features dataframe
    :type df: pd.DataFrame
    :param df_line: Line features dataframe
    :type df_line: pd.DataFrame
    :return: Updated block features dataframe
    :rtype: pd.DataFrame
    """

    def retrieve_text(df: pd.DataFrame) -> pd.DataFrame:

        df['text_rule'] = 0
        df['prev_id'] = df['block_id'].shift(1)
        df['next_id'] = df['block_id'].shift(-1)
        df = df.fillna(0)

        txt_block1 = df[
            # text block are those that have the most lines or the most words in it
            (df['block_linecounts'] > df['block_count_med'])
            | (df['block_word_count'] >= df['block_word_ratio'] / 3)
            ]

        txt_block1['block_text'] = 1
        txt_block1['text_rule'] = 1

        df.update(txt_block1[['block_text', 'text_rule']])


        # Any block that has less lines than the others but whose height is normal
        # and is surrounded by other text block is a text block
        txt_block2 = df[
             (df['block_linecounts'] <= df['block_count_med'])
             & (df['block_height'] <= df['doc_height'])
        ]
        txt_block2 = txt_block2[
            (txt_block2['prev_id'].isin(txt_block1['block_id'].values))
            | (txt_block2['next_id'].isin(txt_block1['block_id'].values))
        ]

        txt_block2['block_text'] = 1
        txt_block2['text_rule'] = 2

        df.update(txt_block2[['block_text', 'text_rule']])

        return df

    def retrieve_title(df: pd.DataFrame) -> pd.DataFrame:

        df['title_rule'] = 0
        baserule = df[
            df['block_type'] == 'titre1'
            ]
        baserule['block_title'] = 1
        baserule['title_rule'] = 1

        df.update(baserule[['block_table', 'title_rule']])

        df['prev_text'] = df['block_text'].shift(1)
        df['next_text'] = df['block_text'].shift(-1)
        df = df.fillna(0)

        titles = df[
            (df['prev_text'] == 1)
            & (df['block_text'] == 0)
            | (df['next_text'] == 1)
            & (df['block_text'] == 0)

            ]
        titles = titles[
            (titles['block_previous_space'] > titles['block_interline_space_med'] + 5)
            & (titles['block_linecounts'] <= 3)
            | (titles['block_following_space'] > titles['block_interline_space_med'] + 5)
            & (titles['block_linecounts'] <= 3)
        ]

        titles['block_title'] = 1
        titles['title_rule'] = 2

        df.update(titles[['block_title', 'title_rule']])
        return df

    # def retrieve_tables(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Rules to identify table blocks
    #     """
    #     df['table_rule'] = 0
    #     baserule = df[
    #         df['block_type'] == 'table'
    #         ]
    #     baserule['block_table'] = 1
    #     baserule['table_rule'] = 1

    #     df.update(baserule[['block_table', 'table_rule']])

    #     # df['block_table'] = 0
    #     # either the block width is smaller than 1/3 of the median block width
    #     # or the block starts with the word Total / Totaux
    #     tablerule1 = df[
    #         (df['block_width'] < df['doc_width'] / 3)
    #         | (df['ctn_total_word'] == 1)

    #         ]
    #     # for the selected block, either they have more than words than the median
    #     # or the proportion of digits in them is superior to 30%
    #     tablerule2 = tablerule1[
    #         (tablerule1['block_word_count'] > tablerule1['doc_words'])
    #         | (tablerule1['digits_prop'] >= 30)
    #         ]
    #     tablerule2['block_table'] = 1
    #     tablerule2['table_rule'] = 2

    #     df.update(tablerule2[['block_table', 'table_rule']])

    #     return df


    # def retrieve_headers(df: pd.DataFrame, df_line: pd.DataFrame) -> pd.DataFrame:

    #     # by default, no TextBlock is a Header block
    #     df['header_rule'] = 0

    #     # implementation of rule 4, to detect header block in pages other than the first page of document
    #     doc_firstpage_num = df['block_page'].min()

    #     # gets TextBlock tags from the first page
    #     firstpageblock = df[df['block_page'] == doc_firstpage_num]

    #     # only keeps the first 30 lines of the document
    #     filtered_lines = df_line[df_line['block_id'].isin(firstpageblock['block_id'])].dropna()
    #     filtered_lines = filtered_lines.iloc[:30]
    #     headers_lines = filtered_lines[
    #         (filtered_lines['simheader'] >= 90)
    #         | (filtered_lines['simtitle'] >= 90)
    #         | (filtered_lines['ctn_page'] == 1)
    #         | (filtered_lines['ctn_month'] == 1)
    #         | (filtered_lines['ctn_currency'] == 1)
    #         | (filtered_lines['ctn_address'] == 1)
    #         ]
    #     # each Block found in headers_lines is marked as Header
    #     rule1 = df[df['block_id'].isin(headers_lines['block_id'])].dropna()
    #     df.loc[df['block_id'].isin(rule1['block_id']), 'is_header'] = 1
    #     df.loc[df['block_id'].isin(rule1['block_id']), 'header_rule'] = 1

    #     # implementation of rule 5, to detect header block in pages other than the first page of document
    #     # gets TextBlock tags after the first page
    #     otherpageblock = df[df['block_page'] != doc_firstpage_num]
    #     # only keeps the first 4 lines
    #     filtered_lines = df_line[df_line['block_id'].isin(otherpageblock['block_id'])].dropna()
    #     filtered_lines = filtered_lines.iloc[:4]
    #     headers_lines = filtered_lines[
    #         (filtered_lines['simheader'] >= 90)
    #         | (filtered_lines['simtitle'] >= 90)
    #         | (filtered_lines['ctn_page'] == 1)
    #         | (filtered_lines['ctn_dash'] == 1)
    #         ]
    #     # each Block found in headers_lines is marked as Header
    #     rule2 = df[df['block_id'].isin(headers_lines['block_id'])].dropna()
    #     df.loc[df['block_id'].isin(rule2['block_id']), 'is_header'] = 1
    #     df.loc[df['block_id'].isin(rule2['block_id']), 'header_rule'] = 1
    #     return df


    def retrieve_headers(df: pd.DataFrame, df_line: pd.DataFrame) -> pd.DataFrame:
        # by default, no TextBlock is a Header block
        df['header_rule'] = 0

        # implementation of rule 4, to detect header block in pages 
        # other than the first page of document
        doc_firstpage_num = df['block_page'].min()

        # gets TextBlock tags from the first page
        firstpageblock = df[df['block_page'] == doc_firstpage_num]

        # only keeps the first 30 lines of the document
        filtered_lines = df_line[df_line['block_id'].isin(firstpageblock['block_id'])].dropna()
        filtered_lines = filtered_lines.iloc[:30]


        headers_lines = filtered_lines[
            (filtered_lines['simheader'] >= 90)
            | (filtered_lines['simtitle'] >= 90)
            | (filtered_lines['ctn_page'] == 1)
            | (filtered_lines['ctn_month'] == 1)
            | (filtered_lines['ctn_currency'] == 1)
            | (filtered_lines['ctn_address'] == 1)
            ]

        # each Block found in headers_lines is marked as Header
        rule1 = df[df['block_id'].isin(headers_lines['block_id'])].dropna()
        df.loc[df['block_id'].isin(rule1['block_id']), 'block_header'] = 1
        df.loc[df['block_id'].isin(rule1['block_id']), 'header_rule'] = 1

        # implementation of rule 5, to detect header block in pages 
        # other than the first page of document

         # gets TextBlock tags after the first page
        otherpageblock = df[df['block_page'] != doc_firstpage_num]
        # only keeps the first 4 lines
        filtered_lines = df_line[df_line['block_id'].isin(otherpageblock['block_id'])].dropna()
        filtered_lines = filtered_lines.iloc[:4]


        headers_lines = filtered_lines[
            (filtered_lines['simheader'] >= 90)
            | (filtered_lines['simtitle'] >= 90)
            | (filtered_lines['ctn_page'] == 1)
            | (filtered_lines['ctn_dash'] == 1)

            ]
        
        # each Block found in headers_lines is marked as Header
        rule2 = df[df['block_id'].isin(headers_lines['block_id'])].dropna()
        df.loc[df['block_id'].isin(rule2['block_id']), 'block_header'] = 1
        df.loc[df['block_id'].isin(rule2['block_id']), 'header_rule'] = 1

        return df

    def retrieve_others(df: pd.DataFrame, df_line: pd.DataFrame):
        """

        """

        df['other_rule'] = 0
        df['prev_linecount'] = df['block_linecounts'].shift(1)
        df['next_linecount'] = df['block_linecounts'].shift(-1)
        df = df.fillna(0)

        small_blocks = df[
            df['block_linecounts'] <= df['block_count_med']
            ]

        ## goes through line of text contained in small blocks
        filtered_lines = df_line[df_line['block_id'].isin(small_blocks['block_id'])].dropna()

        # headers_lines = filtered_lines[
        #     (filtered_lines['simheader'] >= 90)
        #     | (filtered_lines['simtitle'] >= 90)
        #     | (filtered_lines['ctn_page'] == 1)
        #     | (filtered_lines['ctn_month'] == 1)
        #     | (filtered_lines['ctn_currency'] == 1)
        #     | (filtered_lines['ctn_address'] == 1)
        #
        #     ]
        #
        # rule1 = small_blocks[small_blocks['block_id'].isin(headers_lines['block_id'])].dropna()
        #
        # rule1['block_other'] = 1
        # rule1['other_rule'] = 1
        # df.update(rule1)

        other_lines = filtered_lines[
            # (filtered_lines['capital_prop'] > 55)
             (filtered_lines['non_alpha_prop'] >= 35)
            | (filtered_lines['ctn_dash'] == 1)
            & (filtered_lines['word_count'] <= 4)  # ARBITRARY HERE
            | (filtered_lines['digits_prop'] >= 35)
            ]

        rule2 = small_blocks[small_blocks['block_id'].isin(other_lines['block_id'])].dropna()

        rule2['block_other'] = 1
        rule2['other_rule'] = 2
        df.update(rule2)

        start_idx, count = 0, 0
        l_subdf = []

        # We start by making groups of block that contain less lines than the median.
        # For that, we go through each block. Each block is added to a group until a
        # block with more lines than the median is found, which closes the group.
        # The next starts with the following block with less lines than the median.
        for page in small_blocks['block_page'].unique():
            doc_page = small_blocks[small_blocks['block_page'] == page]
            for index, next_index in zip(doc_page.index, doc_page.index[1:]):
                if next_index == index + 1:
                    if count == 0:
                        start_idx = index
                    count += 1
                else:
                    if count != 0:
                        subdf = doc_page.loc[start_idx:index]

                        subdf['group'] = f"group_{start_idx}"
                        count = 0
                        l_subdf.append(subdf)

        if l_subdf:
            groups = pd.concat(l_subdf).dropna()

            # below, calculates the proportion of each predicted type (text, table, title) in the groups
            groups['count'] = groups['group'].map(groups['group'].value_counts())

            grpby = groups.groupby('group')
            grp_sum = grpby.sum().reset_index()

            count = grpby.first()['count'].values
            tl = grpby.first()['block_line'].values

            grp_sum['count'] = count
            grp_sum['block_line'] = tl

            grp_sum['freq_text'] = grp_sum['block_text'].values / count * 100
            grp_sum['freq_table'] = grp_sum['block_table'].values / count * 100
            grp_sum['freq_title'] = grp_sum['block_title'].values / count * 100

            ## filters out by taking groups with more than 2 groups
            ## and who has not a type that represent more than 40% of
            # the group
            grp_sum = grp_sum[grp_sum['count'] > 2]
            freq_threshold = 40
            grp_sum = grp_sum[
                (grp_sum['freq_text'] <= freq_threshold)
                & (grp_sum['freq_title'] <= freq_threshold)
                & (grp_sum['freq_table'] <= freq_threshold)
                ]

            rule3 = groups[groups['group'].isin(grp_sum['group'])]

            rule3['block_other'] = 1
            rule3['other_rule'] = 3
            df.update(rule3)

        rest = df[
            (df['block_text'] == 0)
            & (df['block_title'] == 0)
            & (df['block_table'] == 0)
            & (df['block_other'] == 0)
            ]
        rest['block_other'] = 1
        rest['other_rule'] = 4

        df.update(rest[['block_other', 'other_rule']])

        return df

    def postprocess(df: pd.DataFrame, postprocess_others:bool = False) -> pd.DataFrame:
        """
        Sets of steps to postprocess data, mainly resolve conflict between two predictions

        :param df: Block feature dataframe, after applying the rules
        :type df: pd.DataFrame
        :param postprocess_others: Either to post-process blocks labelled as Other, defaults to False
        :type postprocess_others: bool, optional
        :return: Postprocessed block feature dataframe
        :rtype: pd.DataFrame
        """

        def text_title_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """
            Set of rules to resolve conflict for block both classed as text and title
            """
            conflict_text_title = df[
                (df['block_title'] == 1)
                & (df['block_text'] == 1)
                ]
            # a conflicted block is a title if its height is above or equal to the median height and a half
            postproc_titles = conflict_text_title[
                (conflict_text_title['block_height'] >= conflict_text_title['block_height_med'] + conflict_text_title[
                    'block_height_med'] / 2)
            ]
            # Otherwise, it is a text block
            postproc_text = conflict_text_title[~conflict_text_title.isin(postproc_titles)].dropna()

            postproc_titles['block_text'] = 0
            postproc_titles['text_rule'] = 0
            postproc_titles['block_post_rule'] = -1

            postproc_text['block_title'] = 0
            postproc_titles['title_rule'] = 0
            postproc_text['block_post_rule'] = -1

            df.update(postproc_titles[['block_text', 'text_rule', 'block_post_rule']])
            df.update(postproc_text[['block_title', 'title_rule', 'block_post_rule']])

            return df

        def title_table_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """
            Set of rules to resolve conflict for block both classed as table and title
            """
            # TODO: IMPLEMENTER PLUS TARD
            conflict = df[
                (df['block_table'] == 1)
                & (df['block_title'] == 1)
                ]
            conflict['block_table'] = 0
            conflict['table_rule'] = 0
            conflict['block_post_rule'] = -2
            df.update(conflict[['block_text', 'text_rule', 'block_post_rule']])
            # pass
            return df

        def text_table_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """
            Set of rules to resolve conflict for block both classed as text and tables.
            At the moment, conflict is resolved by setting blocks to table type
            """
            conflict = df[
                (df['block_table'] == 1)
                & (df['block_text'] == 1)
                ]
            conflict['block_text'] = 0
            conflict['text_rule'] = 0
            conflict['block_post_rule'] = -2
            df.update(conflict[['block_text', 'text_rule', 'block_post_rule']])
            return df

        def text_other_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """

            """
            conflict = df[
                (df['block_text'] == 1)
                & (df['block_other'] == 1)
            ]

            conflict.loc[conflict['block_linecounts'] >= 4, 'block_other'] = 0
            conflict.loc[conflict['block_linecounts'] >= 4, 'other_rule'] = 0

            conflict.loc[conflict['block_linecounts'] < 4, 'block_text'] = 0
            conflict.loc[conflict['block_linecounts'] < 4, 'text_rule'] = 0

            conflict['block_post_rule'] = -3

            df.update(conflict[['block_other', 'block_text', 'other_rule', 'text_rule', 'block_post_rule']])

            return df

        def table_other_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """

            """
            conflict = df[
                (df['block_table'] == 1)
                & (df['block_other'] == 1)
            ]
            conflict['block_other'] = 0
            conflict['other_rule'] = 0
            conflict['block_post_rule'] = -4
            df.update(conflict[['block_other', 'other_rule', 'block_post_rule']])
            return df

        def title_other_resolution(df: pd.DataFrame) -> pd.DataFrame:
            """

            """
            conflict = df[
                (df['block_title'] == 1)
                & (df['block_other'] == 1)
            ]

            resolution = conflict[
                (conflict['block_linecounts'] == 1)
                & (conflict['prev_linecount'] >= 10)
                | (conflict['block_linecounts'] == 1)
                & (conflict['next_linecount'] >= 10)
            ]

            conflict.loc[conflict['block_id'].isin(resolution['block_id']), 'block_other'] = 0
            conflict.loc[conflict['block_id'].isin(resolution['block_id']), 'other_rule'] = 0

            conflict.loc[~conflict['block_id'].isin(resolution['block_id']), 'block_title'] = 0
            conflict.loc[~conflict['block_id'].isin(resolution['block_id']), 'title_rule'] = 0
            conflict['block_post_rule'] = -5

            df.update(conflict[['block_other', 'block_title', 'block_post_rule']])
            return df

        # def header_resolution(df:pd.DataFrame) -> pd.DataFrame:
        #     """
        #     Solves conflict for header block by setting them to header

        #     :param df: TextBlock Feature DataFrame
        #     :type df: pd.DataFrame
        #     :return: Updated TextBlock Feature DataFrame
        #     :rtype: pd.DataFrame
        #     """
        #     # selects all rows marked as headers
        #     headers = df[df['is_header'] == 1]
        #     # selects conflicting rows
        #     conflict = headers[
        #         (headers['is_text'] == 1)
        #         | (headers['is_title'] == 1)
        #         | (headers['is_other'] == 1)
        #     ]
        #     # implementation of Rule 6, to solve conflict
        #     # between Header, Text or Title
        #     resolution = conflict[
        #         (conflict['block_linecounts'] < 15)
        #         & (conflict['block_word_count'] < 50)
        #     ]
        #     resolution['is_text'] = 0
        #     resolution['is_title'] = 0
        #     resolution['is_other'] = 0
        #     df.update(resolution[[
        #         'is_text', 'is_title'
        #         ]])

        #     return df


        def header_resolution(df:pd.DataFrame) -> pd.DataFrame:
            """
            Solves conflict for header block by setting them to header

            :param df: TextBlock Feature DataFrame
            :type df: pd.DataFrame
            :return: Updated TextBlock Feature DataFrame
            :rtype: pd.DataFrame
            """
            # selects all rows marked as headers
            headers = df[df['block_header'] == 1]
            # selects conflicting rows
            conflict = headers[
                (headers['block_text'] == 1)
                | (headers['block_title'] == 1)
                # | (headers['block_table'] == 1)
                | (headers['block_other'] == 1)
            ]
            # implementation of Rule 6, to solve conflict
            # between Header, Text or Title
            resolution = conflict[
                (conflict['block_linecounts'] < 15)
                & (conflict['block_word_count'] < 50)
            ]
            resolution['block_text'] = 0
            resolution['block_title'] = 0
            # resolution['block_table'] = 0
            resolution['block_other'] = 0
            resolution['block_post_rule'] = -6
            df.update(resolution[['block_text', 'block_title', 'block_table', 'block_other', 'block_post_rule']])

            return df

        if not postprocess_others:
            df = header_resolution(df)

            df = text_title_resolution(df)
            df = text_table_resolution(df)
            df = title_table_resolution(df)
        else:
            df = text_other_resolution(df)
            df = table_other_resolution(df)
            df = title_other_resolution(df)
        return df

    def convert_to_labels(df: pd.DataFrame) -> List[str]:
        """
        Converts predicted predictions into list of labels

        :param df: Block feature dataframe
        :type df: pd.DataFrame
        :return: List of labels for each block
        :rtype: List[str]
        """
        predicted_class = []
        for text, title, table, header, other in zip(df['block_text'].values, df['block_title'].values, df['block_table'].values, df['block_header'], df['block_other'].values):
            if text == 1:
                predicted_class.append('text')
            elif title == 1:
                predicted_class.append('title')
            elif table == 1:
                predicted_class.append('table')
            elif header == 1:
                predicted_class.append('header')
            elif other == 1:
                predicted_class.append('other')
            else: # default value for block of text which havent been assigned a category
                predicted_class.append('other')
                # predicted_class.append('No_type')
                # predicted_class.append('text')

        return predicted_class

    # applies the rules 

    df['block_text'] = 0
    df['block_title'] = 0
    df['block_table'] = 0
    df['block_header'] = 0
    df['block_other'] = 0
    df['block_post_rule'] = 0

    df = retrieve_text(df)
    # df = retrieve_tables(df)
    df = retrieve_title(df)
    df = retrieve_headers(df, df_line)
    df = postprocess(df)

    df.update(df[['block_text', 'block_title', 'block_table', 'block_header', 'block_other']])
    df.update(df[['block_type', 'block_post_rule']])

    # df = retrieve_others(df, df_line)
    #
    # df = postprocess(df, postprocess_others=True)

    df['block_type'] = convert_to_labels(df)

    # df_text['block_type'] = convert_to_labels(df_text)
    df.update(df[['block_type', 'block_post_rule']])

    df['block_rule'] = df['text_rule'] + df['title_rule'] + df['header_rule']
    # df['block_rule'] = df['text_rule'] + df['title_rule'] + df['table_rule']
    # df['block_rule'] = df['text_rule'] + df['title_rule'] + df['table_rule'] + df['other_rule']
    return df
