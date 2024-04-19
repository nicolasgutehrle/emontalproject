import pandas as pd


class Postprocessor:
    """
    Class to postprocess results obtained from rules
    """

    def postprocess_tables(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess data that is table type
        TODO: TO COMPLETE
        """
        df_tables = page_metadata[
            page_metadata['block_type'] == 'table'
            ]
        df_tables['is_text'] = 0
        df_tables['is_firstline'] = 0
        df_tables['is_title'] = 0
        df_tables['is_header'] = 0
        return df_tables
        # page_metadata.update(df_tables)
        # return page_metadata

    def postprocess_text(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess data that is text type
        """

        def add_firsttitle(df: pd.DataFrame) -> pd.DataFrame:
            """
            Makes sure the first line of the document is a title
            """
            firstline_header = df.loc[0, 'is_header']
            if firstline_header != 1:
                df.loc[0, 'is_title'] = 1
                df.loc[0, 'post_rule'] = 4
            return df

        def add_firstline(df: pd.DataFrame) -> None:
            """
            Makes sure that there is at least one line considered 'first_line' after a title
            so that to avoid unwanted bugs
            """
            df['prev_title'] = df['is_title'].shift(1)
            df['next_title'] = df['is_title'].shift(-1)
            df = df.fillna(0)

            titles = df[
                (df['is_title'] == 1)
                ]

            subdf = df[~df.isin(titles)].dropna()
            firstlines = subdf[
                (subdf['is_firstline'] == 0)
                & (subdf['prev_title'] == 1)
                ]

            firstlines['is_firstline'] = 1
            firstlines['post_rule'] = 5
            df.update(firstlines[['is_firstline', 'post_rule']])
            return df

            # return firstlines

        # df_text = page_metadata[
        #     (page_metadata['block_type'] == 'text')
        #     | (page_metadata['block_type'] == 'title')
        # ]
        df_text = add_firsttitle(page_metadata)

        # page_metadata = add_firsttitle(page_metadata)

        # df_text = align_df(df_text)
        df_text = add_firstline(df_text)
        # df_text.update(title_firstline_resolution(df_text))
        # page_metadata = add_firstline(page_metadata)
        # df_text.update(add_firstline(df_text))

        return df_text

    def postprocess_headers(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Postprocess header type dataline
        """

        df_headers = page_metadata[page_metadata['is_header'] == 1]
        df_headers['is_title'] = 0
        df_headers['title_rule'] = 0

        df_headers['is_table'] = 0
        df_headers['table_rule'] = 0

        df_headers['is_firstline'] = 0
        df_headers['firstline_rule'] = 0

        df_headers['is_other'] = 0
        df_headers['other_rule'] = 0

        df_headers['post_rule'] = 6

        page_metadata.update(df_headers)

        page_metadata['prev_header'] = page_metadata['is_header'].shift(1)
        page_metadata['next_header'] = page_metadata['is_header'].shift(-1)
        page_metadata = page_metadata.fillna(0)

        # Sets every line in the first 30 lines of the first page
        # as header if they are preceded / followed by a header
        firstpage = page_metadata[
            (page_metadata['page'] == 1)
        ]
        firstpage = firstpage.iloc[:30]

        # sets as header the first line of the document up to the last detected header
        # (among the first 30 lines of the first page)
        last_header_index = firstpage[firstpage['is_header'] == 1]
        if not last_header_index.empty:
            last_header_index = last_header_index.index[-1]
            firstpage.loc[:last_header_index, 'is_header'] = 1
            firstpage.loc[:last_header_index, 'post_rule'] = 7

            # firstpage.loc[
            #     (firstpage['prev_header'] == 1)
            #     | (firstpage['next_header'] == 1),
            #     'is_header'] = 1

            firstpage['is_title'] = 0
            firstpage['title_rule'] = 0

            # df_headers['is_subtitle'] = 0
            # df_headers['subtitle_rule'] = 0

            firstpage['is_table'] = 0
            firstpage['table_rule'] = 0

            firstpage['is_firstline'] = 0
            firstpage['firstline_rule'] = 0

            firstpage['is_other'] = 0
            firstpage['other_rule'] = 0

            # the first line afte the last header becomes a title
            firstpage.loc[last_header_index + 1, 'is_title'] = 1
            firstpage.loc[last_header_index + 1, 'is_table'] = 0
            firstpage.loc[last_header_index + 1, 'is_firstline'] = 0
            firstpage.loc[last_header_index + 1, 'is_header'] = 0
            firstpage.loc[last_header_index + 1, 'is_other'] = 0

        # page_metadata.update(firstpage)
        # return page_metadata
        return firstpage

    def postprocess_titles(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Sets of rules to resolve conflict between blocks that are classified as
        titles and something else at the same time
        """
        def title_firstline_resolution(page_metadata: pd.DataFrame) -> pd.DataFrame:
            """

            """
            conflict = page_metadata[
                (page_metadata['is_title'] == 1)
                & (page_metadata['is_firstline'] == 1)
            ]
            conflict['post_rule'] = 8
            rule1 = conflict[
                (conflict['line_following_space'] <= conflict['linespace_median'] + 10)
                & (conflict['capital_prop'] < 15)
            ]
            rule1['is_title'] = 0
            # page_metadata.update(rule1)
            rest = conflict[~conflict['line_id'].isin(rule1['line_id'])]

            rule2 = rest[
               (rest['line_previous_space'] > rest['linespace_median'])
                & (rest['line_following_space'] > rest['linespace_median'])
                ]
            rule2['is_firstline'] = 0


            # page_metadata.update(rule2)

            rest = rest[~rest['line_id'].isin(rule2['line_id'])]
            rest['is_firstline'] = 0

            df_titles = pd.concat([rule1, rule2, rest])
            return df_titles
            # page_metadata.update(rest)
            # return page_metadata

        def title_text_resolution(page_metadata: pd.DataFrame) -> pd.DataFrame:
            """

            """
            df_titles = page_metadata[page_metadata['is_title'] == 1]
            df_titles['is_text'] = 0
            return df_titles

        df_titles = title_firstline_resolution(page_metadata)
        df_titles = title_text_resolution(page_metadata)
        return df_titles

    def postprocess_firstlines(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Sets of rules to resolve conflict between blocks that are classified as
        titles and something else at the same time
        """
        df_firstlines = page_metadata[page_metadata['is_firstline'] == 1]
        df_firstlines['is_text'] = 0
        return df_firstlines

    def postprocess(self, page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Main methods: postprocess successively:
        * text type
        * table type
        """
        page_metadata['post_rule'] = 0

        df_headers = self.postprocess_headers(page_metadata)
        page_metadata.update(df_headers)

        # page_metadata = self.postprocess_headers(page_metadata)
        df_titles = self.postprocess_titles(page_metadata)
        page_metadata.update(df_titles)

        df_tables = self.postprocess_tables(page_metadata)
        page_metadata.update(df_tables)

        df_firstlines = self.postprocess_tables(page_metadata)
        page_metadata.update(df_firstlines)

        df_text = self.postprocess_text(page_metadata)
        page_metadata.update(df_text)

        return page_metadata
