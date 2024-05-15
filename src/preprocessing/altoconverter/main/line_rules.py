import pandas as pd

def classify_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Contains sets of rules to categorise TextLine

    :param df: Line feature dataframe
    :type df: pd.DataFrame
    :return: Updated line feature dataframe
    :rtype: pd.DataFrame
    """

    def retrieve_others(df: pd.DataFrame) -> pd.DataFrame:

        def find_others(df: pd.DataFrame) -> pd.DataFrame:
            """

            """
            df['is_other'] = 0
            df['other_rule'] = 0

            rule1 = df[
                (df['block_type'] == 'other')
            ]
            rule1['is_other'] = 1
            rule1['other_rule'] = 1
            df.update(rule1[['is_other', 'other_rule']])
            return df

        df = find_others(df)
        return df

    def retrieve_headers(df: pd.DataFrame) -> pd.DataFrame:

        def find_headers(df: pd.DataFrame) -> pd.DataFrame:
            """

            """
            df['is_header'] = 0
            df['header_rule'] = 0

            headers_block = df[df['block_type'] == 'header']
            headers_block['is_header'] = 1
            headers_block['header_rule'] = 1
            df.update(headers_block[['is_header', 'header_rule']])
            return df

        df = find_headers(df)

        return df


    def retrieve_titles(df: pd.DataFrame) -> pd.DataFrame:
        """
        Finds titles and subtitles in Alto document and postprocess the results
        """

        def find_titles(df: pd.DataFrame) -> pd.DataFrame:
            """
            Rules to identify titles in ALto document.
            There are 3 rules:
            * the line is the first of the document
            * the interline space before and after the line are above the means
            * either interline space is above the mean and the line height is above the means

            Results which height is greater than 95 pixel is a title, else is a subtitle
            """
            titre1 = df[
                (df['block_type'] == 'titre1')
                | (df['block_type'] == 'title')
            ]
            titre1['title_rule'] = 1

            df = df[
            df['block_type'] == 'text'
            ]

            # third_quartile = df['linespace_median'] + df['linespace_median'] / 2

            # finds every line that is the first of its block
            # zeros = df[
            #     (df['line_previous_space'] == 0)
            #     & (df['stw_capital'] == 1)
            # ]
            zeros = df[
                (df['line_previous_space'] == 0)
                & (df['stw_capital'] == 1)
                & (df['line_following_space'] > df['linespace_median'])
                & (df['simtitle'] < 60)
                & (df['simheader'] < 60)

                ]
            zeros['title_rule'] = 2
            subdf = df[~df.isin(zeros)].dropna()


    #         # finds line where both linespace is above the mean
            both_above = subdf[
                (subdf['line_previous_space'] >= subdf['linespace_median'] + subdf['linespace_median'] / 2)
                & (subdf['line_following_space'] >= subdf['linespace_median'] + subdf['linespace_median'] / 2)
                & (subdf['word_count'] < subdf['doc_word'])

                ]
            both_above['title_rule'] = 3
            subdf = subdf[~subdf.isin(both_above)].dropna()

    #         # finds title that are not as obvious than the others
            other_titles = subdf[
                (subdf['line_previous_space'] >= subdf['linespace_median'] + subdf['linespace_median'] / 2)
                & (subdf['capital_prop'] > 10)
                & (subdf['word_count'] < subdf['doc_word'])
                & (subdf['line_height'] <= subdf['height_median'])
                | (subdf['line_following_space'] >= subdf['linespace_median'] + subdf['linespace_median'] / 2)
                & (subdf['capital_prop'] > 10)
                & (subdf['word_count'] < subdf['doc_word'])
                & (subdf['line_height'] <= subdf['height_median'])

                ]
            other_titles['title_rule'] = 4
            # titles = pd.concat([both_above, zeros, other_titles, titre1])


            ## rules to identify small titles, which are similar more similar to other texts
            subdf = subdf[~subdf.isin(other_titles)].dropna()
            small_titles = subdf[
                (subdf['diff_hpos'] >= 105)
                & (subdf['capital_prop'] > 0)
                & (subdf['line_previous_space'] > subdf['linespace_median'])
                & (subdf['line_following_space'] > subdf['linespace_median'])
                ]
            # small_titles['title_rule'] = 5
    #         # joins the DataFrame into one that contains every title
            titles = pd.concat([both_above, zeros, other_titles, small_titles, titre1])
            titles['is_title'] = 1

            return titles

        # def postprocess(titles: pd.DataFrame) -> pd.DataFrame:
        #     """
        #     Removes any results which:
        #     * height is below the means
        #     * interline space before and after that line are equals
        #     """
        #     # post-correcting the results to filter bad lines
        #     post_correction = titles[
        #         titles['line_height'] < titles['height_mean']
        #     ]
        #     post_correction = post_correction[
        #         (post_correction['line_previous_space'] == post_correction['line_following_space']) |
        #         (post_correction['line_previous_space'] == post_correction['line_following_space'] - 5) |
        #         (post_correction['line_following_space'] == post_correction['line_previous_space'] - 5)
        #     ]
        #     # filters and sort titles by index
        #     titles = titles[~titles.isin(post_correction)].dropna()
        #     titles.sort_index(inplace=True)
        #     return titles

        df['is_title'] = 0
        df['title_rule'] = 0

        titles = find_titles(df)
    #     titles = postprocess(titles)
        titles = titles[['is_title', 'title_rule']]
        df.update(titles)
        return df


    def retrieve_subtitles(df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves subtitles from Alto documents
        """
        def find_subtitles(df: pd.DataFrame) -> pd.DataFrame:
            """

            """

            space = df[
                (df['line_previous_space'] > df['linespace_median'])
                | (df['line_following_space'] > df['linespace_median'])
            ]

            space_word = space[
                (space['word_count'] < space['doc_word'])
            ]
            hpos_mean = round(space_word['diff_hpos'].mean()) + 5

            indent_space_word = space_word[
                (space_word['diff_hpos'] >= hpos_mean + 5)
            ]
            indent_space_word['subtitle_rule'] = 1

            subtitles = pd.concat([indent_space_word])
            subtitles['is_subtitle'] = 1

            # df['is_subtitle'] = 0
            # df['subtitle_rule'] = 1
            return subtitles

        def postprocess(subtitles: pd.DataFrame) -> pd.DataFrame:
            """
            :TODO to implement
            """
            pass

        df['is_subtitle'] = 0
        df['subtitle_rule'] = 0
        # subtitles = find_subtitles(df)
        # subtitles = postprocess(subtitles)
        # subtitles = subtitles[['is_subtitle', 'subtitle_rule']]
        # df.update(subtitles)
        return df
        # return pd.merge(df, subtitles, how='left').fillna(0)

    def retrieve_tables(df: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieves tables from Alto and postprocess results

        """
        def find_tables(df: pd.DataFrame) -> pd.DataFrame:
            """
            For now, just takes block that are 'table' type
            TODO: maybe add rules
            """
            df_tables = df[
                df['block_type'] == 'table'
            ]
            df_tables['is_table'] = 1
            df_tables['table_rule'] = 1
            return df_tables

        def postprocess(df: pd.DataFrame) -> pd.DataFrame:
            """
            :TODO Implement
            """
            pass
        df['is_table'] = 0
        df['table_rule'] = 0
        tables = find_tables(df)
        # tables = postprocess(df_tables)
        tables = tables[['is_table', 'table_rule']]
        df.update(tables)
        return df
        # return pd.merge(df, tables, how='left').fillna(0)


    # def retrieve_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    #     """Retrieves and postprocess paragraph from Alto document
    #     :param df: TextLine Feature DataFrame
    #     :type df: pd.DataFrame
    #     :return: Updated TextLine Feature DataFrame
    #     :rtype: pd.DataFrame
    #     """
    #     df['is_firstline'] = 0
    #     df['firstline_rule'] = 0

    #     # selects Text blocks
    #     df = df[
    #         df['block_type'] == 'text'
    #     ]

    #     # Rule 7 to find detect Firstline, based on text indentation
    #     line_with_indentation = df[
    #         (df['line_hpos'] > df['hpos_median'] + 5)
    #         & (df['diff_hpos'] < 105)
    #         & (df['stw_capital'] == 1)

    #         | (df['line_hpos'] > df['hpos_median'] + 5)
    #         & (df['diff_hpos'] < 105)
    #         & (df['stw_digit'] == 1)
    #     ]
    #     line_with_indentation['is_firstline'] = 1
    #     line_with_indentation['firstline_rule'] = 1

    #     # ignores already annotated lines
    #     subdf = df[~df.isin(line_with_indentation)].dropna()

    #     # Rule 8 to determine if previous line is the last line of a paragraph
    #     df['is_lastline'] = 0
    #     df.loc[
    #         (df['line_width'] < df['width_median'])
    #         & (df['word_count'] < df['count_median'])
    #         & (df['line_hpos'] <= df['hpos_median'] + 5),
    #         'is_lastline'
    #     ] = 1
    #     df['prev_lastline'] = df['is_lastline'].shift(1)
    #     df = df.fillna(0)

    #     # Rule 9 to detect beginning of paragraph which are not indented and not preceded by a title
    #     prev_last_line = subdf[
    #         (subdf['prev_lastline'] == 1)
    #         & (subdf['stw_capital'] == 1)
    #         & (subdf['line_following_space'] <= subdf['linespace_median'] + 5)
    #         ]
    #     prev_last_line['is_firstline'] = 1
    #     prev_last_line['firstline_rule'] = 2

    #     subdf = subdf[~subdf.isin(prev_last_line)].dropna()

    #     other = subdf[
    #         # Rule 10 to detect paragraph when Lastline has been missed based on the following linespace
    #         (df['prev_lastline'] == 0)
    #         & (df['stw_capital'] == 1)
    #         & (df['line_previous_space'] == 0)
    #         & (df['line_following_space'] <= df['linespace_median'] + 5)
    #         | (df['prev_lastline'] == 0)
    #         & (df['stw_capital'] == 1)
    #         & (df['line_previous_space'] > df['linespace_median'] + 5)
    #         & (df['line_following_space'] <= df['linespace_median'] + 5)

    #         # Rule 11, detects Firstline if starts with a capital letter and is indented
    #         | (df['prev_lastline'] == 0)
    #         & (df['stw_capital'] == 1)
    #         & (df['line_hpos'] > df['hpos_median'] + 5)
    #         ]

    #     other['is_firstline'] = 1
    #     other['firstline_rule'] = 3

    #     paragraphs = pd.concat([line_with_indentation, prev_last_line, other])
    #     paragraphs = paragraphs[['is_firstline', 'firstline_rule']]
    #     df.update(paragraphs)
    #     return df

    #     paragraphs = find_paragraphs(df)
    #     # paragraphs = postprocess(paragraphs)


    def retrieve_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
        """Retrieves and postprocess paragraph from Alto document

        :param df: TextLine Feature DataFrame
        :type df: pd.DataFrame
        :return: Updated TextLine Feature DataFrame
        :rtype: pd.DataFrame
        """

        def find_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
            # selects Text blocks
            df = df[
                df['block_type'] == 'text'
            ]

            # Rule 7 to find detect Firstline, based on
            # text indentation
            line_with_indentation = df[
                (df['line_hpos'] > df['hpos_median'] + 5)
                & (df['diff_hpos'] < 105)
                & (df['stw_capital'] == 1)

                | (df['line_hpos'] > df['hpos_median'] + 5)
                & (df['diff_hpos'] < 105)
                & (df['stw_digit'] == 1)
            ]
            line_with_indentation['is_firstline'] = 1
            line_with_indentation['firstline_rule'] = 1

            # ignores already annotated lines
            subdf = df[~df.isin(line_with_indentation)].dropna()

            # Rule 8 to determine if previous line is the
            # last line of a paragraph
            df['is_lastline'] = 0
            df.loc[
                (df['line_width'] < df['width_median'])
                & (df['word_count'] < df['count_median'])
                & (df['line_hpos'] <= df['hpos_median'] + 5),
                'is_lastline'
            ] = 1
            df['prev_lastline'] = df['is_lastline'].shift(1)
            df = df.fillna(0)


            # Rule 9 to detect beginning of paragraph which
            # are not indented and not preceded by a title
            prev_last_line = subdf[
                (subdf['prev_lastline'] == 1)
                & (subdf['stw_capital'] == 1)
                & (subdf['line_following_space'] <= subdf['linespace_median'] + 5)
                ]

            prev_last_line['is_firstline'] = 1
            prev_last_line['firstline_rule'] = 2

            subdf = subdf[~subdf.isin(prev_last_line)].dropna()

            other = subdf[
                ## Rule 10 to detect paragraph when Lastline has been missed,
                # based on the following linespace
                (df['prev_lastline'] == 0)
                & (df['stw_capital'] == 1)
                & (df['line_previous_space'] == 0)
                & (df['line_following_space'] <= df['linespace_median'] + 5)

                | (df['prev_lastline'] == 0)
                & (df['stw_capital'] == 1)
                & (df['line_previous_space'] > df['linespace_median'] + 5)
                & (df['line_following_space'] <= df['linespace_median'] + 5)

                # Rule 11, detects Firstline if starts with a capital letter
                # and is indented
                | (df['prev_lastline'] == 0)
                & (df['stw_capital'] == 1)
                & (df['line_hpos'] > df['hpos_median'] + 5)
                ]

            other['is_firstline'] = 1
            other['firstline_rule'] = 3

            paragraphs = pd.concat([line_with_indentation, prev_last_line, other])

            return paragraphs

        df['is_firstline'] = 0
        df['firstline_rule'] = 0
        paragraphs = find_paragraphs(df)
        # paragraphs = postprocess(paragraphs)
        paragraphs = paragraphs[['is_firstline', 'firstline_rule']]
        df.update(paragraphs)
        return df

    def convert_to_labels(df):
        """
        Converts predicted predictions into list of labels

        :param df: Line feature dataframe
        :type df: pd.DataFrame
        :return: List of labels for each line
        :rtype: List[str]
        """
        predicted_label = []
        for title, subtitle, firstline, header, other in zip(df['is_title'].values, df['is_subtitle'].values, df['is_firstline'].values, df['is_header'].values, df['is_other'].values):

        # for title, subtitle, firstline, table, header, other in zip(doc['is_title'].values, doc['is_subtitle'].values, doc['is_firstline'].values,doc['is_table'].values, doc['is_header'].values, doc['is_other'].values):
            if title == 1:
                predicted_label.append('title')
    #         elif subtitle == 1:
    #             predicted_class.append(2)
            elif firstline == 1:
                predicted_label.append('firstline')
            # elif table == 1:
            #     predicted_label.append('table')
            elif header == 1:
                predicted_label.append('header')
            elif other == 1:
                predicted_label.append('other')
            else:
                predicted_label.append('text')
        return predicted_label

    def postprocess(page_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Applies sets of rules to post-process predictions (mainly to remove conflincting labels)

        :param page_metadata: Lines feature dataframe
        :type page_metadata: pd.DataFrame
        :return: Update line feature dataframe
        :rtype: pd.DataFrame
        """

        def postprocess_tables(page_metadata: pd.DataFrame) -> pd.DataFrame:
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

        def postprocess_text(page_metadata: pd.DataFrame) -> pd.DataFrame:
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

            df_text = add_firsttitle(page_metadata)
            df_text = add_firstline(df_text)

            return df_text

        def postprocess_headers(page_metadata: pd.DataFrame) -> pd.DataFrame:
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

            # some doc don't start at page 1
            # thus, the first page is other than 1
            doc_firstpage_num = page_metadata['page'].min()

            # Sets every line in the first 30 lines of the first page
            # as header if they are preceded / followed by a header
            firstpage = page_metadata[
                (page_metadata['page'] == doc_firstpage_num)
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

        def postprocess_titles(page_metadata: pd.DataFrame) -> pd.DataFrame:
            """
            Sets of rules to resolve conflict between blocks that are classified as
            titles and something else at the same time
            """


            # def title_firstline_resolution(df: pd.DataFrame) -> pd.DataFrame:
            #     """
            #     Solves conflict between Title and Firstline 

            #     :param df: TextLine Feature DataFrame
            #     :type df: pd.DataFrame
            #     :return: Updated TextLine Feature DataFrame
            #     :rtype: pd.DataFrame
            #     """
            #     # selects conflicting rows between Title and Firstline
            #     conflict = df[
            #         (df['is_title'] == 1)
            #         & (df['is_firstline'] == 1)
            #         ]
                
            #     # Rule 14 to solve conflict between Title and Firstline
            #     rule1 = conflict[
            #         (conflict['line_following_space'] < conflict['linespace_median'])
            #         & (conflict['capital_prop'] < 15)
            #         ]
            #     rule1['is_firstline'] = 0
            #     rest = conflict[~conflict['line_id'].isin(rule1['line_id'])]

            #     # sets Firstline for all remaining rows
            #     rest['is_title'] = 0

            #     df_titles = pd.concat([rule1, rest])
            #     df.update(df_titles)
            #     return df

            def title_firstline_resolution(page_metadata: pd.DataFrame) -> pd.DataFrame:
                """

                """
                # selects conflicting rows between Title and Firstline
                conflict = page_metadata[
                    (page_metadata['is_title'] == 1)
                    & (page_metadata['is_firstline'] == 1)
                    ]
                
                # add identifier to correcting rule
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
                page_metadata.update(rest)
                return page_metadata

            def title_text_resolution(page_metadata: pd.DataFrame) -> pd.DataFrame:
                """

                """
                df_titles = page_metadata[page_metadata['is_title'] == 1]
                df_titles['is_text'] = 0
                return df_titles

            df_titles = title_firstline_resolution(page_metadata)
            df_titles = title_text_resolution(page_metadata)
            return df_titles

        def postprocess_firstlines(page_metadata: pd.DataFrame) -> pd.DataFrame:
            """
            Sets of rules to resolve conflict between blocks that are classified as
            titles and something else at the same time
            """
            df_firstlines = page_metadata[page_metadata['is_firstline'] == 1]
            df_firstlines['is_text'] = 0
            return df_firstlines

        page_metadata['post_rule'] = 0

        df_headers = postprocess_headers(page_metadata)
        page_metadata.update(df_headers)

        # page_metadata = self.postprocess_headers(page_metadata)
        df_titles = postprocess_titles(page_metadata)
        page_metadata.update(df_titles)

        # df_tables = postprocess_tables(page_metadata)
        # page_metadata.update(df_tables)

        df_firstlines = postprocess_firstlines(page_metadata)
        page_metadata.update(df_firstlines)

        df_text = postprocess_text(page_metadata)
        page_metadata.update(df_text)


        return page_metadata

    # applies the rules
    
    df = retrieve_titles(df)

    # STILL THERE BUT NOT ACTUALLY USED TODO: TO REMOVE
    df = retrieve_subtitles(df)

    df = retrieve_paragraphs(df)

    # df = retrieve_tables(df)

    df = retrieve_headers(df)

    df = retrieve_others(df)

    # page_metadata = retrieve_others(page_metadata)
    # df['is_other'] = 0

    df = postprocess(df)

    df['line_class'] = convert_to_labels(df)

    return df