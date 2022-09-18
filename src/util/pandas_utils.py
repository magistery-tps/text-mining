def group_by_count(df, group_cols, count_col, ascending=False, count_col_name='count'):
    result = df.groupby(group_cols)[count_col] \
        .count() \
        .reset_index(name=count_col) \
        .sort_values(by=[count_col], ascending=ascending)

    return result.rename(columns={count_col: count_col_name if count_col_name else count_col })