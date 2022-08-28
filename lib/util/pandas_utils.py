def group_by_count(df, group_cols, count_col, ascending=False):
    return df.groupby(group_cols)[count_col] \
        .count() \
        .reset_index(name=count_col) \
        .sort_values(by=[count_col], ascending=ascending)