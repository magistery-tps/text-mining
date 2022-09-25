

class FeatureColumnBuilder:
    def __init__(self, ds, source_column, target_column, tokens_count_col, max_tokens = 150):
        self.__target_column     = target_column
        self.__max_tokens        = max_tokens
        self.__ds                = ds.copy()[ds[tokens_count_col] < self.__max_tokens]
        self.__ds[self.__target_column] = f'{source_column}: ' + self.__ds[source_column]        
        self.__tokens_count_col  = tokens_count_col

    def __append_str(self, name, series):
        def get_values(v):
            return '' if v is None else f'. {name}: {v}'
        self.__append_raw(name, series.apply(get_values))

    def __append_str_list(self, name, series, max_tokens):
        def get_values(v):
            if v is None:
                return ''
            values = v.replace('[', '').replace(']', '').strip().split(',')[:max_tokens]
            return f'. {name}: ' + ','.join(values)

        self.__append_raw(name, series.apply(get_values))

    def __append_list(self, name, series, max_tokens):
        def get_values(v):
            if v is None or len(v) == 0:
                return ''
            values = v.tolist()[:max_tokens]
            return f'. {name}: ' + ', '.join(values)
        self.__append_raw(name, series.apply(get_values))

    def __append_raw(self, name, series):
        self.__ds[self.__target_column]  = self.__ds[self.__target_column] + series 

    def __append(self, col, col_type, max_tokens):
        if 'str' == col_type:
            self.__append_str(col, self.__ds[col])
        elif 'list' == col_type:
            self.__append_list(col, self.__ds[col], max_tokens)
        elif 'str_list' == col_type:
            self.__append_str_list(col, self.__ds[col], max_tokens)

    def __update_count(self):
        self.__ds[self.__tokens_count_col] = self.__ds[self.__target_column].apply(lambda x: len(x.split(' ')))

    def build(self, columns = {}, max_tokens = 10):
        [self.__append(col, col_type, max_tokens) for col, col_type in columns.items()] 
        self.__update_count()  
        return self.__ds