import logging


class SourceColumn:
    def __init__(self, name, type, max_tokens):
        self.name = name
        self.type = type
        self.max_tokens = max_tokens

class FeatureColumnBuilder:
    def __init__(self, target_column, tokens_count = True):
        self.target       = target_column
        self.sources      = []
        self.tokens_count = target_column


    def add(self, name, type = 'str', max_tokens=None):
        self.sources.append(SourceColumn(name, type, max_tokens))
        return self


    def __append_str(self, ds, name, series):
        get_values = lambda v: '' if v is None else f'{name.capitalize()}: {v}. '
        self.__append_raw(ds, series.apply(get_values))


    def __append_str_list(self, ds, name, series, max_tokens):
        def get_values(v):
            try:
                if v is None:
                    return ''
                values = v.replace('[', '').replace(']', '').strip().split(',')
                if max_tokens > 0:
                    values = values[:max_tokens]

                values = [e.replace("'", '').replace('"', '') for e in values]

                return f'{name.capitalize()}: {", ".join(values)}. '
            except Exception as error:
                logging.info(f'column: {name}. Error: {error}')
                exit(1)

        self.__append_raw(ds, series.apply(get_values))


    def __append_list(self, ds, name, series, max_tokens):
        def get_values(v):
            try:
                if v is None or len(v) == 0:
                    return ''
                values = v.tolist()

                if max_tokens > 0:
                    values = values[:max_tokens]

                values = [e.replace("'", '').replace('"', '') for e in values]

                return f'{name.capitalize()}: {", ".join(values)}. '
            except Exception as error:
                logging.info(f'column: {name}. Error: {error}')

        self.__append_raw(ds, series.apply(get_values))


    def __append_raw(self, ds, series):
        if self.target in ds.columns:
            ds[self.target] = ds[self.target] + series
        else:
            ds[self.target] = series


    def __append(self, ds, name, type, max_tokens):
        if 'str' == type:
            self.__append_str(ds, name, ds[name])
        elif 'list' == type:
            self.__append_list(ds, name, ds[name], max_tokens)
        elif 'str_list' == type:
            self.__append_str_list(ds, name, ds[name], max_tokens)


    def __update_count(self, ds):
        if self.tokens_count:
            ds['tokens_count'] = ds[self.target].apply(lambda x: len(x.split(' ')))


    def __call__(self, ds):
        ds = ds.copy()
        [self.__append(ds, source.name, source.type, source.max_tokens) for source in self.sources]
        self.__update_count(ds)
        return ds
