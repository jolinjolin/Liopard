import numpy as np

__version__ = '0.0.1'


class DataFrame:

    def __init__(self, data):
        """
        A DataFrame holds two dimensional heterogeneous data.
        data: A dictionary of strings mapped to NumPy arrays. The key will become the column name.
        """
        self._check_input_types(data)
        self._check_array_lengths(data)
        self._data = self._convert_unicode_to_object(data)
        self.str = StringMethods(self)
        self._add_docs()

    def _check_input_types(self, data):
        if not isinstance(data, dict):
            raise TypeError('`data` must be a dictionary')
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError('keys of `data` must be strings')
            if not isinstance(value, np.ndarray):
                raise TypeError('values of `data` must be numpy arrays')
            if value.ndim != 1:
                raise ValueError('values of data must be 1-dimensional arrays')

    def _check_array_lengths(self, data):
        for i, value in enumerate(data.values()):
            if i == 0:
                length = len(value)
            elif length != len(value):
                raise ValueError("All arrays must have the same length")

    def _convert_unicode_to_object(self, data):
        new_data = {}
        for key, value in data.items():
            if value.dtype.kind == 'U':
                new_data[key] = value.astype('object')
            else:
                new_data[key] = value
        return new_data

    def __len__(self):
        pass

    @property
    def columns(self):
        pass

    @columns.setter
    def columns(self, columns):
        pass

    @property
    def shape(self):
        pass

    def _repr_html_(self):
        pass

    @property
    def values(self):
        pass

    @property
    def dtypes(self):
        DTYPE_NAME = {'O': 'string', 'i': 'int', 'f': 'float', 'b': 'bool'}
        pass

    def __getitem__(self, item):
        pass

    def _getitem_tuple(self, item):
        pass

    def _ipython_key_completions_(self):
        pass

    def __setitem__(self, key, value):
        pass

    def head(self, n=5):
        pass

    def tail(self, n=5):
        pass

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        pass

    def isna(self):
        pass

    def count(self):
        pass

    def unique(self):
        pass

    def nunique(self):
        pass

    def value_counts(self, normalize=False):
        pass

    def rename(self, columns):
        pass

    def drop(self, columns):
        pass

    def abs(self):
        return self._non_agg(np.abs)

    def cummin(self):
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        return self._non_agg(np.round, decimals=n)

    def copy(self):
        return self._non_agg(np.copy)

    def _non_agg(self, funcname, **kwargs):
        pass

    def diff(self, n=1):
        def func():
            pass
        return self._non_agg(func)

    def pct_change(self, n=1):
        def func():
            pass
        return self._non_agg(func)

    def __add__(self, other):
        return self._oper('__add__', other)

    def __radd__(self, other):
        return self._oper('__radd__', other)

    def __sub__(self, other):
        return self._oper('__sub__', other)

    def __rsub__(self, other):
        return self._oper('__rsub__', other)

    def __mul__(self, other):
        return self._oper('__mul__', other)

    def __rmul__(self, other):
        return self._oper('__rmul__', other)

    def __truediv__(self, other):
        return self._oper('__truediv__', other)

    def __rtruediv__(self, other):
        return self._oper('__rtruediv__', other)

    def __floordiv__(self, other):
        return self._oper('__floordiv__', other)

    def __rfloordiv__(self, other):
        return self._oper('__rfloordiv__', other)

    def __pow__(self, other):
        return self._oper('__pow__', other)

    def __rpow__(self, other):
        return self._oper('__rpow__', other)

    def __gt__(self, other):
        return self._oper('__gt__', other)

    def __lt__(self, other):
        return self._oper('__lt__', other)

    def __ge__(self, other):
        return self._oper('__ge__', other)

    def __le__(self, other):
        return self._oper('__le__', other)

    def __ne__(self, other):
        return self._oper('__ne__', other)

    def __eq__(self, other):
        return self._oper('__eq__', other)

    def _oper(self, op, other):
        pass

    def sort_values(self, by, asc=True):
        pass

    def sample(self, n=None, frac=None, replace=False, seed=None):
        pass

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        pass

    def _add_docs(self):
        agg_names = ['min', 'max', 'mean', 'median', 'sum', 'var',
                     'std', 'any', 'all', 'argmax', 'argmin']
        agg_doc = \
            """
        Find the {} of each column
        
        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)


class StringMethods:

    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = ' '
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding='utf-8', errors='strict'):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        pass


def read_csv(fn):
    pass
