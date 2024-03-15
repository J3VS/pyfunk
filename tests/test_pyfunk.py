import pytest

from pyfunk.pyfunk import *


class NotGetItemClass(object):
    pass


def test_partition():
    assert partition(2, range(10)) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert partition(3, range(10)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    d = {i: i for i in range(10)}
    assert partition(2, d) == [{0: 0, 1: 1}, {2: 2, 3: 3}, {4: 4, 5: 5}, {6: 6, 7: 7}, {8: 8, 9: 9}]
    assert partition(3, d) == [{0: 0, 1: 1, 2: 2}, {3: 3, 4: 4, 5: 5}, {6: 6, 7: 7, 8: 8}, {9: 9}]


def test_map_partitioned():
    assert map_partitioned(2, identity, range(10)) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert map_partitioned(3, identity, range(10)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert map_partitioned(3, sum, range(10)) == [3, 12, 21, 9]

    d = {i: i for i in range(10)}
    assert map_partitioned(2, identity, d) == [{0: 0, 1: 1}, {2: 2, 3: 3}, {4: 4, 5: 5}, {6: 6, 7: 7}, {8: 8, 9: 9}]
    assert map_partitioned(3, identity, d) == [{0: 0, 1: 1, 2: 2}, {3: 3, 4: 4, 5: 5}, {6: 6, 7: 7, 8: 8}, {9: 9}]

    assert map_partitioned(2, identity, None) == []
    assert map_partitioned(2, identity, []) == []
    assert map_partitioned(2, None, range(10)) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_select_keys():
    assert select_keys({'a': 1, 'b': 2}, ['a']) == {'a': 1}
    assert select_keys({'a': 1, 'b': 2}, ['a', 'c']) == {'a': 1}


def test_massoc():
    assert massoc(None, 'x', None) == {'x': None}
    x = {'a': 1}
    assert massoc(x, 'b', 2) == {'a': 1, 'b': 2}
    assert x == {'a': 1, 'b': 2}


def test_assoc():
    assert assoc(None, 'x', None) == {'x': None}
    x = {'a': 1}
    assert assoc(x, 'b', 2) == {'a': 1, 'b': 2}
    assert x == {'a': 1}


def test_mdissoc():
    x = {'a': 1, 'b': 2}
    assert mdissoc(x, 'b') == {'a': 1}
    assert x == {'a': 1}


def test_dissoc():
    x = {'a': 1, 'b': 2}
    assert dissoc(x, 'b') == {'a': 1}
    assert x == {'a': 1, 'b': 2}


def test_mupdate():
    x = {'a': 1, 'b': 2}
    assert mupdate(x, 'b', even) == {'a': 1, 'b': True}
    assert x == {'a': 1, 'b': True}


def test_update():
    x = {'a': 1, 'b': 2}
    assert update(x, 'b', even) == {'a': 1, 'b': True}
    assert update(x, 'b', plus, 2) == {'a': 1, 'b': 4}
    assert x == {'a': 1, 'b': 2}


def rename_keys():
    x = {'a': 1, 'b': 2, 'c': 3}
    assert rename_keys(x, {'a': 'x', 'b': 'y'}) == {'x': 1, 'y': 2, 'c': 3}
    assert x == {'a': 1, 'b': 2, 'c': 3}


def test_keys():
    assert keys(None) is None
    assert keys({}) == []
    assert keys({'a': 1, 'b': 2, 'c': 3}) == ['a', 'b', 'c']


def test_values():
    assert values(None) is None
    assert values({}) == []
    assert values({'a': 1, 'b': 2, 'c': 3}) == [1, 2, 3]


def test_select_keys_as():
    x = {'a': 1, 'b': 2, 'c': 3}
    assert select_rename(x, {'a': 'x', 'b': 'y'}) == {'x': 1, 'y': 2}
    assert x == {'a': 1, 'b': 2, 'c': 3}


class GetItemClass(object):
    def __getitem__(self, x):
        if x == 'a':
            return 1
        elif x == 'b':
            return 2
        elif x == 'c':
            return 3


def test_get_items_as():
    x = GetItemClass()
    assert get_items_as(x, {'a': 'x', 'b': 'y'}) == {'x': 1, 'y': 2}
    y = NotGetItemClass()
    with pytest.raises(TypeError):
        get_items_as(y, {'a': 'x', 'b': 'y'})


def test_filter_none_values():
    x = {'a': 1, 'b': None, 'c': 3, 'd': []}
    assert filter_none_values(x) == {'a': 1, 'c': 3, 'd': []}
    assert x == {'a': 1, 'b': None, 'c': 3, 'd': []}
    y = [1, 2, None, 4, []]
    assert filter_none_values(y) == [1, 2, 4, []]
    assert y == [1, 2, None, 4, []]


def test_filter_falsey_values():
    x = {'a': 1, 'b': None, 'c': 3, 'd': []}
    assert filter_falsey_values(x) == {'a': 1, 'c': 3}
    assert x == {'a': 1, 'b': None, 'c': 3, 'd': []}
    y = [1, 2, None, 4, []]
    assert filter_falsey_values(y) == [1, 2, 4]
    assert y == [1, 2, None, 4, []]


def test_zipmap():
    assert zipmap(['a', 'b', 'c'], [1, 2, 3]) == {'a': 1, 'b': 2, 'c': 3}
    assert zipmap(['a', 'b', 'c', 'd'], [1, 2, 3]) == {'a': 1, 'b': 2, 'c': 3}
    assert zipmap(['a', 'b', 'c'], [1, 2, 3, 4]) == {'a': 1, 'b': 2, 'c': 3}


def test_dedupe():
    assert dedupe([1, 2, 3, 5, 2, 4, 1, 2, 4, 3, 2]) == [1, 2, 3, 4, 5]


def test_dedupe_with_order():
    assert dedupe_with_order([1, 2, 3, 5, 2, 4, 1, 2, 4, 3, 2]) == [1, 2, 3, 5, 4]


def test_identity():
    assert identity(None) is None
    assert identity(1) == 1
    assert identity('Hello') == 'Hello'
    assert identity([1, 2, 3]) == [1, 2, 3]


def test_inc():
    assert inc(1) == 2
    assert inc(-1) == 0


def test_dec():
    assert dec(1) == 0
    assert dec(0) == -1


def test_plus():
    assert plus() == 0
    assert plus(5) == 5
    assert plus(1, -1, 40, -30) == 10


def test_minus():
    assert minus(5) == 5
    assert minus(1, -1, 40, -30) == -8


def test_multiply():
    assert multiply(5) == 5
    assert multiply(5, 1, 2, 3) == 30


def test_divide():
    assert divide(5) == 5
    assert divide(5, 1, 2, 5) == 0.5


def test_apply():
    assert apply(inc, [1]) == 2
    assert apply(multiply, [2, 4]) == 8

    def sum_kws(a=1, b=2):
        return a + b

    assert apply(sum_kws, {'a': 5, 'b': 2}) == 7
    assert apply(sum_kws, {'a': 1, 'b': 4}) == 5


def test_deapply():
    assert deapply(comp(inc, first), 1) == 2


def test_spread_into():
    assert spread_into(concat)([[1], [2], [3]]) == [1, 2, 3]


def test_condense_into():
    assert condense_into(first)(1, 2, 3) == 1


def test_lisp_eval():
    assert lisp_eval(None) is None
    assert lisp_eval(1) == 1
    assert lisp_eval([1, 2, 3]) == [1, 2, 3]
    assert lisp_eval([plus, 2, 3]) == 5


def test_into_list():
    assert into_list() == []
    assert into_list(None) == [None]
    assert into_list(1) == [1]
    assert into_list(1, 2, 5, 3) == [1, 2, 5, 3]


def test_even():
    assert even(4) is True
    assert even(3) is False
    assert even(2) is True
    assert even(1) is False
    assert even(0) is True
    assert even(-1) is False
    assert even(-2) is True
    assert even(-3) is False
    assert even(-4) is True


def test_odd():
    assert odd(4) is False
    assert odd(3) is True
    assert odd(2) is False
    assert odd(1) is True
    assert odd(0) is False
    assert odd(-1) is True
    assert odd(-2) is False
    assert odd(-3) is True
    assert odd(-4) is False


def test_count():
    assert count(None) == 0
    assert count([]) == 0
    assert count(repeat(10, "A")) == 10
    with pytest.raises(TypeError):
        count(123123)


def test_first():
    assert first(None) is None
    assert first([1, 2, 3]) == 1


def test_last():
    assert last(None) is None
    assert last([]) is None
    assert last([1, 2, 3]) == 3


def test_rest():
    assert rest(None) == []
    assert rest([]) == []
    assert rest([1, 2, 3]) == [2, 3]


def test_take():
    with pytest.raises(TypeError):
        take(None, None)
    with pytest.raises(TypeError):
        take(None, [])
    assert take(3, None) == []
    assert take(3, [1]) == [1]
    assert take(-1, [1]) == []
    assert take(3, [1, 2]) == [1, 2]
    assert take(3, [1, 2, 5, 6, 7, 2, 6]) == [1, 2, 5]


def test_drop():
    with pytest.raises(TypeError):
        drop(None, None)
    with pytest.raises(TypeError):
        drop(None, [])
    assert drop(3, None) == []
    assert drop(3, [1]) == []
    assert drop(-1, [1]) == [1]
    assert drop(3, [1, 2]) == []
    assert drop(3, [1, 2, 5, 6, 7, 2, 6]) == [6, 7, 2, 6]


def test_take_last():
    with pytest.raises(TypeError):
        take_last(None, None)
    with pytest.raises(TypeError):
        take_last(None, [])
    assert take_last(3, []) is None
    assert take_last(3, None) is None
    assert take_last(0, [1]) is None
    assert take_last(-1, [1]) is None
    assert take_last(3, [1, 2]) == [1, 2]
    assert take_last(3, [1, 2, 5, 6, 7, 2, 6]) == [7, 2, 6]


def test_drop_last():
    with pytest.raises(TypeError):
        drop_last(None, None)
    with pytest.raises(TypeError):
        drop_last(None, [])
    assert drop_last(3, []) == []
    assert drop_last(3, None) == []
    assert drop_last(0, [1]) == [1]
    assert drop_last(-1, [1]) == [1]
    assert drop_last(3, [1, 2]) == []
    assert drop_last(3, [1, 2, 5]) == []
    assert drop_last(3, [1, 2, 5, 6, 7, 2, 6]) == [1, 2, 5, 6]
    assert drop_last(10, range(100)) == range(0, 90)


def test_butlast():
    assert butlast(None) == []
    assert butlast([]) == []
    assert butlast([1, 2, 3]) == [1, 2]


def test_cons():
    assert cons(1, None) == [1]
    assert cons(1, []) == [1]
    assert cons(1, [2, 3]) == [1, 2, 3]


def test_conj():
    assert conj(None, 1) == [1]
    assert conj([], 1) == [1]
    assert conj([1, 2], 3) == [1, 2, 3]


def test_concat():
    assert concat(None, None) == []
    assert concat(None, [1]) == [1]
    assert concat([1], None) == [1]
    assert concat() == []
    assert concat([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert concat([1, 2], [3, 4], [5, 6], [7, 8]) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_thread_first():
    assert thread_first(1) == 1
    assert thread_first(1, inc, [plus, 5], [minus, 2], dec) == 4


def test_thread_last():
    assert thread_last(1) == 1
    assert thread_last(1, inc, [plus, 5], [minus, 2], dec) == -6


def test_reverse():
    x = [1, 2, 3, 4]
    assert reverse(x) == [4, 3, 2, 1]
    assert x == [1, 2, 3, 4]


def test_juxt():
    assert juxt(inc, dec, even)(1) == [2, 0, False]


def test_comp():
    assert comp(inc, dec)(1) == 1
    assert comp(even, inc)(1) is True


def test_every():
    assert every(even, None) is True
    assert every(even, []) is True
    assert every(even, [2, 4, 6, 8]) is True
    assert every(even, [1, 2, 4, 6, 8]) is False


def test_some():
    assert some(even, None) is False
    assert some(even, []) is False
    assert some(even, [1, 3, 5, 6, 7]) is True
    assert some(even, [1, 3, 5, 7]) is False


def test_frequency():
    assert frequency(even, None) == 0
    assert frequency(even, []) == 0
    assert frequency(even, [1, 2, 3, 4]) == 2


def test_frequencies():
    assert frequencies(None) == {}
    assert frequencies([]) == {}
    assert frequencies(['a', 'b', 'a', 'c', 'b', 'a']) == {'a': 3, 'b': 2, 'c': 1}


def test_distinct_values():
    assert distinct_values(None) is None
    assert distinct_values({}) == []
    assert distinct_values({'a': 1, 'b': 2, 'c': 1}) == [1, 2]


def test_get():
    assert get(None, None) is None
    assert get({}, None) is None
    assert get(None, 'a') is None
    assert get({}, 'a') is None
    assert get({'a': 1}, 'a') == 1
    x = GetItemClass()
    assert get(x, 'a') == 1
    assert get(x, 'd') is None
    y = NotGetItemClass()
    with pytest.raises(TypeError):
        get(y, 'd')
    assert get({'a': 1}, 'b', 4) == 4

    class TestGet:
        def __init__(self, a):
            self.a = a

    assert get(TestGet(1), 'a') == 1
    with pytest.raises(TypeError):
        get(TestGet(1), 'b')
    with pytest.raises(TypeError):
        get(TestGet(1), 'b', 3)


def test_get_either():
    assert get_either(None, None) is None
    assert get_either({}, None) is None
    assert get_either(None, 'a') is None
    assert get_either({}, ['a']) is None
    assert get_either({'a': 1}, ['a', 'b']) == 1
    assert get_either({'b': 1}, ['a', 'b']) == 1
    assert get_either({'c': 1}, ['a', 'b']) is None


def test_getter():
    assert getter(None)(None) is None
    assert getter(None)({}) is None
    assert getter('a')(None) is None
    assert getter('a')({}) is None
    assert getter('a')({'a': 1}) == 1


def test_renamer():
    assert renamer(None)('a') == 'a'
    assert renamer({})('a') == 'a'
    assert renamer({'a': 'b'})('a') == 'b'
    assert renamer({'a': 'b'})('c') == 'c'


def test_stream_to_dict():
    assert stream_to_dict(inc, None) == {}
    assert stream_to_dict(inc, []) == {}
    assert stream_to_dict(inc, [1, 2, 3]) == {1: 2, 2: 3, 3: 4}


def test_stream():
    assert stream(inc, None) == []
    assert stream(inc, []) == []
    assert stream(inc, [1, 2, 3], workers=4) == [2, 3, 4]


def test_stream_eval():
    assert stream_eval(None) == []
    assert stream_eval([]) == []
    assert stream_eval([[inc, 1], [plus, 2, 3]], workers=4) == [2, 5]


def test_stream_dict():
    def fn(k, v):
        return k + v

    assert stream_dict(fn, None) == {}
    assert stream_dict(fn, {}) == {}
    assert stream_dict(fn, {1: 1, 2: 2, 3: 3}) == {1: 2, 2: 4, 3: 6}


def test_stream_values():
    assert stream_values(inc, None) == {}
    assert stream_values(inc, {}) == {}
    assert stream_values(inc, {1: 1, 2: 2, 3: 3}) == {1: 2, 2: 3, 3: 4}


def test_get_in():
    assert get_in(None, None) is None
    assert get_in(None, ['a', 'b', 'c']) is None
    assert get_in({}, ['a', 'b', 'c']) is None
    assert get_in({}, ['a', 'b', 'c'], 1) == 1
    assert get_in({'a': {'b': {'c': 3}}}, None) == {'a': {'b': {'c': 3}}}
    assert get_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'c']) == 3
    assert get_in({'a': {'b': 2}}, ['a', 'c'], 4) == 4


def test_assoc_in():
    assert assoc_in(None, None, None) == {None: None}
    assert assoc_in(None, None, 1) == {None: 1}
    assert assoc_in(None, ['a', 'b', 'c'], None) == {'a': {'b': {'c': None}}}
    assert assoc_in({'a': {'b': {'c': 3}}}, None, None) == {'a': {'b': {'c': 3}}, None: None}
    assert assoc_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'c'], 4) == {'a': {'b': {'c': 4}}}
    assert assoc_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'd'], 4) == {'a': {'b': {'c': 3, 'd': 4}}}
    assert assoc_in({'a': 1}, ['b', 'd'], 4) == {'a': 1, 'b': {'d': 4}}


def test_massoc_in():
    x = {'a': 1}
    massoc_in(x, ['a'], 2)
    assert x == {'a': 2}
    massoc_in(x, ['a', 'b', 'c'], 3)
    assert x == {'a': {'b': {'c': 3}}}
    massoc_in(x, ['a', 'b', 'c'], 4)
    assert x == {'a': {'b': {'c': 4}}}
    massoc_in(x, ['a', 'b', 'd'], 5)
    assert x == {'a': {'b': {'c': 4, 'd': 5}}}


def test_dissoc_in():
    assert dissoc_in(None, None) == {}
    assert dissoc_in(None, ['a', 'b', 'c']) == {}
    assert dissoc_in({'a': {'b': {'c': 3}}}, None) == {'a': {'b': {'c': 3}}}
    assert dissoc_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'c']) == {'a': {'b': {}}}
    assert dissoc_in({'a': {'b': {'c': 3}}}, ['a', 'b']) == {'a': {}}
    assert dissoc_in({'a': {'b': {'c': 3, 'd': 4}}}, ['a', 'b', 'c']) == {'a': {'b': {'d': 4}}}
    assert dissoc_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'd']) == {'a': {'b': {'c': 3}}}


def test_mdissoc_in():
    x = None
    mdissoc_in(x, None)
    assert x is None

    x = None
    mdissoc_in(x, ['a', 'b', 'c'])
    assert x is None

    x = {'a': {'b': {'c': 3}}}
    mdissoc_in(x, None)
    assert x == {'a': {'b': {'c': 3}}}

    x = {'a': {'b': {'c': 3}}}
    mdissoc_in(x, ['a', 'b', 'c'])
    assert x == {'a': {'b': {}}}

    x = {'a': {'b': {'c': 3}}}
    mdissoc_in(x, ['a', 'b'])
    assert x == {'a': {}}

    x = {'a': {'b': {'c': 3, 'd': 4}}}
    mdissoc_in(x, ['a', 'b', 'c'])
    assert x == {'a': {'b': {'d': 4}}}

    x = {'a': {'b': {'c': 3}}}
    mdissoc_in(x, ['a', 'b', 'd'])
    assert x == {'a': {'b': {'c': 3}}}


def test_update_in():
    with pytest.raises(TypeError):
        update_in(None, None, None)
    assert update_in(None, None, identity) == {None: None}
    assert update_in({'a': {'b': {'c': 3}}}, None, identity) == {'a': {'b': {'c': 3}}, None: None}
    assert update_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'c'], inc) == {'a': {'b': {'c': 4}}}
    with pytest.raises(TypeError):
        update_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'd'], inc)
    assert update_in({'a': 1}, ['a'], plus, 5) == {'a': 6}


def test_values_for_key():
    assert values_for_key(None, None) is None
    assert values_for_key(None, []) == []
    assert values_for_key('a', None) is None
    assert values_for_key('a', []) == []
    assert values_for_key('a', [{'a': 1}, {'a': 2}]) == [1, 2]


def test_constantly():
    assert constantly(1)('a', 1, [1, 2, 3]) == 1


def test_case():
    with pytest.raises(NoMatchingClause):
        case('c', {'a': 1, 'b': 2})
    assert case('b', {'a': 1, 'b': 2}) == 2


def test_merge():
    assert merge(None, None) is None
    assert merge(None, {}) == {}
    assert merge({}, None) == {}
    assert merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 3, 'c': 4}
    assert merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4}, {'d': 5, 'e': 6}) == {'a': 1, 'b': 3, 'c': 4, 'd': 5, 'e': 6}
    assert merge(None, {'b': 3, 'c': 4}, {'d': 5, 'e': 6}) == {'b': 3, 'c': 4, 'd': 5, 'e': 6}
    assert merge({'a': 1, 'b': 2}, None, {'d': 5, 'e': 6}) == {'a': 1, 'b': 2, 'd': 5, 'e': 6}
    assert merge(None, None, {'d': 5, 'e': 6}) == {'d': 5, 'e': 6}


def test_merge_with():
    assert merge_with(None, None, None) is None
    assert merge_with(None, {}, None) == {}
    assert merge_with(None, None, {}) == {}
    with pytest.raises(TypeError):
        merge_with(None, {}, {})
    assert merge_with(plus, None, None) is None
    assert merge_with(plus, None, {}) == {}
    assert merge_with(plus, {}, None) == {}
    assert merge_with(plus, {}, {}) == {}
    assert merge_with(plus, {'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
    assert merge_with(plus, {'a': 1, 'b': 3}, {'a': 2, 'b': 5}) == {'a': 3, 'b': 8}


def deep_merge():
    assert deep_merge(None, None) is None
    assert deep_merge(None, {}) == {}
    assert deep_merge({}, None) == {}
    assert deep_merge({}, {}) == {}
    assert deep_merge({'a': 1}, {'a': 2}) == {'a': 2}
    assert deep_merge({'a': 1}, {'b': 2}) == {'a': 1, 'b': 3}
    assert deep_merge({'a': {'b': {'c': 1}, 'd': 3}},
                      {'a': {'b': {'c': 2}, 'e': 4}}) == {'a': {'b': {'c': 2}, 'd': 3, 'e': 4}}


def test_lower():
    assert lower(None) is None
    with pytest.raises(AttributeError):
        lower(1)
    assert lower('Hello') == 'hello'
    assert lower('hello') == 'hello'
    assert lower('heLLo') == 'hello'


def test_inc0():
    assert inc0(None) == 1
    assert inc0(0) == 1
    assert inc0(4) == 5


def test_dec0():
    assert dec0(None) == -1
    assert dec0(0) == -1
    assert dec0(4) == 3


def test_group_by():
    assert group_by(None, None) == {}
    with pytest.raises(TypeError):
        group_by(None, [1, 2, 3])
    assert group_by(identity, None) == {}
    assert group_by(identity, [1, 2, 3, 4]) == {1: [1], 2: [2], 3: [3], 4: [4]}
    assert group_by(even, [1, 2, 3, 4]) == {True: [2, 4], False: [1, 3]}
    assert group_by(getter('a'), [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 1, 'b': 4}]) == \
           {1: [{'a': 1, 'b': 2}, {'a': 1, 'b': 4}], 2: [{'a': 2, 'b': 3}]}


def test_count_grouped():
    assert count_grouped(None, None) == {}
    with pytest.raises(TypeError):
        count_grouped(None, [1, 2, 3])
    assert count_grouped(identity, None) == {}
    assert count_grouped(identity, [1, 2, 3, 4]) == {1: 1, 2: 1, 3: 1, 4: 1}
    assert count_grouped(even, [1, 2, 3, 4]) == {True: 2, False: 2}
    assert count_grouped(getter('a'), [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}, {'a': 1, 'b': 4}]) == {1: 2, 2: 1}


def test_nest():
    assert nest(None, None) == {}
    assert nest(None, []) == {}
    assert nest([], None) == {}
    assert nest(['a'], [{'a': 1, 'b': 2, 'c': 10},
                        {'a': 1, 'b': 2, 'c': 11},
                        {'a': 1, 'b': 3, 'c': 12},
                        {'a': 4, 'b': 2, 'c': 13},
                        {'a': 5, 'b': 6, 'c': 14}]) == \
           {1: [{'a': 1, 'b': 2, 'c': 10}, {'a': 1, 'b': 2, 'c': 11}, {'a': 1, 'b': 3, 'c': 12}],
            4: [{'a': 4, 'b': 2, 'c': 13}],
            5: [{'a': 5, 'b': 6, 'c': 14}]}
    assert nest(['a', 'b'], [{'a': 1, 'b': 2, 'c': 10},
                             {'a': 1, 'b': 2, 'c': 11},
                             {'a': 1, 'b': 3, 'c': 12},
                             {'a': 4, 'b': 2, 'c': 13},
                             {'a': 5, 'b': 6, 'c': 14}]) == \
           {1: {2: [{'a': 1, 'b': 2, 'c': 10}, {'a': 1, 'b': 2, 'c': 11}],
                3: [{'a': 1, 'b': 3, 'c': 12}]},
            4: {2: [{'a': 4, 'b': 2, 'c': 13}]},
            5: {6: [{'a': 5, 'b': 6, 'c': 14}]}}


def test_mapcat():
    with pytest.raises(TypeError):
        mapcat(identity, [1, 2, 3])
    assert mapcat(identity, None) == []
    assert mapcat(identity, []) == []
    assert mapcat(identity, [[1], [2], [3]]) == [1, 2, 3]
    assert mapcat(partial(cons, 1), [[1], [2], [3]]) == [1, 1, 1, 2, 1, 3]


def test_if_value():
    assert if_value(1) == 1
    assert if_value([]) is None
    assert if_value({}) is None
    assert if_value(None) is None
    assert if_value([1, 2, 3]) == [1, 2, 3]


def test_gt():
    assert gt(3, 2, 1) is True
    assert gt(3, 1, 2) is False
    assert gt(1, 2, 3) is False
    assert gt(1, 1, 1) is False


def test_lt():
    assert lt(3, 2, 1) is False
    assert lt(3, 1, 2) is False
    assert lt(1, 2, 3) is True
    assert lt(1, 1, 1) is False


def test_gte():
    assert gte(3, 2, 1) is True
    assert gte(3, 1, 2) is False
    assert gte(1, 2, 3) is False
    assert gte(1, 1, 1) is True


def test_lte():
    assert lte(3, 2, 1) is False
    assert lte(3, 1, 2) is False
    assert lte(1, 2, 3) is True
    assert lte(1, 1, 1) is True


def test_sort_by():
    assert sort_by(None, None) == []
    with pytest.raises(TypeError):
        sort_by(None, [1, 2, 3])
    assert sort_by(identity, None) == []
    assert sort_by(getter('a'), [{'a': 1}, {'a': 3}, {'a': 4}, {'a': 6}, {'a': 3}, {'a': -1}]) == \
           [{'a': -1}, {'a': 1}, {'a': 3}, {'a': 3}, {'a': 4}, {'a': 6}]


def test_sort_ascending():
    assert sort_ascending(None) == []
    assert sort_ascending([]) == []
    with pytest.raises(TypeError):
        sort_ascending(['a', 1])
    assert sort_ascending([1, 5, 7, 3, 3, 3, 5, 7, 1]) == [1, 1, 3, 3, 3, 5, 5, 7, 7]


def test_sort_descending():
    assert sort_descending(None) == []
    assert sort_descending([]) == []
    with pytest.raises(TypeError):
        sort_descending(['a', 1])
    assert sort_descending([1, 5, 7, 3, 3, 3, 5, 7, 1]) == [7, 7, 5, 5, 3, 3, 3, 1, 1]


def test_repeat():
    assert repeat(None, "A") is None
    assert repeat(None, None) is None
    assert repeat(1, None) == [None]
    assert repeat(5, None) == [None, None, None, None, None]
    assert repeat(5, 4) == [4, 4, 4, 4, 4]
    assert repeat(2, "A") == ["A", "A"]


def test_repeatedly():
    assert repeatedly(None, inc0) is None
    assert repeatedly(None, None) is None
    assert repeatedly(1, inc0) == [1]
    assert repeatedly(5, inc0) == [1, 1, 1, 1, 1]

    l = [0]

    def fn():
        last_value = last(l)
        new_value = inc(last_value)
        l.append(new_value)
        return last_value

    assert repeatedly(5, fn) == [0, 1, 2, 3, 4]


def test_average():
    assert average(None) is None
    assert average([]) is None
    assert average([1, 2, 3]) == 2
    with pytest.raises(TypeError):
        average(['a', 1])


def test_find():
    c = [0, 1, 2, 3, 4, 5, 6]
    assert [0, 2, 4, 6] == find(lambda x: x % 2 == 0, c)
    assert [3, 4, 5, 6] == find(lambda x: x >= 3, c)


def test_mapl():
    c = [0, 1, 2, 3, 4, 5, 6]
    assert [1, 2, 3, 4, 5, 6, 7] == mapl(inc, c)


def test_map_indexed():
    assert [0, 2, 4, 6, 8] == map_indexed(lambda i, e: i + e, [0, 1, 2, 3, 4])


def test_map_keys():
    assert {2: 1, 3: 2, 4: 3} == map_keys(inc, {1: 1, 2: 2, 3: 3})


def test_map_value():
    assert {1: 2, 2: 3, 3: 4} == map_values(inc, {1: 1, 2: 2, 3: 3})


def test_split():
    assert split(None) is None
    assert split("hello world") == ["hello", "world"]
    assert split("hello world", "o") == ["hell", " w", "rld"]


def test_join():
    assert join(None) is None
    assert join([]) == ""
    assert join(["hello", "world"]) == "hello world"
    assert join(["hell", " w", "rld"], "o") == "hello world"


def test_first_valid():
    assert first_valid(None, None) is None
    assert first_valid(None, []) is None
    assert first_valid(inc, []) is None
    assert first_valid(inc, ["cant inc", inc, {'a': 1}, 3]) == 4


def test_empty():
    assert empty(None) is True
    assert empty([]) is True
    assert empty({}) is True
    assert empty([1, 2]) is False
    assert empty({'a': 1, 'b': 2}) is False
    assert empty('') is True
    assert empty('foo') is False


def test_most_frequent_values():
    l = [{'text': 'a'}, {'text': 'a'}, {'text': 'a'}, {'text': 'b'}, {'text': 'b'}, {'text': 'c'}]
    f = getter('text')
    assert most_frequent_values(f, l) == l
    assert most_frequent_values(f, l, limit=2) == [{'text': 'a'}, {'text': 'a'}]
    assert most_frequent_values(f, l, limit=4) == [{'text': 'a'}, {'text': 'a'}, {'text': 'a'}, {'text': 'b'}]
    assert most_frequent_values(f, l, threshold=2) == [{'text': 'a'}, {'text': 'a'}, {'text': 'a'}, {'text': 'b'},
                                                       {'text': 'b'}]
    assert most_frequent_values(f, l, threshold=3) == [{'text': 'a'}, {'text': 'a'}, {'text': 'a'}]
    assert most_frequent_values(f, l, threshold=3, limit=4) == [{'text': 'a'}, {'text': 'a'}, {'text': 'a'}]


def test_percentile():
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(6, count(a)) == 60.0

    a = [2, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 8, 8, 9, 9, 10, 11, 11, 12]
    assert percentile(0, count(a)) == 0.0
    assert percentile(5, count(a)) == 25.0
    assert percentile(8, count(a)) == 40.0
    assert percentile(9, count(a)) == 45.0
    assert percentile(16, count(a)) == 80.0
    assert percentile(20, count(a)) == 100.0


def test_string_safe():
    assert string_safe(None) == ''
    assert string_safe(None, default='default string') == 'default string'
    assert string_safe('') == ''
    assert string_safe('  Default String  ') == 'default string'
    assert string_safe('  Default StrinG  ', ignore_case=False) == 'Default StrinG'


def test_multi_method():
    greeting = MultiMethod('language', default="Hey Hey Hey!")

    greeting.method['English'] = "Hello!"
    greeting.method['French'] = "Bonjour!"
    greeting.method['German'] = lambda x: "Auf Wiedersehen!"

    english_map = {'id': 1, 'language': 'English'}
    french_map = {'id': 2, 'language': 'French'}
    german_map = {'id': 3, 'language': 'German'}

    assert greeting(english_map) == 'Hello!'
    assert greeting(french_map) == 'Bonjour!'
    assert greeting(german_map) == 'Auf Wiedersehen!'
    assert greeting({}) == 'Hey Hey Hey!'

    factorial = MultiMethod(identity, default=lambda num: num * factorial(dec(num)))
    factorial.method[0] = 1

    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(3) == 6
    assert factorial(7) == 5040


def test_for_dict():
    assert for_dict(identity, None) is None

    assert for_dict(juxt(condense_into(second), condense_into(first)), {'a': 1, 'b': 2}) == {1: 'a', 2: 'b'}

    def str_combine(*args):
        """
        Combines args into a string, str_combine('a', 1) => 'a1'
        """
        return "".join(map(str, args))

    repeat_twice = partial(repeat, 2)

    assert for_dict(comp(repeat_twice, str_combine), {'a': 1, 'b': 2}) == {'a1': 'a1', 'b2': 'b2'}


def test_map_map():
    assert map_map(identity, identity, None) is None

    assert map_map(first, second, [[1, 2], [3, 4], [5, 6]]) == {1: 2, 3: 4, 5: 6}


def key_by():
    assert key_by('a', None) is None

    assert key_by('a', [{'a': 1, 'b': 2}, {'a': 2, 'b': 3}]) == {1: {'a': 1, 'b': 2}, 2: {'a': 2, 'b': 3}}


def pair_by():
    assert pair_by('a', 'b', None) is None

    assert pair_by('id', 'count', [{'id': 1, 'count': 2}, {'id': 2, 'count': 3}]) == {1: 2, 2: 3}


def test_get_dir():
    assert get_dir(None) is None
    assert get_dir('hello') is None
    assert get_dir('hello/world') == 'hello'
    assert get_dir('/hello/world') == '/hello'
    assert get_dir('hello/world/') == 'hello/world'
    assert get_dir('hello/world/filename.x') == 'hello/world'


def test_get_extension():
    assert get_extension(None) is None
    assert get_extension("/file/path/name.ext") == "ext"
    assert get_extension("/file/path/name") is None
    assert get_extension("/file/path/foo.bar.baz.ext") == "ext"
    assert get_extension("Tic Tac SOW_ Te\u0301ber_Rami\u0301rez.pdf.PdF") == "pdf"
    assert get_extension("foo_slãsh.jpg") == "jpg"


def test_get_filename():
    assert get_filename(None) is None
    assert get_filename("hello") == "hello"
    assert get_filename("/hello/world") == "world"
    assert get_filename("file.ext") == 'file.ext'
    assert get_filename("/hello/world/file.ext") == 'file.ext'


def test_break_path():
    assert break_path(None) is None
    assert break_path("hello") == (None, "hello", None)
    assert break_path("hello/world") == ("hello", "world", None)
    assert break_path("world.txt") == (None, "world", "txt")
    assert break_path("hello/world.txt") == ("hello", "world", "txt")
    assert break_path("/hello/little/world") == ("/hello/little", "world", None)
    assert break_path("/hello/little/world.txt") == ("/hello/little", "world", "txt")


def test_dict_f():
    assert dict_f(None)('a') is None
    assert dict_f({'a': 1, 'b': 2})('a') == 1
    assert dict_f({'a': 1, 'b': 2})('c') is None
    assert dict_f({'a': 1, 'b': 2}, default=3)('c') == 3
    assert mapl(dict_f({'a': 1, 'b': 2, 'c': 3}), ['a', 'b', 'c']) == [1, 2, 3]


def test_lower_keys():
    assert lower_keys(None) is None
    assert lower_keys({}) == {}
    assert lower_keys({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}
    assert lower_keys({'A': 1, 'b': 2}) == {'a': 1, 'b': 2}
    assert lower_keys({'AA': 1, 'b': 2}) == {'aa': 1, 'b': 2}
    assert lower_keys({'AA': 1, 'BB': 2}) == {'aa': 1, 'bb': 2}


def test_value_equals_f():
    assert value_equals_f(None, None)(None) is True
    assert value_equals_f(None, None, default=1)(None) is False
    assert value_equals_f(None, None)({}) is True
    assert value_equals_f(None, None)({None: None}) is True
    assert value_equals_f(None, 1)({None: 1}) is True
    assert value_equals_f(1, None)(None) is True
    assert value_equals_f(1, None)({}) is True
    assert value_equals_f(1, None)({1: None}) is True
    assert value_equals_f("a", 1)({"a": 1}) is True
    assert value_equals_f("a", 1)({"a": 2}) is False


def test_unpack_values():
    x = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}, {'a': 7, 'b': 8, 'c': 9}]
    collected = []
    for b, a, c in unpack_values(x, ['b', 'a', 'c']):
        collected.append(b)
        collected.append(a)
        collected.append(c)
    assert collected == [2, 1, 3, 5, 4, 6, 8, 7, 9]


def test_snaked():
    assert snaked() == ""
    assert snaked(1, 2, 3) == "1_2_3"
    assert snaked(1, 'hello', 2, 'world') == "1_hello_2_world"


def test_kebabed():
    assert kebabed() == ""
    assert kebabed(1, 2, 3) == "1-2-3"
    assert kebabed(1, 'hello', 2, 'world') == "1-hello-2-world"


def test_largest_dict_entries():
    assert largest_dict_entries(None, 1) is None
    assert largest_dict_entries({}, 1) is None
    assert largest_dict_entries({'a': 1}, 0) == []
    assert largest_dict_entries({'a': 1}, 1) == [['a', 1]]
    assert largest_dict_entries({'a': 1, 'b': 3.1, 'c': 2.5}, 1) == [['b', 3.1]]
    assert largest_dict_entries({'a': 1, 'b': 3.1, 'c': 2.5}, 2) == [['b', 3.1], ['c', 2.5]]
    assert largest_dict_entries({'a': 1, 'b': 3.1, 'c': 2.5}, 3) == [['b', 3.1], ['c', 2.5], ['a', 1]]


def test_max_dict_entry():
    assert max_dict_entry(None) is None
    assert max_dict_entry({}) is None
    assert max_dict_entry({'a': 1}) == ['a', 1]
    assert max_dict_entry({'a': 1, 'b': 3.1, 'c': 2.5}) == ['b', 3.1]


def test_key_values_in():
    assert key_values_in(None, None) is None
    assert key_values_in('v', None) is None
    assert key_values_in(None, []) == {}
    assert key_values_in('v', []) == {}
    assert key_values_in('v', [
        {'a': 1, 'v': [1, 2, 3]},
        {'a': 2, 'v': [2, 3]},
        {'a': 3, 'v': [1, 3]}
    ]) == {1: [{'a': 1, 'v': [1, 2, 3]}, {'a': 3, 'v': [1, 3]}],
           2: [{'a': 1, 'v': [1, 2, 3]}, {'a': 2, 'v': [2, 3]}],
           3: [{'a': 1, 'v': [1, 2, 3]}, {'a': 2, 'v': [2, 3]}, {'a': 3, 'v': [1, 3]}]}


def test_dict_average():
    assert dict_values_proportion(None) is None
    assert dict_values_proportion([]) == {}
    assert dict_values_proportion([{'a': 1, 'b': 1}, {'b': 1, 'c': 1}]) == {'a': 0.25, 'b': 0.5, 'c': 0.25}
    assert dict_values_proportion([{'a': None, 'b': 1, 'd': 1}, {'a': 0, 'c': 1, 'd': 1}]) == {'a': 0.0, 'b': 0.25, 'c': 0.25, 'd': 0.5}


def test_flip():
    assert flip(None) is None
    assert flip([]) == []
    with pytest.raises(ValueError):
        flip([[1, 2], [3, 4, 5]])
    assert flip([[1], [2], [3]]) == [[1, 2, 3]]
    assert flip([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]]


def test_pad():
    assert pad(0, None) == []
    assert pad(3, None) == [None, None, None]
    assert pad(3, [1, 2, 3]) == [1, 2, 3]
    assert pad(5, [1, 2, 3], default=5) == [1, 2, 3, 5, 5]


def test_merge_lists_with():
    assert merge_lists_with(identity, None) is None
    assert merge_lists_with(identity, []) == []
    assert merge_lists_with(identity, [{'a': 1}, {'a': 1}]) == [{'a': 1}]
    assert merge_lists_with(identity, [{'a': 1}], [{'b': 1}]) == [{'a': 1}, {'b': 1}]
    assert merge_lists_with(getter('a'), [{'a': 1}], [{'a': 1}]) == [{'a': 1}]
    assert merge_lists_with(comp(sum, values), [{'a': 1, 'b': 2}], [{'a': 2, 'b': 1}]) == [{'a': 2, 'b': 1}]


def test_millis_from_string():
    assert millis_from_string(None) is None
    with pytest.raises(TypeError):
        assert millis_from_string(4)
    assert millis_from_string('1ms') == 1
    assert millis_from_string('12s') == 12000
    assert millis_from_string('4m') == 240000
    assert millis_from_string('55h') == 198000000
    assert millis_from_string('9d') == 777600000
    assert millis_from_string('12w') == 7257600000
    assert millis_from_string('4w 10d 9h 32m 46s 1ms') == 3317566001
    with pytest.raises(ValueError):
        assert millis_from_string('3y')
    with pytest.raises(ValueError):
        assert millis_from_string('5y 4w 10d 9h 32m 46s 1ms')


def test_remove_all():
    assert remove_all('a', None) is None
    assert remove_all('a', []) == []
    assert remove_all(1, [2, 3, 4, 5]) == [2, 3, 4, 5]
    assert remove_all(1, [2, 3, 1, 4, 1, 5]) == [2, 3, 4, 5]
    assert remove_all('a', ['m', 'a', 'k', 'e']) == ['m', 'k', 'e']


def test_all_none():
    assert all_none() is True
    assert all_none(None) is True
    assert all_none(None, 1) is False
    assert all_none(1) is False
    assert all_none(1, "thing") is False
    assert all_none(None, None) is True


def test_some_none():
    assert some_none() is False
    assert some_none(None) is True
    assert some_none(None, 1) is True
    assert some_none(1) is False
    assert some_none(1, "thing") is False
    assert some_none(None, None) is True


def test_no_none():
    assert no_none() is True
    assert no_none(None) is False
    assert no_none(None, 1) is False
    assert no_none(1) is True
    assert no_none(1, "thing") is True
    assert no_none(None, None) is False


def test_not_all_none():
    assert not_all_none() is False
    assert not_all_none(None) is False
    assert not_all_none(None, 1) is True
    assert not_all_none(1) is True
    assert not_all_none(1, "thing") is True
    assert not_all_none(None, None) is False


def test_format_number():
    assert format_number(None) is None
    assert format_number("dsfdfds") is None
    assert format_number(1) == '1'
    assert format_number('1') == '1'
    assert format_number(1234567890) == '1,234,567,890'
    assert format_number('1234567890') == '1,234,567,890'
