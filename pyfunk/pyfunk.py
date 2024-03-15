import base64
import collections
import datetime
import json
import os
import re
import time
from collections import abc
from concurrent.futures import ThreadPoolExecutor
from functools import cmp_to_key, partial, reduce
from itertools import chain, islice
from re import match
from typing import Any, Callable, List, Optional

from math import floor

MAX_WORKERS = 50


def partition_generator(l, size: int):
    it = iter(l)
    for i in range(0, len(l), size):
        if type(l) is dict:
            yield {k: l[k] for k in islice(it, size)}
        else:
            yield list(l[i:i + size])


def partition(size: int, col):
    """
    Partitions a list into batches of the proposed size
    :param size: the size of the batches
    :param col: the list to partition
    :return: a list of lists each of size `count`
    partition(2, [1, 2, 3, 4, 5]) => [[1, 2], [3, 4], [5]]
    """
    return list(partition_generator(col, size))


def map_partitioned(size: int, f: Callable, col) -> List[Any]:
    if not col:
        return []

    if not f:
        return partition(size, col)

    return mapl(f, partition(size, col))


def keys(d):
    """
    Returns the keys for the dictionary
    There is no guarantee of ordering
    :param d: the dictionary
    :return: the keys
    """
    if d is None:
        return None
    if isinstance(d, abc.ItemsView):
        return [e for e, _ in d]
    return list(d.keys())


def values(d):
    """
    Returns the values for the dictionary
    There is no guarantee of ordering
    :param d: the dictionary
    :return: the values
    """
    if d is None:
        return None
    if isinstance(d, abc.ItemsView):
        return [e for _, e in d]
    return list(d.values())


def as_dict(o):
    """
    Returns the object as a dict if it can be
    :param o: the object
    :return: the dict
    """
    if isinstance(o, dict):
        return o
    if hasattr(o, '__dict__'):
        return o.__dict__
    return None


def select_keys(d, ks):
    """
    Returns a dict containing only those entries in `d` whose key is in `ks`
    :param d: the dictionary
    :param ks: the keys
    :return: a new dictionary with the keys in `ks` and the corresponding values from `d`
    select_keys({'a': 1, 'b': 2}, ['a']) => {'a': 1}
    """
    retval = {}
    d = as_dict(d)
    for k in ks:
        try:
            retval[k] = d[k]
        except:
            pass
    return retval


def select_values(d, ks):
    """
    Returns a list of values in the dict `d` for the `ks` in the list
    Order is preservec
    :param d:
    :param ks:
    :return:
    """
    if d is None:
        return None
    if not isinstance(d, dict):
        raise TypeError("Ids to Keys must be a dictionary")

    if not (isinstance(ks, list) or isinstance(ks, tuple)):
        raise TypeError("Ids must be a list")

    return [get(d, k) for k in ks]


def massoc(d, *args):
    """
    Associates the arguments in the dictionary, this is a mutable operation
    :param d: the dictionary
    :param kwargs: the args to associate
    :return: the provided dictionary with the assoc'd args
    assoc({'a': 1}, b=2, c=3) => {'a': 1, 'b': 2, 'c': 3}
    """
    if d is None:
        d = {}
    for k, v in partition(2, args):
        d[k] = v
    return d


def assoc(d, *args):
    """
    Returns a new dictionary created by associating the kwargs to the provided
    dictionary
    :param d: the original dictionary
    :param args: the args to associate
    :return: the new dictionary with the args assoc'd
    assoc({'a': 1}, b=2, c=3) => {'a': 1, 'b': 2, 'c': 3}
    """
    if d is None:
        dc = {}
    else:
        dc = d.copy()
    return massoc(dc, *args)


def mdissoc(d, *args):
    """
    Disassociates args from the provided dictionary, this is a mutable operation
    :param d: the dictionary
    :param args: the args to dissoc
    :return: the dictionary with the args dissoc'd
    dissoc({'a': 1, 'b': 2}, 'b') => {'a': 1}
    """
    for arg in args:
        try:
            d.pop(arg)
        except:
            pass
    return d


def dissoc(d, *args):
    """
    Returns a new dictionary created by disassociating the args from the provided
    dictionary
    :param d: the original dictionary
    :param args: the args to dissoc
    :return: the new dictionary
    dissoc({'a': 1, 'b': 2}, 'b') => {'a': 1}
    """
    dc = d.copy()
    return mdissoc(dc, *args)


def mupdate(d, k, f, *args):
    """
    Updates a dictionary `d` a key `k` with function `f`.
    The function takes the original value.
    This is a mutable operation
    :param d: the dictionary
    :param k: the key of the value to update
    :param f: the update function
    :return: the updated dictionary
    update({'a': 1}, 'a', inc) => {'a': 2}
    """
    d[k] = apply(f, cons(d.get(k), args))
    return d


def append_value(m, k, lv):
    """
    Appends a value to dictionary `d` with key `k`.
    This is a mutable operation
    :param d: the dictionary
    :param k: the key of the value to update

    :return: the updated dictionary
    append_value({'k': [1]}, 2) => {'k': [1, 2]}
    """
    mupdate(m, k, conj, lv)


def update(d, k, f, *args):
    """
    Returns a new dictionary based on the provided dictionary `d`, with
    the key `k` value updated with the function `f`
    :param d: the dictionary
    :param k: the key of the value to update
    :param f: the update function
    :return: a new dictionary
    update({'a': 1}, 'a', inc) => {'a': 2}
    """
    dc = d.copy()
    return mupdate(dc, k, f, *args)


def rename_keys(d, keymap):
    """
    Returns a new dictionary based on the dictionary provided, with renamed
    keys based on the keymap
    :param d: the original dictionary
    :param keymap: the map of old key to new key
    :return: the new dictionary with renamed keys
    rename_keys({'a': 1, 'b': 2, 'c': 3}, {'a': 'x', 'b': 'y'})
    => {'x': 1, 'y': 2, 'c': 3}
    """
    retval = {}
    for k in d.keys():
        new_k = keymap.get(k, k)
        retval[new_k] = d[k]
    return retval


def select_rename(d, keymap):
    """
    Returns a new dictionary based on the provided dictionary, but with the
    keys renamed based on the keymap
    :param d: the original dictionary
    :param keymap: the map of old key to new key
    :return: the new dictionary
    rename_select({'a': 1, 'b': 2, 'c': 3}, {'a': 'x', 'b': 'y'})
    => {'x': 1, 'y': 2}
    """
    return thread_first(d, [rename_keys, keymap],
                        [select_keys, keymap.values()])


def get(obj, k, default=None):
    """
    Function to return value from obj `d` by key `k`
    :param k: the key
    :param obj: the object
    :param default: a default, defaults to NONE
    :return: the value, if the object doesn't implement __getitem__ or has no
    value, None will be returned
    get({'a': 1}, 'a') => 1
    get(anObject, 'key') => None
    """
    result = default
    if obj:
        if hasattr(obj, '__getitem__'):
            try:
                result = obj[k]
            except KeyError:
                pass
        elif isinstance(k, str) and hasattr(obj, k):
            result = getattr(obj, k)
        else:
            raise TypeError("No way to get k from obj")
    return result


def agg_col_by_key(col, group_by_key, value_key, func=lambda x, y: x + y, default_val=0):
    '''
    Aggregates collection of data by a specifc key.
    :param col:
    :param group_by_key:
    :param value_key:
    :return:
    agg_col_by_key([{'k': 1, 'v': 1}, {'k': 1, 'v': 3}], 'k', 'v') => [{'k': 1, 'v': 4}]
    '''
    dict_data = {}
    for item in col:
        dict_data.setdefault(item[group_by_key], default_val)
        dict_data[item[group_by_key]] = func(dict_data[item[group_by_key]], item[value_key])

    return [{group_by_key: k, value_key: v} for k, v in dict_data.items()]


def get_either(obj, ks, default=None):
    """
    Function to return first not-None value from obj `d` by keys `ks`
    :param obj: the object
    :param ks: the keys
    :param default: a default, defaults to NONE
    :return: the value, if the object doesn't implement __getitem__ or has no
    value, None will be returned
    get({'b': 1}, ['a', 'b']) => 1
    get({'a': 1}, ['a', 'b']) => 1
    get(anObject, ['key']) => None
    """
    if obj:
        for k in ks:
            try:
                result = obj[k]
                if result is not None:
                    return result
            except KeyError:
                pass
    return default


def get_items_as(obj, keymap):
    """
    The same as select_rename, but has no requirement that the provided
    object is a dictionary, or supports .keys()
    :param obj: the original object, this must support __getitem__
    :param keymap: the map of old key to new key
    :return: the new dictionary
    get_items_as({'a': 1, 'b': 2, 'c': 3}, {'a': 'x', 'b': 'y'})
    => {'x': 1, 'y': 2}
    """
    retval = {}
    ks = keymap.keys()
    for k in ks:
        v = get(obj, k)
        if v:
            retval[keymap[k]] = v
    return retval


def filter_none_values(d):
    """
    Accepts a list or dict and removes None values
    :param d: the dictionary or list
    :return: the dictionary or list without None values
    filter_none_values({'a': 1, 'b': None}) => {'a': 1}
    filter_none_values([1, 2, None, [], 3]) => [1, 2, [], 3]
    """
    if type(d) is dict:
        return dict((a, b) for (a, b) in d.items() if b is not None)
    if type(d) is list or type(d) is tuple:
        return list(filter(lambda x: x is not None, d))


def filter_falsey_values(d):
    """
    Accepts a list or dict and removes falsey values
    :param d: the dictionary or list
    :return: the dictionary or list without None values
    filter_falsey_values({'a': 1, 'b': None}) => {'a': 1}
    filter_falsey_values([1, 2, None, [], 3]) => [1, 2, 3]
    """
    if type(d) is dict:
        return dict((a, b) for (a, b) in d.items() if b)
    if type(d) is list or type(d) is tuple:
        return list(filter(lambda x: x, d))


def zipmap(ks, vals):
    """
    Returns a dict with the keys mapped to the corresponding vals.
    :param ks: the keys
    :param vals: the vals
    :return: the dictionary with keys and vals
    zipmap(['a', 'b', 'c'], [1, 2, 3]) => {'a': 1, 'b': 2, 'c': 3}
    """
    return dict(zip(ks, vals))


def dedupe(l):
    """
    Removes duplicate values from the list
    NOTE: ORDER IS NOT GUARANTEED
    :param l: the list
    :return: the deduplicated list
    dedupe([1, 2, 1, 3, 2, 4]) => [1, 2, 3, 4]
    """
    if l is None:
        return None
    return list(set(l))


def dedupe_with_order(l):
    """
    Removes duplicate values from the list
    NOTE: ORDER IS GUARANTEED
    :param l: the list
    :return: the deduplicated list
    dedupe([1, 2, 1, 3, 2, 4]) => [1, 2, 3, 4]
    """
    if l is None:
        return None

    def reducer(acc, x):
        if x not in acc:
            acc.append(x)
        return acc

    return reduce(reducer, l, [])


def remove(pred, l):
    """
    Removes values from l that satisfy a predicate
    :param pred: the predicate
    :param l: the list
    :return: the list with the elements removed
    """
    return [x for x in l if not pred(x)]


def identity(x):
    """
    The identity function, simply returns the value
    :param x: any value
    :return: x
    identity(1) => 1
    """
    return x


def inc(x):
    """
    Increments a number
    :param x: a number
    :return: the incremented number
    inc(1) => 2
    """
    return x + 1


def dec(x):
    """
    Decrements a number
    :param x: a number
    :return: the decremented number
    dec(1) => 0
    """
    return x - 1


def plus(*args):
    """
    Adds together an arbitrary arglist of numbers
    :param args: the args to add
    :return: the result
    plus(1, 2, 3, 4) => 10
    """
    if not args:
        return 0
    return reduce(lambda acc, y: acc + y, rest(args), first(args))


def minus(x, *args):
    """
    Subtracts an arbitrary arglist of numbers, at least one number
    must be supplied
    :param x: the first number
    :param args: the args to subtract
    :return: the result
    plus(5, 1, 2) => 2
    """
    return reduce(lambda acc, y: acc - y, args, x)


def multiply(x, *args):
    """
    Multiplies together an arbitrary arglist of numbers, at least one number
    must be supplied
    :param x: the first number
    :param args: the args to multiply
    :return: the result
    multiply(3, 3, 5) => 45
    """
    return reduce(lambda acc, y: acc * y, args, x)


def divide(x, *args):
    """
    Divides an arbitrary arglist of numbers, at least one number
    must be supplied
    :param x: the first number
    :param args: the args to divide
    :return: the result as a float
    divide(3, 3, 5) => 0.2
    """
    return reduce(lambda acc, y: acc / float(y), args, float(x))


def apply(f, args):
    """
    Applys the function `f` to the list of `args`
    If the args is a list, they are applied as arguments,
    if it's a dict, they're applied as kwargs
    :param f: the function to apply
    :param args: the list of args
    :return: the result of applying the function to the args
    apply(plus, [2, 2, 4]) => 8
    """
    if type(args) is dict:
        return f(**args)
    else:
        return f(*args)


def deapply(f, *args):
    """
    Returns a function that when passed variadic arguments, will
    pass those args to the function as a list
    :param f: the function
    :param args: the variadic args
    :return: a function that receives those args as a list
    """
    return f(args)


def lisp_eval(l):
    """
    Lispy functional evaluation
    :param l: a list where the first value is interpreted to be a function, and the other
    value are the function args
    :return: the result of calling the first value on the rest
    """
    if type(l) is list and callable(first(l)):
        return apply(first(l), rest(l))
    else:
        return l


def into_list(*args):
    """
    Returns the args as a list
    :param args: the args
    :return: the args as a list
    """
    return list(args)


def list_to_map(l):
    """
    Converts a list to a map
    :param l: the list
    :return: a map
    list_to_map([1, 2, 3, 4]) => {1: 2, 3: 4}
    """
    if l is None:
        return None
    return {k: v for k, v in partition(2, l)}


def even(x):
    """
    Returns whether a number is even
    :param x: a number
    :return: whether the number is even
    even(1) => False
    """
    return x % 2 == 0


def odd(x):
    """
    Returns whether a number is odd
    :param x: a number
    :return: whether the number is odd
    odd(1) => True
    """
    return x % 2 == 1


def count(l):
    """
    Returns the number of elements in a list, 0 if the arg is None
    :param l: a list
    :return: the count of the elements
    """
    return 0 if l is None else len(l)


def first(x):
    """
    Returns the first value in a list
    :param x: the list
    :return: the first value in the list
    first([1, 2, 3]) => 1
    """
    if x and len(x) > 0:
        return x[0]
    else:
        return None


def second(x):
    """
    Returns the second value in a list
    :param x: the list
    :return: the second value in the list
    second"([1, 2, 3]) => 2
    """
    if len(x) > 1:
        return x[1]
    else:
        return None


def nth(x, n):
    """
    Returns the nth value in a list
    :param x: the list
    :return: the nth value in the list
    """
    if len(x) > n:
        return x[n]
    else:
        return None


def last(x):
    """
    Returns the last value in a list
    :param x: the list
    :return: the last value in the list
    last([1, 2, 3]) => 3
    """
    if x:
        return x[-1]
    else:
        return None


def rest(x):
    """
    Returns all but the first value in the list
    :param x: the list
    :return: all but the first value in the list
    rest([1, 2, 3]) => [2, 3]
    """
    if x:
        return x[1:]
    else:
        return []


def take(n, col):
    """
    Takes the first n elements from a collection
    :param n: the number of elements to take
    :param col: the collection
    :return: the first n elements of the collection
    """
    if n is None:
        raise TypeError("Count cannot be None")
    if col:
        return col[:n]
    else:
        return []


def drop(n, col):
    """
    Drops the first n elements from a collection
    :param n: the number of elements to drop
    :param col: the collection
    :return: all but the first n elements of the collection
    """
    if n is None:
        raise TypeError("Count cannot be None")
    if col:
        return col[n:]
    else:
        return []


def take_last(n, col):
    """
    Takes the last n elements from a collection
    :param n: the number of elements to take
    :param col: the collection
    :return: the last n elements of the collection
    """
    if n is None:
        raise TypeError("Count cannot be None")
    if col and n > 0:
        i = max(len(col) - n, 0)
        return col[i:]
    else:
        return None


def drop_last(n, col):
    """
    Drops the last n elements from a collection
    :param n: the number of elements to drop
    :param col: the collection
    :return: all but the last n elements of the collection
    """
    if n is None:
        raise TypeError("Count cannot be None")
    if n > count(col):
        return []
    if n < 1:
        return col
    if col:
        i = count(col) - n
        return col[:i]
    else:
        return None


def butlast(x):
    """
    Returns all but the last value in the list
    :param x: the list
    :return: all but the last value in the list
    butlast([1, 2, 3]) => [1, 2]
    """
    if x:
        return x[:-1]
    else:
        return []


def cons(v, l):
    """
    Appends the value `v` to the list `l` at the start
    :param l: the list
    :param v: the value
    :return: the new list
    cons([2, 3, 4], 1) => [1, 2, 3, 4]
    """
    if l:
        return [v] + list(l)
    else:
        return [v]


def conj(l, v):
    """
    Appends the value `v` to the list `l` at the end
    :param l: the list
    :param v: the value
    :return: the new list
    conj([1, 2, 3], 4) => [1, 2, 3, 4]
    """
    if l:
        return list(l) + [v]
    else:
        return [v]


def concat(*args):
    """
    Concatenates an arbitrary number of lists
    :param args: the lists to concatenate
    :return: the concatenated lists
    concat([1, 2], [3, 4]) => [1, 2, 3, 4]
    """
    return reduce(lambda acc, li: acc + li, filter_none_values(args), [])


def apply_with_first(x, v):
    if type(x) is list:
        return lisp_eval(concat([first(x), v], rest(x)))
    else:
        return x(v)


def thread_first(val, *args):
    """
    Threads a value through a series of functions or forms.
    The form can be either a function, or a vector containing
    a function and a number of args to be applied after the threaded
    value
    :param val: the initial value
    :param args: the functions/forms to thread the value through
    :return: the resulting value
    thread_first(1, inc, [plus, 5], [minus, 2]) => 5
    """
    return reduce(lambda acc, y: apply_with_first(y, acc), args, val)


def apply_with_last(x, v):
    if type(x) is list:
        return lisp_eval(conj(x, v))
    else:
        return x(v)


def thread_last(val, *args):
    """
    Threads a value through a series of functions or forms.
    The form can be either a function, or a vector containing
    a function and a number of args to be applied before the threaded
    value
    :param val: the initial value
    :param args: the functions/forms to thread the value through
    :return: the resulting value
    thread_last(1, inc, [plus, 5], [minus, 2]) => -5
    """
    return reduce(lambda acc, y: apply_with_last(y, acc), args, val)


def reverse(l):
    """
    Returns the reverse of the list `l`
    :param l: the list to reverse
    :return: the reversed list
    reverse([1, 2, 3, 4]) => [4, 3, 2, 1]
    """
    return list(reversed(l))


def juxt(arg, *args):
    """
    Takes an arbitrary number of functions, and returns a function
    that when applied to a value, returns a list of each function
    applied to the value
    :param arg: the initial function (at least one must be supplied)
    :param args: the functions
    :return: the generated function
    juxt(inc, dec, even)(1) => [2, 0, False]
    """

    def return_f(*inner_args, **inner_kwargs):
        return list(map(lambda f: f(*inner_args, **inner_kwargs), cons(arg, args)))

    return return_f


def comp(*args):
    """
    Takes an arbitrary number of functions and returns the composition
    of those functions
    :param args: function
    :return: the composition of the provided functions
    comp(even, inc)(1) => True

    """

    def return_f(*inner_args):
        initial = apply(last(args), inner_args)
        rest_of_functions = reverse(butlast(args))
        return reduce(lambda acc, f: f(acc), rest_of_functions, initial)

    return return_f


def spread_into(f):
    """
    Does a partial of the application of the function provided
    :param f: function
    :return: a function that when provided args, will apply the args
    as a list to the function
    """
    return partial(apply, f)


def condense_into(f):
    """
    Does a partial of the deapplication of the function provided
    :param f: function
    :return: a function that when provided args, will apply the args
    as a list to the function
    """
    return partial(deapply, f)


def every(pred, col):
    """
    Returns whether the predicate is true for all items in
    the collection
    :param pred: the predicate
    :param col: the collection
    :return: whether the predicate is true for all items in the colleciton
    is_every(even, [1, 2, 3]) => False
    """
    if col:
        for c in col:
            if not pred(c):
                return False
    return True


def some(pred, col):
    """
    Returns whether the predicate is true for at least one item in
    the collection
    :param pred: the predicate
    :param col: the collection
    :return: whether the predicate is true for at least one item in the colleciton
    is_some(even, [1, 2, 3]) => True
    """
    if col:
        for c in col:
            if pred(c):
                return True
    return False


def frequency(pred, col):
    """
    Returns the number of times the predicate is true in the collection
    :param pred: the predicate
    :param col: the collection
    :return: the number of times the predicate is true in the collection
    frequency(even, [1, 2, 3, 4]) => 2
    """
    if col:
        return thread_last(col,
                           [filter, pred],
                           list,
                           len)
    else:
        return 0


def fnil(f, d):
    """
    Returns an augmented version of the function f, but in the case of a None
    value applied to f, d is supplied instead of None
    :param f: the function
    :param d: the default value
    :return: a function that receives d as a value when None was supplied
    """

    def result_f(*args):
        if first(args) is not None:
            return f(*args)
        else:
            return apply(f, cons(d, rest(args)))

    return result_f


inc0 = fnil(inc, 0)
dec0 = fnil(dec, 0)
plus0 = fnil(plus, 0)
minus0 = fnil(minus, 0)
conjE = fnil(conj, [])


def insert_to_list(d, k, e):
    """
    Inserts the element `e` into the list at key `k` in dict `d`
    :param d: the dict
    :param k: the key
    :param e: the element
    :return: the mutated d
    """
    mupdate(d, k, conjE, e)
    return d


def __collate_frequency(acc, v):
    return mupdate(acc, v, inc0)


def frequencies(l):
    """
    Returns a dictionary, the keys are distinct values of the list `l`,
    the values are the number of times the value exists in the list
    :param l: the list
    :return: the dictionary of value => frequency in list
    frequency(['a', 'b', 'a', 'c']) => {'a': 2, 'b': 1, 'c': 1}
    """
    if l:
        return reduce(__collate_frequency, l, {})
    else:
        return {}


def no_op(*args):
    """
    Literally does nothing; useful for default callbacks
    :param args: anything
    :return: nothing
    """
    pass


def distinct_values(d):
    """
    Returns the distinct values of the dictionary.
    NB: order is not guaranteed
    :param d: the dictionary
    :return: the distinct values
    distinct_values({'a': 1, 'b': 1, 'c': 2}) => [1, 2]
    """
    if d is None:
        return None
    elif d:
        return list(set(d.values()))
    else:
        return []


def equals(*args):
    """
    Function to determine whether all provided arguments are equal
    :param args:
    :return:
    """
    f_val = first(args)
    return every(lambda x: x == f_val, rest(args))


def getter(k, default=None):
    """
    Returns a function that when given a dictionary, returns the
    value at field k
    :param k: the key to get on
    :param default: the default value
    :return: a function which performs get(d, k)
    getter('a')({'a': 1}) => 1
    """

    def return_f(dict):
        return get(dict, k, default=default)

    return return_f


def renamer(d):
    """
    Turns a dictionary into an rename transformer.
    This function returns a function that returns the value in the dictionary for a
    matched key, or just the key itself if there is no match
    :param d: the dictionary
    :return: the transformer
    t = renamer({'a': 'b'})
    t('a') => 'b'
    t('c') => 'c'
    """

    def inner(k):
        return get(d, k, k)

    return inner


def matcher(k, v):
    """
    Returns a function that when applied to a dict yields whether
    the value at that key equals the value provided
    in other words
    m = matcher('a', 2)
    m({'a': 2}) => True
    m({'a': 3}) => False
    :param k: the key of the dict to look at
    :param v: the value to match
    :return:
    """
    return comp(partial(equals, v), getter(k))


def pick(f, c):
    """
    Picks the first value in c that returns True when the function f is applied
    :param f: the function
    :param c: te collection
    :return: the first match
    """
    return first(find(f, c))


def pick_value(k, c):
    """
    Picks the first value from the collection that is truthy for the key k
    :param k: the k
    :param c: the collection
    :return: the first truthy value
    """
    return get(first(find(getter(k), c)), k)


def first_key_match(k, v, c):
    """
    Returns the first dictionary in the collection of dictionaries
    that matches with value `v` at key `k`
    :param k: the key to inspect
    :param v: the value to match
    :param c: the collection of dicts to pick from
    :return: the first dict that matches
    """
    return pick(matcher(k, v), c)


def swarm(f, col, callback=no_op, workers=None):
    """
    Multithreads a function `f` for a provided collection `col`.
    Each element is passed to the function as the only value, all
    of which can process on separate workers in the thread pool.
    A callback can also be applied which would accept the original
    element from the collection as well as the result of the operation.
    If no worker thread count is supplied, the length of the collection is used
    :param f: the function to apply in parallel
    :param col: the collection of items to run in parallel
    :param callback: the callback to invoke after processing
    :param workers: the number of workers in the thread pool
    :param rate: the rate limit, provided as a quantity per second
    :return: None
    """
    if col:
        workers = workers or min(len(col), MAX_WORKERS)
        with ThreadPoolExecutor(workers) as executor:
            for col_element, res in zip(col, executor.map(f, col)):
                callback(col_element, res)


def stream_to_dict(f, col, workers=None):
    results = {}

    def wrapper(element):
        return element, f(element)

    if col:
        workers = workers or min(len(col), MAX_WORKERS)
        with ThreadPoolExecutor(workers) as executor:
            for col_element, res in executor.map(wrapper, col):
                results[col_element] = res

    return results


def stream(f, col, workers=None):
    """
    Multithreads a function `f` for a provided indexed collection `col`.
    Each element is passed to the function as the only value, all
    of which can process on separate workers in the thread pool.
    The result of the function is a collection of results that match
    the order of the provided collection
    :param f: the function to apply in parallel
    :param col: the collection of items to run in parallel
    :param workers: the number of workers in the thread pool
    :return: the results of executing the function on each element of the collection,
    order is maintained
    """
    results = {}

    def wrapper(i, element):
        return col, i, f(element)

    if col:
        workers = workers or min(len(col), MAX_WORKERS)
        with ThreadPoolExecutor(workers) as executor:
            for col_element, i, res in executor.map(partial(apply, wrapper), enumerate(col)):
                results[i] = res

        return [results[i] for i in range(inc(max(results.keys())))]

    return []


def stream_dict(f, d, workers=None):
    results = {}

    def wrapper(key):
        return key, f(key, get(d, key))

    if d:
        workers = workers or min(len(d), 100)
        with ThreadPoolExecutor(workers) as executor:
            for k, res in executor.map(wrapper, d):
                results[k] = res

    return results


def stream_values(f, d, workers=None):
    def dict_function(_, v):
        return f(v)

    return stream_dict(dict_function, d, workers=workers)


def stream_eval(sexprs, workers=None):
    """
    Runs a collection of s expressions in parallel
    :param sexprs: the s expressions
    :param workers: the number of workers
    :return: the result of the stream
    """
    return stream(lisp_eval, sexprs, workers=workers)


class TimeFormat(object):
    MILLIS = 'millis'
    SECONDS = 'seconds'


def datetime_to_epoc(datetime_obj, time_format=TimeFormat.MILLIS):
    t = int(datetime_obj.timestamp())
    return t if time_format == TimeFormat.SECONDS else t * 1000


def convert_date_to_datetime(d, time=datetime.datetime.min.time()):
    return datetime.datetime.combine(d, time)


def now(time_format=TimeFormat.MILLIS):
    """
    Returns the time now in milliseconds (unless format is set to TimeFormat.SECONDS)
    :param time_format: the format to return the time in
    :return: the time now in milliseconds (unless format is set to TimeFormat.SECONDS)
    """
    t = time.time()
    t = t if time_format == TimeFormat.SECONDS else t * 1000
    return int(t)


def format_millis(ms):
    """
    Returns a ms timestamp as a string formatted as e.g. 'Mon Nov 29 14:05:59 2021'
    :param ms: the timestamp in ms
    :return: the string formatted datetime
    """
    return datetime.datetime.fromtimestamp(ms / 1000).strftime('%c')


def timestamp(millis=0, seconds=0, minutes=0, hours=0, days=0, weeks=0, years=0, leap_years=None,
              time_format=TimeFormat.MILLIS):
    """
    Returns the timestamp for the parameters
    :param millis: the number of millis
    :param seconds: the number of seconds
    :param minutes: the number of minutes
    :param hours: the number of hours
    :param days: the number of days
    :param weeks: the number of weeks
    :param years: the number of years
    :param leap_years: the number of years, if not set, this will default to floor(years / 4)
    :param time_format: the format of time, MILLIS is the default
    :return: the number of seconds in the provided params
    """
    if not leap_years:
        leap_years = floor(years / 4)
    t = years * 365 + leap_years
    t += (weeks * 7)
    t += days
    t *= 24
    t += hours
    t *= 60
    t += minutes
    t *= 60
    t += seconds
    return int(((t * 1000) + millis) if time_format == TimeFormat.MILLIS else t)


def ago(millis=0, seconds=0, minutes=0, hours=0, days=0, weeks=0, years=0, leap_years=0, time_format=TimeFormat.MILLIS):
    """
    Returns the time ago. In other words, this returns the epoch timestamp now()
    after removing the timestamp created by the provided params
    :param millis: the number of millis
    :param seconds: the number of seconds
    :param minutes: the number of minutes
    :param hours: the number of hours
    :param days: the number of days
    :param weeks: the number of weeks
    :param years: the number of years
    :param leap_years: the number of years, if not set, this will default to floor(years / 4)
    :param time_format: the format to return the timestamp in, defaults to MILLIS
    :return: the time ago
    """
    return now(time_format=time_format) - timestamp(millis=millis, seconds=seconds, minutes=minutes, hours=hours,
                                                    days=days, weeks=weeks, years=years, leap_years=leap_years,
                                                    time_format=time_format)


class TimeLength:
    YEAR = 1000 * 60 * 60 * 24 * 365
    WEEK = 1000 * 60 * 60 * 24 * 7
    DAY = 1000 * 60 * 60 * 24
    HOUR = 1000 * 60 * 60
    MINUTE = 1000 * 60
    SECOND = 1000
    MILLISECOND = 1


def deconstruct_time(ms):
    years = int(ms / TimeLength.YEAR)
    remainder = ms % TimeLength.YEAR
    weeks = int(remainder / TimeLength.WEEK)
    remainder %= TimeLength.WEEK
    days = int(remainder / TimeLength.DAY)
    remainder %= TimeLength.DAY
    hours = int(remainder / TimeLength.HOUR)
    remainder %= TimeLength.HOUR
    minutes = int(remainder / TimeLength.MINUTE)
    remainder %= TimeLength.MINUTE
    seconds = int(remainder / TimeLength.SECOND)
    remainder %= TimeLength.SECOND

    return [remainder, seconds, minutes, hours, days, weeks, years]


def format_time(ms):
    deconstructed = deconstruct_time(ms)
    reversed_times = reverse(deconstructed)
    to_print = []
    time_index = ["y", "w", "d", "h", "m", "s", "ms"]

    def format_part(part):
        length, index = part
        return "{0}{1}".format(length, time_index[index])

    for i, t in enumerate(reversed_times):
        if t > 0:
            to_print.append([t, i])

        if count(to_print) == 2:
            return join(mapl(format_part, to_print), " ")

    return format_part([ms, 6])


def millis_from_string(s):
    """
    Converts a string time into the number of milliseconds
    NOTE: years 'y' are not supported
    > millis_from_string('3m')
    180000
    > millis_from_string('1w')
    604800000
    > millis_from_string('4w 10d 9h 32m 46s 1ms')
    3317566001
    :param s: the string
    :return: the number of milliseconds
    """
    if s is None:
        return None
    elif not isinstance(s, str):
        raise TypeError("Must be a string")
    elif ' ' in s:
        return apply(plus, mapl(millis_from_string, split(s)))
    elif match('[0-9]+ms', s):
        return int(s.replace("ms", ""))
    elif match('[0-9]+s', s):
        return int(s.replace("s", "")) * 1000
    elif match('[0-9]+m', s):
        return int(s.replace("m", "")) * 60 * 1000
    elif match('[0-9]+h', s):
        return int(s.replace("h", "")) * 60 * 60 * 1000
    elif match('[0-9]+d', s):
        return int(s.replace("d", "")) * 24 * 60 * 60 * 1000
    elif match('[0-9]+w', s):
        return int(s.replace("w", "")) * 7 * 24 * 60 * 60 * 1000
    elif match('[0-9]+y', s):
        raise ValueError("Years are not supported")
    return 0


def capitalize(s):
    """
    Capitalizes the string
    :param s: the string
    :return: the capitalized string
    """
    if s is None:
        return None
    return s.capitalize()


def capitalize_all(s):
    """
    Capitalizes every word in a string sentence
    :param s: string
    :return: the capitalized string
    """
    if s:
        return ' '.join(map(capitalize, s.split(' ')))
    else:
        return None


def get_in(d, ks, default=None):
    """
    Gets the value in the dictionary given the path of keys
    :param d: the dictionary
    :param ks: the keys path
    :param default: a default value to return, defaults to None
    :return: the value at the path
    get_in({'a': {'b': {'c': 3}}}, ['a', 'b', 'c']) => 3
    """
    if not ks or d is None:
        return d
    else:
        state = d
        for k in ks:
            if type(state) is dict:
                state = state.get(k, default)
            else:
                return default
        return state


def assoc_in(d, ks, v):
    """
    Associates a value into a dict using an arbitrary key path
    :param d: the dict
    :param ks: the keypath
    :param v: the value
    :return: the new dict with the value associated in
    > assoc_in({'a': {'b': 1}}, ['a', 'b'], 2)
    {'a': {'b': 2}}
    """
    if d is None:
        d = {}
    if ks is None:
        ks = []
    k = first(ks)
    if len(ks) == 0:
        return merge(d, {None: v})
    elif len(ks) == 1:
        return assoc(d, k, v)
    else:
        value = get(d, k)
        if type(value) is dict:
            new_value = merge(value, assoc_in(value, rest(ks), v))
        else:
            new_value = assoc_in({}, rest(ks), v)
        return assoc(d, k, new_value)


def massoc_in(d, ks, v):
    path_keys = butlast(ks)
    last_key = last(ks)

    nested_d = d
    for k in path_keys:
        ndk = get(nested_d, k)
        if ndk is None or not isinstance(ndk, dict):
            ndk = {}
        nested_d[k] = ndk
        nested_d = ndk

    nested_d[last_key] = v


def dissoc_in(d, ks):
    if d is None:
        return {}
    if ks is None:
        ks = []
    k = first(ks)
    if len(ks) == 0:
        return d
    elif len(ks) == 1:
        return dissoc(d, k)
    else:
        kp = butlast(ks)
        if get_in(d, kp) is None:
            return d
        return update_in(d, butlast(ks), dissoc, last(ks))


def mdissoc_in(d, ks):
    kp = butlast(ks)
    if d and ks and len(ks) > 1 and get_in(d, kp) is not None:
        mupdate_in(d, butlast(ks), dissoc, last(ks))


def update_in(d, ks, f, *args):
    """
    Updates a value into a dict using an arbitrary key path, a function and some optional args
    :param d: the dict
    :param ks: the keypath
    :param f: the function
    :param v: the value
    :param args: optional additional args for the function
    :return: the new dict with the value updated in
    > assoc_in({'a': {'b': 1}}, ['a', 'b'], inc)
    {'a': {'b': 2}}
    """
    if ks is None:
        value = None
    else:
        value = get_in(d, ks)
    new_value = f(value, *args)
    return assoc_in(d, ks, new_value)


def mupdate_in(d, ks, f, *args):
    """
    Updates a value into a dict using an arbitrary key path, a function and some optional args
    :param d: the dict
    :param ks: the keypath
    :param f: the function
    :param v: the value
    :param args: optional additional args for the function
    :return: the new dict with the value updated in
    > assoc_in({'a': {'b': 1}}, ['a', 'b'], inc)
    {'a': {'b': 2}}
    """
    if ks is None:
        value = None
    else:
        value = get_in(d, ks)
    new_value = f(value, *args)
    return massoc_in(d, ks, new_value)


def values_for_key(k, col):
    """
    Returns a set based on the values of a k in a collection of dicts
    :param col: a collection of dicts
    :param k: the key
    :return: a set of values based on the value of the key in each dict
    values_for_key([{'a': 1}, {'a': 2}, {'a': 1}], 'a') => {1, 2}
    """
    if not k or not col:
        return col
    else:
        return list(map(getter(k), col))


def constantly(value):
    """
    Returns a function that always returns the provided value
    :param value: the value to always return
    :return: the functin that always returns the provided value
    """

    def return_f(*args):
        return value

    return return_f


ALWAYS_TRUE = constantly(True)


def case(m, d):
    """
    Similar to a traditional case statement, the value in the case dict
    that matches the value will be returned
    :param m: the value to match
    :param d: the dict
    :return: the value in the dict that matches the provided value
    """
    try:
        return d[m]
    except:
        raise NoMatchingClause(str("No clause for: " + str(m)))


class NoMatchingClause(Exception):
    pass


def is_none(x):
    return x is None


def not_none(x):
    return x is not None


def not_empty(l):
    return l and count(l) > 0


def all_none(*args):
    """
    Returns true if all the arguments are none
    :param args:
    :return:
    """
    return every(is_none, args)


def some_none(*args):
    """
    Returns true if at least of the arguments is none
    :param args:
    :return:
    """
    return some(is_none, args)


def no_none(*args):
    """
    Returns true if none of the args are none
    :param args:
    :return:
    """
    return every(not_none, args)


def not_all_none(*args):
    """
    Returns true if not all the arguments are none
    :param args:
    :return:
    """
    return some(not_none, args)


def merge(*args):
    """
    Merges dictionaries to create a new dictionary.
    Values of conflicting keys are preferred in the latter dict
    :param args: dicts
    :return: the new merged dict
    """
    if every(is_none, args):
        return None

    x = first(args) or {}
    y = second(args) or {}

    new = x.copy()
    new.update(y)
    more = rest(rest(args))
    if more:
        return apply(merge, cons(new, more))
    return new


def merge_with(f, x, y):
    """
    Merges 2 dictionaries to create a new dictionary.
    Values of conflicting keys are resolved with the resolution function
    :param f: the resolution function
    :param x: a dict
    :param y: a dict
    :return: the new merged dict
    """
    if f is None:
        if x is None:
            return y
        elif y is None:
            return x
        else:
            raise TypeError(f)
    else:
        if x is None:
            return y
        elif y is None:
            return x
        else:
            result = x.copy()
            for k, v in y.items():
                x_val = x.get(k)
                if x_val:
                    m = f(x_val, v)
                else:
                    m = v
                result[k] = m
            return result


def deep_merge(x, y):
    """
    Performs a deep merge, merging all nested dicts at the same key
    :param x: a dict
    :param y: a dict
    :return: the merged dict
    """
    if not x and not y:
        return {}
    elif not x:
        return y
    elif not y:
        return x
    else:
        new = x.copy()
        for k, v in y.items():
            x_val = x.get(k)
            if isinstance(v, collections.Mapping) and x_val and isinstance(x_val, collections.Mapping):
                r = deep_merge(x_val, v)
                new[k] = r
            else:
                new[k] = y[k]
        return new


def jsonify(obj):
    """
    Safe conversion of an object to json
    :param obj: an object to jsonify
    :return: the json
    """
    if not obj:
        return None
    if isinstance(obj, str):
        return obj

    return json.dumps(obj)


def decode_json(s):
    if s is None:
        return None
    else:
        return json.loads(s)


def lower(s):
    """
    Returns a lowercased version of the string
    :param s: the string to lowercase
    :return: the lowercased string
    """
    if s is None:
        return None
    else:
        return s.lower()


def cond_first(v, forms):
    """
    Conditionally threads a value through a series of forms
    Each form consists of a function followed by a number of args
    that will follow the form
    :param v: the initial value
    :param forms: the list of conditions/forms
    :return: the result
    """
    state = v
    for condition, form in partition(2, forms):
        if condition:
            if type(form) is list:
                state = apply(first(form), cons(state, rest(form)))
            else:
                state = form(state)
    return state


def cond_last(v, forms):
    """
    Conditionally threads a value through a series of forms
    Each form consists of a function followed by a number of args
    that will precede the form
    :param v: the initial value
    :param forms: the list of conditions/forms
    :return: the result
    """
    state = v
    for condition, form in partition(2, forms):
        if condition:
            if type(form) is list:
                state = apply(first(form), conj(rest(form), state))
            else:
                state = form(state)
    return state


def group_by(f, col):
    """
    Groups a collection by the result of applying the function to each element.
    Elements that have the same result after applying the function will be grouped
    in a list as the value in a dict with the key being the result
    :param f: the function to apply to each element
    :param col: the collection to group
    :return: the dictionary of resulting values to lists of elements which yielded the key
    when the function was applied
    group_by(even, [1, 2, 3, 4]) => {True: [2, 4], False: [1, 3]}
    """
    if not col:
        return {}
    if not f:
        raise TypeError

    return reduce(lambda acc, e: update(acc, f(e), conj, e), col, {})


def count_grouped(f, col):
    """
    Returns a count by key for the elements of the collection grouped by the function
    :param f: the function to group by
    :param col: the collection
    :return: a dictionary of grouped key to count
    count_grouped(even, [1, 2, 3, 4]) => {True: 2, False: 2}
    """
    return {k: count(v) for k, v in group_by(f, col).items()}


def nest(ks, col):
    res = {}
    if ks and col:
        for e in col:
            vs = select_values(e, ks)
            res = update_in(res, vs, conj, e)
    return res


def by_key(f, l):
    """
    Returns a dictionary of the values keyed by the function applied to the value
    :param f:
    :param d:
    :return:
    """
    if isinstance(f, str):
        f = getter(f)
    if not isinstance(l, list):
        raise TypeError("Provided collection must be a list")
    res = {}
    for v in l:
        res[f(v)] = v
    return res


def mapcat(f, col):
    """
    Returns the result of applying concat to the result of applying map
    to f and colls.  Thus function f should return a collection.
    :param f: the fuction
    :param col: the collection
    :return: the result of applying concat to the result of applying map
    to f and colls
    mapcat(partial(cons, 1), [[1], [2], [3]]) => [1 1 1 2 1 3]
    """
    if not col:
        return []
    else:
        return reduce(concat, map(f, col))


def if_value(value):
    """
    If the value passes a truthy test, return it
    :param value: any value
    :return: the value if it passes a truthy test, None otherwise
    """
    if value:
        return value


def gt(f, *args):
    """
    Returns whether each argument is strictly greater than each subsequent argument
    :param f: the first argument
    :param args: subsequent args
    :return: True or False
    """
    l = f
    for a in args:
        if l <= a:
            return False
        l = a
    return True


def lt(f, *args):
    """
    Returns whether each argument is strictly less than each subsequent argument
    :param f: the first argument
    :param args: subsequent args
    :return: True or False
    """
    l = f
    for a in args:
        if l >= a:
            return False
        l = a
    return True


def gte(f, *args):
    """
    Returns whether each argument is greater than or equal to each subsequent argument
    :param f: the first argument
    :param args: subsequent args
    :return: True or False
    """
    l = f
    for a in args:
        if l < a:
            return False
        l = a
    return True


def lte(f, *args):
    """
    Returns whether each argument is less than or equal to each subsequent argument
    :param f: the first argument
    :param args: subsequent args
    :return: True or False
    """
    l = f
    for a in args:
        if l > a:
            return False
        l = a
    return True


def bool_key_wrapper(f, comp):
    def comp_f(a, b):
        a_v = f(a)
        b_v = f(b)
        v = comp(a_v, b_v)
        if type(v) is int:
            return v
        elif v is True:
            return 1
        elif v is False:
            return -1
        else:
            return 0

    return cmp_to_key(comp_f)


def sort_by(f, col, comp=None):
    """
    Sorts a collection by the results of a function applied to each element
    A custom comparator can be provided
    :param f: the function to extract the value to compare from each element
    :param col: the collection
    :param comp: the custom comparator
    :return: the sorted collection
    """
    if col and not f:
        raise TypeError("No function supplied for collection")
    if not col:
        return []
    if comp:
        key_f = bool_key_wrapper(f, comp)
    else:
        key_f = f
    return sorted(col, key=key_f)


def sort_by_key(k, col, comp=None):
    return sort_by(getter(k), col, comp=comp)


desc = lt
asc = gt


def sort_ascending(col):
    """
    Sorts a numerical collection in ascending order
    :param col: the numerical collection
    :return: the sorted collection
    """
    return sort_by(identity, col)


def sort_descending(col):
    """
    Sorts a numerical collection in descending order
    :param col: the numerical collection
    :return: the sorted collection
    """
    return sort_by(identity, col, comp=desc)


def sorted_keys_by_value(d, f=identity, comp=None):
    if not isinstance(d, dict):
        raise TypeError("Can only sort dict by value")

    ks = keys(d)

    return sort_by(lambda k: f(get(d, k)), ks, comp=comp)


def utf8_encode(s):
    """
    Converts a string to bytes using utf-8 encoding
    :param s: string
    :return: bytes
    """
    return bytes(s, 'utf-8')


def utf8_decode(b):
    """
    Decodes bytes to a string using utf-8
    :param b:
    :return:
    """
    return b.decode('utf-8')


def base64enocde(s):
    """
    Base 64 encodes a string
    :param s: the string
    :return: base 64 encoded string
    """
    return utf8_decode(base64.b64encode(utf8_encode(s)))


def repeat(n, v):
    """
    Repeats the value `v` `n` times
    :param n: the number of times to repeat the value `v`
    :param v: the value to repeat `n` times
    :return: the list of n elements, all of which are v
    """
    if n is None:
        return None
    else:
        return list(map(constantly(v), range(n)))


def repeatedly(n, f):
    """
    Repeatedly calls the function `f` `n` times and returns
    the list of results
    :param n: the number of times to repeat the function call
    :param f: the function to call `n` times
    :return: the result of calling the function `n` times
    """
    if n is None:
        return None
    else:
        def invoke(_):
            return f()

        return list(map(invoke, range(n)))


def average(l):
    """
    Finds the average of a list
    :param l: the list of values
    :return: the average of the list
    """
    if not l:
        return None
    else:
        return apply(plus, l) / count(l)


def invert(d):
    if d is None:
        return None
    if isinstance(d, dict):
        inverted = {}
        for k, v in d.items():
            inverted[v] = k
        return inverted

    raise TypeError("Input should be a dictionary")


def find(f, c):
    """
    Filters the given collection by the given function
    :param f: filter function
    :param c: collection to filter
    :return: filtered list
    """
    if c is None:
        return None
    return list(filter(f, c))


def mapl(f, c):
    """
    An unlazy version of map
    :param f:
    :param c:
    :return:
    """
    if c is not None:
        return list(map(f, c))


def map_key(k, c, default=None):
    """
    Gets the values from the dicts in `c` at key `k`
    :param k: the key
    :param c: the collection
    :param default: a default value
    :return: the list of values at k, order is maintained
    """
    return mapl(getter(k, default=default), c)


def map_map_key(kk, vk, c):
    """
    Builds a dict from collection of objects with the keys being the values at kk
    and the values being the values at vk
    :param kk: the key key
    :param vj: the value key
    :param c: the collection
    :return: a dict
    """
    return {get(e, kk): get(e, vk) for e in c}


def map_indexed(f, c):
    """
    Similar to map, but the function accepts two arguments,
    the first being index of the element in the map
    :param f: the function f(index, element)
    :param c: the collection
    :return: the result
    """
    if c is not None:
        return [f(i, e) for i, e in enumerate(c)]


def keep(f, c):
    """
    Returns non-None results of appling f to elements in c
    :param f:
    :param c:
    :return:
    """
    return [f(e) for e in c if f(e) is not None]


def map_keys(f, d):
    """
    Applies the function `f` to every key in `d`
    :param f: the function
    :param d: the dictionary
    :return: the new dictionary
    """
    return {f(k): v for k, v in d.items()}


def map_values(f, d):
    """
    Applies the function `f` to every value in `d`
    :param f: the function
    :param d: the dictionary
    :return: the new dictionary
    """
    return {k: f(v) for k, v in d.items()}


def split(s, sep=None):
    """
    Splits the string using the separator
    wrapper for s.split(sep)
    :param s: the string
    :param sep: the separator
    :return: the split string
    """
    if s is None:
        return None
    elif isinstance(s, str):
        return s.split(sep=sep)
    else:
        raise TypeError("Cannot be split")


def replace(s, before, after):
    """
    None safe replace function
    :param s: the string
    :param before: the before string to replace
    :param after: the after string to replace with
    :return: the changed string
    e.g.
    > replace("hello world", "world", "universe")
    "hello universe"
    """
    if s is None:
        return None
    return s.replace(before, after)


def first_part(s, sep=None):
    """
    Returns the first part when splitting with a separator
    :param s: the string
    :param sep: the separator
    :return: the first part after splitting
    """
    return first(split(s, sep=sep))


def last_part(s, sep=None):
    """
    Returns the last part when splitting with a separator
    :param s: the string
    :param sep: the separator
    :return: the last part after splitting
    """
    return last(split(s, sep=sep))


def nth_part(s, n, sep=None):
    """
    Returns the nth part when splitting with a separator
    :param s: the string
    :param n: the index
    :param sep: the separator
    :return: the nth part after splitting
    """
    return nth(split(s, sep=sep), n)


def drop_first_part(s, sep=None):
    """
    Drops the first part when splitting with a separator
    :param s: the string
    :param sep: the separator
    :return: the string without the last part after splitting
    """
    return join(rest(split(s, sep)), sep)


def drop_last_part(s, sep=None):
    """
    Drops the last part when splitting with a separator
    :param s: the string
    :param sep: the separator
    :return: the string without the last part after splitting
    """
    return join(butlast(split(s, sep)), sep)


def join(l, sep=" "):
    """
    Joins the list using the separator
    wrapper for sep.join(l)
    :param l: the list
    :param sep: the separator
    :return: the joined string
    """
    if l is None:
        return None
    elif isinstance(l, (list, tuple)):
        l = mapl(str, l)
        return sep.join(l)
    else:
        raise TypeError("Cannot be joined")


def first_valid(f, l):
    """
    Applies the function to each value in the list `l` until
    a value is returned, if calling the function on an element
    throws an exception it will move to the next value in the list
    :param f: the function
    :param l: the list of values
    :return: the first non-exceptional value of f(li) where li in l
    """
    if l:
        item = first(l)
        try:
            return f(item)
        except:
            return first_valid(f, rest(l))
    return None


def empty(v):
    """
    Return whether the value is empty
    :param v: the value
    :return: whether it's empty
    """
    return count(v) == 0 if v else True


def contains_nil_values(v):
    if isinstance(v, dict):
        for key, value in v.items():
            if value is None:
                return True
        return False
    elif isinstance(v, list):
        for value in v:
            if value is None:
                return True
        return False
    return None


def most_frequent_values(f, col, limit=None, threshold=None):
    """
    Filters a list of maps by the frequency of the values (yielded by the function applied
    to the elements) in the whole collection.
    If no limit or threshold is specified, the collection is returned unchanged.
    A threshold will ensure that a map is only returned if its value by `field`
    has been noticed at least that many times in the collection
    A limit will ensure that after we've sorted the collection maps by the frequency
    their value has been noticed, we simply cap the first `limit` resuts
    :param f: the function that yields the value under inspection
    :param col: the collection to filter
    :param limit: the max number of elements to return
    :param threshold: the minimum number of times a value has to appear to be returned
    :return: the filtered collection
    """
    if limit or threshold:
        field_values = mapl(f, col)
        field_frequencies = frequencies(field_values)
        map_list = []
        for value, c in field_frequencies.items():
            if not threshold or c >= threshold:
                map_list.append({'value': value, 'count': c})
        ml_sorted = sort_by(getter('count'), map_list, comp=lte)
        allowed_values = mapl(getter('value'), ml_sorted)
        return_col = []
        for x in col:
            v = f(x)
            if v in allowed_values:
                return_col.append(x)

        if limit:
            return take(limit, return_col)
        else:
            return return_col

    else:
        return col


def camel_to_snake(string):
    """
    Converts a camel cased string to a snake cased string
    :param string: a camel cased string
    :return: a snake cased string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def percentile(index, total):
    if not total or total == 0:
        return 0

    i = index
    if index == 0 and total == 1:
        i = 1

    return float(i / total * 100)


def string_safe(s, ignore_case=True, default=''):
    if s:
        result = s

        if ignore_case:
            result = s.lower()

        return result.strip()

    return default


def number_safe(i, default=None):
    return int(i) if i else default


class MultiMethod:
    def __init__(self, dispatch, default=None):
        if not dispatch:
            raise ValueError('Multimethod requires dispatch.')

        if callable(dispatch):
            self.dispatch = dispatch
        else:
            self.dispatch = lambda x: get(x, dispatch)
        self.default = default
        self.method = {}

    def resolve_method(self, *args, **kwargs):
        v = self.dispatch(*args, **kwargs)
        f = get(self.method, v, self.default)
        if not f:
            raise NotImplementedError("No dispatch found for %".format(v))
        if not callable(f):
            return constantly(f)
        return f

    def __call__(self, *args, **kwargs):
        f = self.resolve_method(*args, **kwargs)
        return f(*args, **kwargs)


def map_to_multi(m, dispatch, default):
    mm = MultiMethod(dispatch, default)

    for k, f in m.items():
        mm.method[k] = f

    return mm


def for_dict(f, d):
    """
    Accepts a dictionary a returns a dictionary with keys
    and values determined by applying a function to the k-v
    pair of the passed in d.
    => for_dict(juxt(second, first), {'a': 1, 'b': 2})
    {1: 'a', 2: 'b'}
    :param f:
    :param d:
    :return:
    """
    if d is None:
        return None
    rd = {}
    for k, v in d.items():
        r = f(k, v)
        rd[first(r)] = second(r)
    return rd


def map_map(kf, vf, l):
    """
    Accepts a list, applies a `kf` to the entry to generate a key,
    a `vf` to generate the value and returns the resulting dict
    :param kf: the key function
    :param vf: the value function
    :param l: the list
    :return: the resulting dict
    """
    if l is None:
        return None
    return {kf(e): vf(e) for e in l}


def key_by(k, l):
    return map_map(getter(k), identity, l)


def pair_by(k, v, l):
    return map_map(getter(k), getter(v), l)


def get_dir(file_path):
    if file_path is None:
        return None
    dirname = os.path.dirname(file_path)
    return dirname if dirname != '' else None


def get_filename(file_path):
    if file_path is None:
        return None
    filename = second(os.path.split(file_path))
    return filename if filename != '' else None


def get_extension(file_path):
    if file_path is None:
        return None
    ext = second(os.path.splitext(file_path))
    replaced = lower(ext.replace('.', ''))
    return replaced if replaced != '' else None


def break_path(file_path):
    if file_path is None:
        return None
    path_with_name, _ = os.path.splitext(file_path)
    return get_dir(file_path), get_filename(path_with_name), get_extension(file_path)


def upper(s):
    if s is None:
        return None
    return s.upper()


def lower(s):
    if s is None:
        return None
    return s.lower()


def dict_f(d, default=None):
    """
    Accepts a dict and returns a function that accepts a key,
    and returns the value from the dict passed in
    :param d: the dict
    :param default: a default value if there is no value at the key
    :return: a function
    """

    def inner_f(k):
        return get(d, k, default=default)

    return inner_f


def lower_keys(d):
    """
    Makes keys lower case
    :param d: the dict
    :return: the dict with lower case keys
    """
    if d is not None:
        return map_keys(lower, d)


def value_equals_f(k, v, default=None):
    """
    Returns a function that checks the value of a dict d
    equals v for key k
    :param k: the key in the dict
    :param v: the value
    :param default: the default value
    :return: whether get(d, k) = v
    """

    def inner_f(d):
        return get(d, k, default=default) == v

    return inner_f


def unpack_values(it, ks):
    """
    Wraps an generator (which must yield dicts), and yields
    the values of the dict at the provided ks
    :param it: the iterator
    :param ks: the keys to get the values for
    :return: a new generator
    """
    for d in it:
        yield select_values(d, ks)


def str_has_upper_case(s):
    if s is None:
        return None
    return any(x.isupper() for x in s)


def snaked(*args):
    """
    Joins the args up with underscores
    :param args: the arguments to join
    :return: a string with the args separated by an underscore
    snaked(1, 'hello', 2) => "1_hello_2"
    """
    return join(mapl(str, args), sep="_")


def kebabed(*args):
    """
    Joins the args up with dashes
    :param args: the arguments to join
    :return: a string with the args separated by a dash
    snaked(1, 'hello', 2) => "1-hello-2"
    """
    return join(mapl(str, args), sep="-")


def date_to_millis(dt):
    """
    Converts a datetime to a UTC millis timestamp
    :param dt: the datetime
    :return: millis
    """
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def millis_to_date(millis):
    """
    Converts millis to a UTC datetime
    :param millis: the millis
    :return: the datetime
    """
    return datetime.datetime.fromtimestamp(millis / 1000, tz=datetime.timezone.utc)


def flatten_list(l):
    '''
    Flattens an lists of lists
    :param l:
    :return:
    flatten_list([[1,2,3], [4,5,6], None, [None]]) => [1,2,3,4,5,6]
    '''
    l = l if l else []
    return list(chain(*filter_none_values(l)))


class DateFormat:
    DATE_TIME = "%Y-%m-%d %H:%M:%S"
    ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
    DATE = "%Y-%m-%d"


def format_millis(millis, fmt=DateFormat.DATE_TIME):
    dt = datetime.datetime.utcfromtimestamp(millis / 1000)
    return dt.strftime(fmt)


def largest_dict_entries(d, n):
    """
    Returns the largest n dict entries by value
    :param d: the dict
    :param n: the number of entries to return
    :return: a dict with k/v with the largest n vs in d
    """
    if d:
        as_tuples = [[k, v] for k, v in d.items()]
        s = sort_by(second, as_tuples, comp=desc)
        return take(n, s)


def max_dict_entry(d):
    """
    Returns the key, value pair in the dict with the largest value
    :param d: the dict
    :return: the key, value pair with the largest value
    """
    return first(largest_dict_entries(d, 1))


def unique_values(k, ds):
    return dedupe(map_key(k, ds))


def key_values_in(k, ds):
    """
    Expects a list value (or None) at the key `k` for each d in ds,
    returns a dict of all possible values within the key `k` of the ds, as keys,
    whose values are all the ds whose value at `k` contains that key
    :param ds: the dicts
    :param k: the key
    :return:
    => key_in('v', [{'a': 1, 'v': [1, 2, 3]}, {'a': 2, 'v': [2, 3]}, {'a': 3, 'v': [1, 3]}])
    {1: [{'a': 1, 'v': [1, 2, 3]}, {'a': 3, 'v': [1, 3]}],
     2: [{'a': 1, 'v': [1, 2, 3]}, {'a': 2, 'v': [2, 3]}],
     3: [{'a': 1, 'v': [1, 2, 3]}, {'a': 2, 'v': [2, 3]}, {'a': 3, 'v': [1, 3]}]}
    """
    if ds is None:
        return None
    return_d = {}
    for d in ds:
        vs = get(d, k)
        if vs:
            for v in vs:
                mupdate(return_d, v, conjE, d)
    return return_d


def largest(vs):
    """
    The largest of the args
    :param args: args
    :return: the largest of the args
    """
    if not vs:
        return None
    return max(vs)


def smallest(vs):
    """
    The smallest of the args
    :param args: args
    :return: the smallest of the args
    """
    if not vs:
        return None
    return min(vs)


def dict_values_proportion(l: list[dict]) -> Optional[dict]:
    if l is None:
        return None

    total = {}

    for d in l:
        total = merge_with(plus, total, d)

    values_sum = sum(values(total))

    return {
        k: v / (values_sum or 1)
        for k, v in total.items()
    }


def flip(l):
    if l is None:
        return None
    if count(dedupe(mapl(count, l))) > 1:
        raise ValueError("Not all vectors are the same length")

    return [[row[i] for row in l] for i in range(count(first(l)))]


def find_key_matches(k, v, ds):
    """
    Find dicts in `ds` whose value at `k` matches `v`
    :param k: the key
    :param v: the value
    :param ds: the dicts
    :return: the dicts whose value for k matches v
    """
    return find(lambda d: get(d, k) == v, ds)


def pad(n, c, default=None):
    """
    Ensures the length of the collection c is at least n
    :param n: the minimum size of collection
    :param c: the collection
    :param default: the elements to pad the collection with if it is less than n
    :return: the padded collection which is at least `n` in size
    """
    length = count(c) if c else 0
    missing = n - length
    return (c or []) + repeat(missing, default)


def merge_lists_with(f, *c):
    """
    Merges lists of dicts, preferring the latter dict when there is a conflict determined by the function f
    :param c: the list of dicts
    :param f: a function to determine equivalence
    :return: the merged lists
    """
    c = filter_none_values(c)
    if not c:
        return None

    c = reverse(c)
    keyvals = []
    merged = []
    for e in concat(*c):
        v = f(e)
        if v not in keyvals:
            merged.append(e)
            keyvals.append(v)
    return reverse(merged)


def value_in_range(v, min, max):
    '''
    Returns true or false if value is in range.
    :param v: Integer value in question
    :param min:
    :param max:
    :return: value_in_range(10, 0, 10) => True
    value_in_range(5, None, 10) => True
    '''
    if v is None:
        return False
    if min and v < min:
        return False
    if max and v > max:
        return False

    return True


def truncate(v, limit, end_with=''):
    if not v:
        return v
    return "{}{}".format(v[:limit], end_with)


def keyify(*args):
    return join(args, sep="|")


def remove_all(v, l):
    """
    Removes all instances of v from l, preserves order
    :param v: the value to remove
    :param l: the list
    :return: the new list without instances of v
    """
    return find(lambda e: e != v, l)


def timed(gen):
    """
    Returns a time elapsed from the generator provided
    :param gen:
    :return:
    """
    init = now()
    for e in gen:
        t = now()
        yield init, (t - init), e


def predictive_timed(gen, total, count_fn=constantly(1)):
    """
    Returns a new generator which indicates a projected end time
    given the total size of the elements in the generator
    :param gen: the generator
    :param total: the total number to use for proportion
    :param count_fn: the function to apply to the element to determine how much
    to increment the count by, defaults to a function that always returns 1
    :return:
    """
    processed = 0
    for init, elapsed, e in timed(gen):
        projected_end = None
        if processed:
            proportion = processed / total
            projected_end = int(init + (elapsed / proportion))
        processed += count_fn(e)
        yield processed, elapsed, projected_end, e


def with_progress_logging(gen, total, logger, count_fn=constantly(1)):
    for processed, elapsed, projected_end, e in predictive_timed(gen, total, count_fn=count_fn):
        if projected_end:
            remaining = projected_end - now()
            logger.info(f"Elapsed: {format_time(elapsed)}, Remaining: {format_time(remaining)}, Projected End: {format_millis(projected_end)}")
        logger.info("Processing {0}/{1}".format(processed, total))
        yield e


def is_url_relative(url):
    if 'http://' in url or 'https://' in url:
        return False
    return True


def get_attribute(instance, attrs, default=None):
    """
    Similar to Python's built in getattr(instance, attr), but takes a list of nested attributes.
    Also accepts either attribute lookup on objects or dictionary lookups.
    """
    attribute_value = instance
    for attr_name in attrs.split('.'):
        try:
            attribute_value = getattr(attribute_value, attr_name)
        except AttributeError:
            return default
        if attribute_value is None:
            return default
    return attribute_value


def is_number_string(s):
    try:
        return isinstance(s, str) and int(s)
    except:
        return False


def format_number(number):
    """
    Formats a number as a string with commas
    :param number: the number, either an int or a string that is a number
    :return: the formatted number
    >>> format_number(123456789)
    "123,456,789"
    """
    if isinstance(number, int) or is_number_string(number):
        digits = reverse(mapl(identity, str(number)))
        number_sections = reverse(mapl(reverse, partition(3, digits)))
        return join(mapl(lambda section: join(section, ""), number_sections), sep=",")
    return None


def get_default_if_none(value: Any, default: Any):
    return value or default
