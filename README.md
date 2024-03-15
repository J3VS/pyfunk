# pyfunk
useful python functions

### partition(size, col)
*Partitions a collection into chunks of a specified size*
```python
>>> partition(3, [1, 2, 3, 4, 5, 6, 7, 8])
[[1, 2, 3], [4, 5, 6], [7, 8]]
```

### select_keys(dict, keys)
*Returns a dict containing only those entries in dict whose key is in keys*
```python
>>> select_keys({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
{'a': 1, 'c': 3}
```

### assoc(dict, **kwargs)
*Returns a new dictionary with the kwargs associated into the dict, the original dict is unchanged*
```python
>>> x = {'a': 1}
>>> y = assoc(x, b=2)
>>> y
{'a': 1, 'b': 2, 'c': 3}
>>> x
{'a': 1}
```

### dissoc(dict, **args)
*Returns a new dictionary with provided keys removed, the original dict is unchanged*
```python
>>> x = {'a': 1, 'b': 2, 'c': 3}
>>> y = dissoc(x, 'b', 'c')
>>> y
{'a': 1}
>>> x
{'a': 1, 'b': 2, 'c': 3}
```

### update(dict, key, func)
*Returns a new dictionary with the value at key updated with the function, the original dict is unchanged*
```python
>>> x = {'a': 1, 'b': 2}
>>> y = update(x, 'b', inc)
>>> y
{'a': 1, 'b': 3}
>>> x
{'a': 1, 'b': 2}
```

### massoc(dict, **kwargs)
*Associates the kwargs into the dictionary, this is a mutable operation and alters the state of the provided dict*
```python
>>> x = {'a': 1}
>>> massoc(x, b=2, c=3)
{'a': 1, 'b': 2, 'c': 3}
>>> x
{'a': 1, 'b': 2, 'c': 3}
```

### mdissoc(dict, **args)
*Removes the provided keys from the dictionary, this is a mutable operation and alters the state of the provided dict*
```python
>>> x = {'a': 1, 'b': 2, 'c': 3}
>>> mdissoc(x, 'b', 'c')
{'a': 1}
>>> x
{'a': 1}
```

### mupdate(dict, key, func)
*Updates the value of key in the dict with the provided function, this is a mutable operation and alters the state of dict*
```python
>>> x = {'a': 1, 'b': 2}
>>> mupdate(x, 'b', inc)
{'a': 1, 'b': 3}
>>> x
{'a': 1, 'b': 3}
 ```

### rename_keys(dict, keymap)
*Renames the keys in a dictionary as outlined by the keymap; unspecified keys are left unchanged*
```python
>>> x = {'a': 1, 'b': 2, 'c': 3}
>>> y = rename_keys(x, {'a': 'x', 'b': 'y'})
>>> y
{'x': 1, 'y': 2, 'c': 3}
```

### select_rename(dict, keymap)
*Renames the keys in a dictionary as outlined by the keymap; unspecified keys are removed*
```python
>>> x = {'a': 1, 'b': 2, 'c': 3}
>>> y = rename_keys(x, {'a': 'x', 'b': 'y'})
>>> y
{'x': 1, 'y': 2}
```

### get(obj, key, default=None)
*Returns the value in the object for the given key. The obj must implement __getitem__,
a default value can be supplied and is returned if the value would be None*
```python
>>> x = {'a': 1, 'b': 2}
>>> get(x, 'b')
2
>>> get(x, 'c', default=3)
3
```

### get_items_as(obj, keymap)
*The same as select_rename, but extends beyond dicts to objects that implement __getitem__*
```python
>>> x = {'a': 1, 'b': 2, 'c': 3}
>>> y = rename_keys(x, {'a': 'x', 'b': 'y'})
>>> y
{'x': 1, 'y': 2}
```

### filter_none_values(x)
*Returns a new object with none values removed from the original one, this function accepts either a dict or a list*
```python
>>> filter_none_values({'a': 1, 'b': None})
{'a': 1}
>>> filter_none_values(['a', 'b', None, 'c', None])
['a', 'b', 'c']
```

### zipmap(keys, vals)
*Returns a dictionary with the keys mapped to the values, any additional values that dont match a key/value are dropped*
```python
>>> zipmap(['a', 'b', 'c'], [1, 2, 3])
{'a': 1, 'b': 2, 'c': 3}
```

### dedupe(list)
*Removes duplicates from a list. Order is not maintained*
```python
>>> dedupe([1, 2, 1, 2, 1, 1, 2, 3, 4, 5, 3, 5])
[1, 2, 3, 4, 5]
```

### identity(x)
*The identity function, returns the arg*
```python
>>> identity(1)
1
>>> map(identity, [1, 2, 3])
[1, 2, 3]
```

### inc(x)
*Increments a number*
```python
>>> inc(1)
2
```

### dec(x)
*Decrements a number*
```python
>>> dec(2)
1
```

### plus(x, *args)
*Adds together at least one number*
```python
>>> plus(1, 3, 5, -3, 6)
12
```

### minus(x, *args)
*Subtracts a sequence of numbers from an initial number*
```python
>>> minus(1, 4, 2)
-5
```

### multiply(x, *args)
*Multiplies at least one number together*
```python
>>> multiply(2, 4, 2)
16
```

### divide(x, *args)
*Divides an initial number by a sequence of numbers*
```python
>>> divide(30, 6, 5)
1.0
```

### apply(f, argslist)
*Applies the function f to the argslist*
```python
>>> apply(sum, [1, 2, 3, 4])
10
```

### lisp_eval(l)
*Evaluates a list where the first value is the function and the rest are the arguments.
Returns the result of calling the function on the args*
```python
>>> lisp_eval([inc, 0])
1
>>> lisp_eval([plus, 1, 2])
3
```

### into_list(*args)
*Returns all the provided args in a list*
```python
>>> into_list(1, 2, 3)
(1, 2, 3)
```

### even(x)
*Returns whether the argument is an even number*
```python
>>> even(4)
True
>>> even(17)
False
```

### odd(x)
*Returns whether the argument is an odd number*
```python
>>> odd(4)
False
>>> odd(17)
True
```

### count(l)
*Returns the numer of elements in a list, count(None) is 0*
```python
>>> count(None)
0
>>> count([])
0
>>> count([1, 2, 3])
3
```

### first(l)
*Returns the first element from the list*
```python
>>> first([1, 2, 3, 4, 5])
1
```

### last(l)
*Returns the last element from the list*
```python
>>> last([1, 2, 3, 4, 5])
5
```

### rest(l)
*Returns all but the first element of a list*
```python
>>> rest([1, 2, 3, 4, 5])
[2, 3, 4, 5]
```

### take(n, l)
*Takes the first n elements from the list*
```python
>>> take(5, [1])
[1]
>>> take(5, [6, 5, 2, 7, 3, 8, 4, 2])
[6, 5, 2, 7, 3]
```

### drop(n, l)
*Drops the first n elements from the list*
```python
>>> drop(5, [1])
[]
>>> drop(5, [6, 5, 2, 7, 3, 8, 4, 2])
[8, 4, 2]
```

### take_last(n, l)
*Takes the last n elements from the list*
```python
>>> take_last(5, [1])
[1]
>>> take_last(5, [6, 5, 2, 7, 3, 8, 4, 2])
[7, 3, 8, 4, 2]
```

### drop_last(n, l)
*Drops the last n elements from the list*
```python
>>> drop_last(5, [1])
[]
>>> drop_last(5, [6, 5, 2, 7, 3, 8, 4, 2])
[6, 5, 2]
```

### butlast(l)
*Returns all but the last element of a list*
```python
>>> butlast([1, 2, 3, 4, 5])
[1, 2, 3, 4]
```

### cons(v, l)
*Appends the value v to the beginning of the list l*
```python
>>> cons(1, [2, 3, 4, 5[)
[1, 2, 3, 4, 5)
```

### conj(l, v)
*Appends the value v to the end of the list l*
```python
>>> conj([1, 2, 3, 4], 5)
[1, 2, 3, 4, 5]
```

### concat(*args)
*Concatenates an arbitrary number of lists*
```python
>>> concat([1, 2, 3], [4, 5], [6, 7, 8])
[1, 2, 3, 4, 5, 6, 7, 8]
```

### thread_first(initial, *args)
*Threads a value through numerous forms passing the resulting value to the next form. Each form is either a function
(in which case the threaded value is passed as the only argument), or a vector with the first value being a function and
subsequent values being additional args (in which case the threaded value is passed as the first argument)*
```python
>>> thread_first(1, inc, [plus 5], dec)
6
```

### thread_last(initial, *args)
*Threads a value through numerous forms passing the resulting value to the next form. Each form is either a function
(in which case the threaded value is passed as the only argument), or a vector with the first value being a function and
subsequent values being additional args (in which case the threaded value is passed as the last argument)*
```python
>>> thread_last(1, inc, [plus 5], dec)
6
>>> thread_last([1, 2, 3, 4, 3, 2, 1],
                [map, inc],
                [filter, even],
                dedupe)
[2, 4]
```

### reverse(l)
*Returns a new list which is the reverse of the provided list, the original list is unchanged*
```python
>>> reverse([1, 2, 3, 4])
[4, 3, 2, 1]
```

### juxt(arg, *args)
*Takes a set of functions and returns a fn that is the juxtaposition of those fns.  The returned fn takes a
variable number of args, and returns a vector containing the result of applying each fn to the args (left-to-right).*
```python
>>> f = juxt(inc, dec, even)
>>> f(1)
[2, 0, True]
```

### comp(*args)
*Composes an arbitrary number of functions into the returned function, the provided functions are composed right-to-left*
```python
>>> c = comp(str, divide)
>>> c(8, 8, 8)
'0.125'
```

### every(pred, col)
*Returns whether the predicate is true for all items in the collection*
```python
>>> every(even, [1, 2, 3, 4])
False
>>> every(even, [2, 4, 6, 8])
True
```

### some(pred, col)
*Returns whether the predicate is true for at least one item in the collection*
```python
>>> some(even, [1, 2, 3, 4])
True
>>> some(even, [3, 5, 7, 8])
False
```

### frequency(pred, col)
*Returns the number of times the predicate is true for items in the collection*
```python
>>> frequency(even, [1, 2, 3, 4])
2
>>> frequency(even, [3, 5, 7, 9])
0
```

### fnil(f, d)
*Returns an augmented version of the function f, but when a None value is supplied as the first argument,
the value of d is handed to f as the first argument instead*
```python
>>> fninc = fnil(inc, 0)
>>> inc(None)
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
>>> fninc(None)
0
```

### inc0(x)
*Equivalent to fnil(inc, 0). This is a None-safe way of inc'ing a number, so that inc0(None) is 1,
where inc(None) raises an Exception*
```python
>>> inc(None)
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
>>> inc0(None)
1
```

### dec0(x)
*Equivalent to fnil(dec, 0). This is a None-safe way of dec'ing a number, so that dec0(None) is -1,
where dec(None) raises an Exception*
```python
>>> dec(None)
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
>>> dec0(None)
-1
```

### frequencies(l)
*Returns a dictionary, the keys are distinct values of the list `l`, the values are the number of times
the value exists in the list*
```python
>>> frequencies([1, 2, 3, 2, 3, 1, 2, 3, 1, 2, 2, 1, 3])
{1: 3, 2: 4, 3: 3}
```

### no_op(*args)
*Does nothing, useful for default callbacks*
```python
>>> no_op(1, 'a', [4, 3, 5])
>>> ...
```

### distinct_values(d)
*Returns a list of the distinct values of a dictionary, there is no order guarantee*
```python
>>> distinct_values({'a': 1, 'b': 1, 'c': 2, 'd': 1})
[1, 2]
```

### getter(k)
*Returns a function that when applied to a dictionary will return the value in the dictionary at key k*
```python
>>> getter('a')({'a': 1})
1
>>> map(getter('a'), [{'a': 1}, {'a': 2}, {'a': 3}])
[1, 2, 3]
```

### swarm(f, col, callback=no_op, workers=len(col))
*Runs the function f for each item in the collection in parallel. This is done in a thread pool with a number
of workers equal to the length of the collection, unless otherwise specified. An optional callback can also be applied
to the result of each operation, this function accepts both the original element and the result of the function*
```python
>>> x = {}
>>> def cb(initial, result):
...    x[initial] = result
...
>>> swarm(inc, [1, 2, 3, 4], callback=cb)
>>> x
{1: 2, 2: 3, 3: 4, 4: 5}
```

### stream(f, col, workers=len(col))
*Runs the function f for each item in the collection in parallel. This is done in a thread pool with a number
of workers equal to the length of the collection, unless otherwise specified. The result of the function is the
collection of results from applying the function to each element. Order is maintained*
```python
>>> stream(inc, [1, 2, 3, 4])
[2, 3, 4, 5]
```

### stream_eval(ls, workers=len(col))
*Runs the s-expressions in parallel*
```python
>>> stream_eval([[inc, 0], [plus, 1, 2]])
[1, 3]
```

### now(time_format=TimeFormat.MILLIS)
*Returns the timestamp now. By default this is returned in milliseconds, but can be coerced to seconds using
TimeFormat.SECONDS as the time_format*
```python
>>> now()
1503952632000
>>> now(time_format=TimeFormat.SECONDS)
1503952632
```

### timestamp(millis=0, seconds=0, minutes=0, hours=0, days=0, weeks=0, years=0, leap_years=None,
              time_format=TimeFormat.MILLIS)
*Returns the number of milliseconds for the provided number of each time denomination. The format can be coerced to seconds using
TimeFormat.SECONDS as the time_format*
```python
>>> timestamp(millis=100, days=2)
172800100.0
```

### ago(millis=0, seconds=0, minutes=0, hours=0, days=0, weeks=0, years=0, leap_years=0, time_format=TimeFormat.MILLIS)
*Returns the unix timestamp for the amount of provided time before the current moment. This is timestamp() subtracted
from now()*
```python
>>> ago(days=2, hours=4, seconds=30)
1503765749000.0
```

### capitalize_all(s)
*Capitalizes each word in a string*
```python
>>> capitalize_all("hello world")
"Hello World"
```

### get_in(d, ks, default=None)
*Returns the value in the dictionary given the nested path of keys, an optional default value can be supplied*
```python
>>> x = {'a': {'b': 1, 'c': 2, 'd': {'e': 3}}, 'f': 4}
>>> get_in(x, ['a'])
{'b': 1, 'c': 2, 'd': {'e': 3}}
>>> get_in(x, ['a', 'b'])
1
>>> get_in(x, ['a', 'd', 'e'])
3
>>> get_in(x, ['a', 'e', 'j'], default=5)
5
```

### assoc_in(d, ks, v)
*Associates a value in a dictionary given the key path, returns the new dictionary*
```python
>>> x = {'a': {'b': 1, 'c': 2, 'd': {'e': 3}}, 'f': 4}
>>> assoc_in(x, ['a', 'd', 'e'], 5)
{'a': {'b': 1, 'c': 2, 'd': {'e': 5}}, 'f': 4}
```

### update_in(d, ks, f, *args)
*Updates the value in a dictionary given a key path, a function and some optional args,
the current value is passed in as the first arg, the new dictionary is returned*
```python
>>> x = {'a': {'b': 1, 'c': 2, 'd': {'e': 3}}, 'f': [4]}
>>> update_in(x, ['a', 'd', 'e'], inc)
{'a': {'b': 1, 'c': 2, 'd': {'e': 4}}, 'f': [4]}
>>> update_in(x, ['f'], conj, 1)
{'a': {'b': 1, 'c': 2, 'd': {'e': 3}}, 'f': [4, 1]}
```

### values_for_key(k, col_of_dicts)
*Returns a list of values for the dicts at the given k*
```python
>>> values_for_key('a', [{'a': 1}, {'a': 2}, {'a': 3}])
[1, 2, 3]
```

### constantly(value)
*Returns a function that constantly returns the provided value*
```python
>>> x = constantly(1)
>>> x()
1
>>> x('a', [1, 2, 3], "Random Things")
1
```

### ALWAYS_TRUE
*A function that always returns true*
```python
>>> ALWAYS_TRUE()
True
>>> ALWAYS_TRUE(1, "Hello")
Ture
```

### case(value, d)
*Similar to a traditional case statement, but the dictionary d has no delayed execution, it is simply a dictionary
of key to value*
```python
>>> x = 'a'
>>> case(x, {'a': 1, 'b': 2, 'c': 3})
1
```

### merge(d1, d2)
*Merges 2 dictionaries together. In the case of a collision, the value from the latter dictionary is preferred*
```python
>>> merge({'a': 1, 'b': 2}, {'b': 3, 'd': 4})
{'a': 1, 'b': 3, 'd': 4}
```

### merge_with(f, d1, d2)
*Merges 2 dictionaries together. In the case of a collision, the values are passed to the function f for resolution*
```python
>>> merge_with(plus, {'a': 1, 'b': 2}, {'b': 3, 'd': 4})
{'a': 1, 'b': 5, 'd': 4}
>>> merge_with(concat, {'a': [1], 'b': [2]}, {'b': [3], 'd': [4]})
{'a': [1], 'b': [2, 3], 'd': [4]}
```

### deep_merge(d1, d2)
*Recursively merges 2 dictionaries, merging all nested dicts at the same key. Non-dict values of key collisions
are resolved by preferring the latter dict*
```python
>>> deep_merge({'a': {'b': 1}, 'c': 3}, {'a': {'d': 4}, 'c': 4})
{'a': {'b': 1, 'd': 4}, 'c': 4}
```

### jsonify(s)
*Safe conversion of an object to json*
```python
>>> jsonify(None)
>>> jsonify("Hello World")
'Hello World'
>>> jsonify({'a': 1})
'{"a": 1}'
```


### lower(s)
*Returns the lower case version of a string*
```python
>>> lower(None)
>>> lower("HELLo WorlD")
'hello world'
```

### cond_first(v, d)
*Threads (first) a value through a series of forms based on conditions*
```python
>>> cond_first({}, [even(2), [assoc, 'a', 1],
                    even(1), [assoc, 'b', 2],
                    even(4), [assoc, 'c', 3],
                    even(6), jsonify])
'{"a": 1, "c": 3}'
```

### cond_first(v, d)
*Threads (last) a value through a series of forms based on conditions*
```python
>>> cond_last([1, 2, 3, 4], [even(2), [map, inc],
                             odd(2),  [filter, even],
                             odd(1),  [map, jsonify]])
['2', '3', '4', '5']
```

### group_by(f, col)
*Returns a map of the elements of col keyed by the result of
f on each element. The value at each key will be a vector of the
corresponding elements, in the order they appeared in coll.*
```python
>>> group_by(len, ["cat", "dog", "sheep", "monkey", "pig", "ox", "mule"])
{2: ["ox"], 3: ["cat", "dog", "pig"], 4: ["mule"], 5: ["sheep"], 6: ["monkey"]}
```

### mapcat(f, col)
*Returns the result of applying concat to the result of applying map
to f and colls.  Thus function f should return a collection.*
```python
>>> mapcat(reverse, [[3, 2, 1, 0], [6, 5, 4], [9, 8, 7]])
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### if_value(value)
*Returns the value if the value passes a truthy test*
```python
>>> if_value(1)
1
>>> if_value([])
>>> if_value([1, 2, 3])
[1, 2, 3]
```

### gt(x, *args)
*Returns whether each argument is strictly greater than each subsequent argument*
```
>>> gt(3, 2, 1)
True
>>> gt(1, 2, 3)
False
>>> gt(3, 1, 2)
False
>>> gt(1, 1, 1)
False
```

### lt(x, *args)
*Returns whether each argument is strictly less than each subsequent argument*
```
>>> lt(3, 2, 1)
False
>>> lt(1, 2, 3)
True
>>> lt(3, 1, 2)
False
>>> lt(1, 1, 1)
False
```

### gte(x, *args)
*Returns whether each argument is greater than or equal to each subsequent argument*
```
>>> gte(3, 2, 1)
True
>>> gte(1, 2, 3)
False
>>> gte(3, 1, 2)
False
>>> gte(1, 1, 1)
True
```

### lte(x, *args)
*Returns whether each argument is less than or equal to each subsequent argument*
```
>>> lte(3, 2, 1)
False
>>> lte(1, 2, 3)
True
>>> lte(3, 1, 2)
False
>>> lte(1, 1, 1)
True
```

### sort_by(f, col, comp=None)
*Sorts a collection by the results of a function applied to each element. A custom comparator
can be provided, otherwise the default comparator will be used. The comparator accepts 2 arguments
and returns*
```python
>>> sort_by('a', [{'a': 1}, {'a': 5}, {'a': 2}, {'a': 7}])
[{'a': 1}, {'a': 2}, {'a': 5}, {'a': 7}]
>>> sort_by(identity, [1, 5, 2, 7, 3, 8, 3, 8], comp=gt)
[1, 2, 3, 3, 5, 7, 8, 8]
```

### sort_ascending(col)
*Sorts a collection of numbers in ascending order*
```python
>>> sort_asending([1, 5, 2, 7, 3, 8, 3, 8])
[1, 2, 3, 3, 5, 7, 8, 8]
```

### sort_descending(col)
*Sorts a collection of numbers in descending order*
```python
>>> sort_descending([1, 5, 2, 7, 3, 8, 3, 8])
[8, 8, 7, 5, 3, 3, 2, 1]
```

### utf8_encode(s)
*Converts a string to bytes using utf-8 encoding*
```python
>>> utf8_encode("abc")
b'abc'
```

### utf8_decode(s)
*Decodes bytes to a string using utf-8*
```python
>>> utf8_decode(b'abc')
"abc"
```

### base64enocde(s)
*Base 64 encodes a string*


### repeat(n, v)
*Repeats a value `v` `n` times*
```python
>>> repeat(5, "Hello")
['Hello', 'Hello', 'Hello', 'Hello', 'Hello']
```

### repeatedly(n, f)
*Repeatedly calls a function `f` `n` times*
```python
>>> repeatedly(5, lambda: "Hello")
['Hello', 'Hello', 'Hello', 'Hello', 'Hello']
```

### average(l)
*Returns the average of the list of numbers*
```python
>>> average([1, 2, 3, 4])
2.5
```

### find(f, c)
*Returns filtered collection by the given function*
```python
>>> find(lambda x: x >= 3, [1, 2, 3, 4])
[3, 4]
```

