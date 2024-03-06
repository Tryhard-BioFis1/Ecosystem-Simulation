# TIME COMPLEXITY

## Lists:
> Lists are a fundamental data structure in Python that allows storing and manipulating a collection of items. The time complexity for common list operations can be summarized as follows:

* Accessing an element by index: **O(1)**
* Inserting or deleting an element at the end: **O(1)**
* Inserting or deleting an element at the beginning or middle: **O(n)**
* Searching for an element: **O(n)**
* Sorting: **O(n log n)**

## Dictionaries:
> Dictionaries provide an efficient way to store and retrieve key-value pairs. They are implemented using a hash table, which makes their time complexity highly dependent on the quality of the hash function. The average and worst-case time complexities for dictionary operations are as follows:

* Accessing an element by key: **O(1)** average case, **O(n)** worst case
* Inserting or deleting an element: **O(1)** average case, **O(n)** worst case
* Searching for a key: **O(1)** average case, **O(n)** worst case

## Sets:
> Sets are unordered collections of unique elements in Python. They are particularly useful when we need to perform membership tests or eliminate duplicates. The time complexity for common set operations is as follows:

* Accessing an element: Not applicable
* Inserting or deleting an element: **O(1)** average case, **O(n)** worst case
* Searching for an element: **O(1)** average case, **O(n)** worst case

## Tuples:
> Tuples are similar to lists but are immutable, meaning they cannot be modified after creation. Their time complexity is comparable to that of lists.

* Accessing an element by index: **O(1)**
* Inserting or deleting an element: Not applicable (immutable)
* Searching for an element: **O(n)**

# SPACE COMPLEXITY 

## Lists, Tuples, and Sets:
The space complexity for lists, tuples, and sets is generally **O(n)**, where n represents the number of elements in the data structure.

## Dictionaries:
The space complexity for dictionaries varies depending on the number of elements and the load factor (the ratio of occupied slots to the total number of slots). In the **average case**, the space complexity is **O(n)**, while in the **worst case**, it can be as high as **O(2n)** due to the need for handling collisions.

Source: https://www.linkedin.com/pulse/demystifying-python-data-structure-time-space-complexity-deepak-s
