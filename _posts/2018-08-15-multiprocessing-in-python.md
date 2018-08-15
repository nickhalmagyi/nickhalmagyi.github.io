---
layout: post
title: "Multiprocessing in Python and Avoiding Copy-on-Write"
categories: machine-learning, python
toc: true
---

* TOC
{:toc}

Peering into my data-science crystal ball I see that the future holds a great wealth of CPU's for everybody. 
I have kubernetes instances at work with 64 CPU's, how about you?
This makes for some serious parallelization and the [multiprocessing](https://docs.python.org/3.7/library/multiprocessing.html)
Python library is a great place to start. Of course, others have also queried this same crystal ball, some folks are even so 
enterprising that they have written a clean multiprocessing wrapper for numpy/pandas/Scikit-learn: 
meet [Dask](https://dask.pydata.org/en/latest/). However to integrate multiprocessing beyond
these standard and excellent Python libraries, I think one must utilize something like the multiprocessing library.
More generally, understanding a little bit of parallelization is a worthy notch in one's belt. 
 
There are quite a few resources out there to learn about multiprocessing in Python, I strongly recommend people 
read up about processes vs threads and the GIL but here I am going to be pretty modest and offer basically my
understanding of [copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write) in Python. The goal is to parallelize some computations which
read from, but do not write to, a large in-memory object without triggering the operating system to create multiple
copies of this object. 
 
I'm going to assume that you the reader are working on Linux or OSX, if you are
on a windows box, you have my deepest sympathies and very little of the following will be of any help to you.
At the bottom of the page, I've collected some references which I have found useful.

But first, something to get you in the mood...

# Gratuitous Metaphor
Imagine if you will, a happy couple blessed with three delightful children and for the upcoming family vacation, 
they generously bought an Ipad for them to share. 

**Kids:** Wow, thanks Mum and Dad, you're the best!

However before you've left the city walls, these three
little monsters in the back seat have erupted into an full-scale, apocalyptic meltdown because they cannot agree which 
Ariana Grande video to watch next. Each child is grabbing at the Ipad and clicking on a new video clip, its totally out of control.

**Parent 1**: Oh no, this family holiday is looking really grim!  
**Parent 2**: Dang it, lets just buy them one each, that'll keep em quiet.

Swerving across several lanes of holiday traffic, they floor it to the nearest electrical retailer and buy two more Ipads.

**Kids:** Seriously you guys rock! Best parents ever! We gonna behave real good, promise!  

A tenuous peace returns to the family unit. 

But wait oh no, they have just pulled up the toll booth on the autoroute and their credit card
has been declined!! Holy crap, they must have maxed out on those damn Ipads!! 

Too bad, this family holiday is now doomed. Best parents ever huh?

So there's a life  lesson here for everybody: there's no need to buy an Ipad each (then you would have to have headphones 
for each etc.), just buy one Ipad and these little terrors can all watch the same Ariana Grande clip. 
Nobody changes the channel until we stop the car.
Any child who complains can get out
and walk. "Understand kids?".... "I said DO YOU UNDERSTAND!!!?"

**Kids**: meep

Which brings us to the central topic of this post....
 
 
# Copy-On-Write
 

Modern computers have multiple CPU's available to them but many languages, Python included, will by
default only use one CPU at a time. Using N CPU's simultaneously to make (N-1)-child processes should 
decrease computation time by roughly a factor
of N. So in this fantasy world of computing, having children reduces processing time. Cool, let's make some kids! 
 
The fly in the ointment is that these child processes
are more or less designed to be completely independent, to have their own independent memory. And if there's one thing we
don't like in this family, it's independent thought.

It would sure be mighty convenient if somehow children could share memory, 
this is where I'm headed. This is not at all an original train of thought, 
there is a long back-story to shared memory management in paraellized computation,
I am merely offering here some understanding and a few tricks to a fairly run-of-the-mill user of Python.


A critical aspect of a decent operating system is [copy-on-write](https://en.wikipedia.org/wiki/Copy-on-write). 
In a nutshell, the idea is that multiple processes
can simultaneously *read* from the same file in memory but when any process attempts to *write* to this file, 
the operating system will immediately start to make a copy of this file. This of course is entirely reasonable, 
and one can find a great explanation by an actual expert, including some historical aspects in this 
[talk](https://www.youtube.com/watch?v=twQKAoq2OPE) by Brandon Rhodes. 

## Copy-On-Write in Python

Now suppose you have a large file in memory of several GB, let's call it *large_file* and you
have a function which accepts *large_file* as an argument. You wish to parallelize a computation which 
involves this function. In any default implementation of this in Python, the operating system will start to make
multiple copies of *large_file*, thus ruining your hard-earned speedup from parallelization and perhaps exhausting your
memory. The reason that 
Python is particularly sensitive to copy-on-write is that each Python object carries around metadata in the form of a 
[reference-count](https://docs.python.org/3/reference/datamodel.html#objects-values-and-types). So just by changing the 
reference-count of *large_file* 
```python
large_file2 = large_file
```
one would trigger an invocation of copy-on-write.

Just in case it's not bleedingly obvious, I now spell out my rather belaboured analogy:  
<p style="text-align: center;"> <br>
parent &#8596; original process  <br>
kids &#8596; child processes  <br>
ipad &#8596; large in-memory object <br>  
parents running out of money &#8596;system running of of memory  <br>
</p>


# Chunking a local process
So let's outline a basic task to be performed. Say we have a numpy array and like the reputable data scientists that 
we all are, we would like to perform some operation on it by iterating over the array. 
I'm going to assume that you
have settled on a job you wish to perform on this array and I will also assume that this job
is *local*. This locality means that it is performed element by element: the operation on a single element 
is independent of other elements in the array. 
If for some reason the operation you wish to perform depended on more global properties of the array, 
it could be tricky or impossible to parallelize the computation. 

We are then going to chunk the array, which basically means break it into a number of pieces, 
each of roughly the same size and simultaneously perform the job on each piece. How many chunks should we make?
We should make as many chunks as you have CPUs....minus one (for the system to use). If you set up a number of processes
larger than the number of CPUs,  you will likely experience some slowdown as the extra processes you have setup
will only commence after some of the initial processes are finished and some CPUs are made availble.

# The Pool class: map and apply

So let's get to it. We first import the stuff we need, and set the number of processes. I will implicitly include this
with all code below:

```python
import numpy as np
import time
import multiprocessing as mp
NBR_PROCESSES = mp.cpu_count()-1 
```
I am working on a MacBook Air with 8Gb RAM and NBR_PROCESSES=3.

We will work with a numpy array of random entries and see if we can speedup a trivial job. I first chunk the array 
using the numpy method [array_split](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.array_split.html)
```python
num_rows = 10**5
arr = np.random.randint(10**3, size=(num_rows,1))
arr_split = np.array_split(arr, NBR_PROCESSES)
```
then I implement a particularly trivial function: for each element in the array, 
it just introduces a certain pause and finally returns the array untouched:
```python
seconds=0.0001
def sleepy_time(arr, seconds=seconds):
    [time.sleep(seconds) for _ in arr]
    return arr
```
Importantly this job is *local*, it operates only on one element of the array at a time and is thus parallelizable.

Now the multiprocessing library gives two ways to implement a processing-pool: *map* and *apply* (and their funky async sisters).
This is how we implement *map*:
```python
def mp_pool_map(func, chunks):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = pool.map(func, chunks)
    pool.close()
    pool.join()
    return out
def mp_pool_map_async(func, chunks):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = pool.map_async(func, chunks)
    pool.close()
    pool.join()
    return out.get()
```
The drawback is that within this approach *func* should not accept multiple arguments, one could zip multiple arguments 
into a tuple but this would require mildly refactoring your single-process code. On the upside the order of the 
results will allign with the ordering of *chunks*, a desirable (or crucial) property missing from *apply*. I have seen 
quite often that people omit the *.join* and *.close*, which from my experience leads to the parental process becoming
entirely unhinged, and nobody wants that.

For *apply* we have
```python
def mp_pool_apply(func, chunks, *args):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = [pool.apply(func, args=(chunk, *args)) for chunk in chunks]
    pool.close()
    pool.join()
    return out
def mp_pool_apply_async(func, chunks, *args):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = [pool.apply_async(func, args=(chunk, *args)) for chunk in chunks]
    output = [p.get() for p in out]
    pool.close()
    pool.join()
    return output
```
With *apply*, we can utilize multiple arguments but the order of the results are NOT alligned with that of *chunks*.

Here are the timings: first we time the single-process computation
```python
%timeit sleepy_time(arr, seconds)
>>> 1 loop, best of 3: 5.14 s per loop
```
and here are the timings of the parallelized computations:
```python
%timeit  mp_pool_map(sleepy_time, arr_split)
%timeit  mp_pool_map_async(sleepy_time, arr_split)
%timeit mp_pool_apply(sleepy_time, arr_split, seconds)
%timeit mp_pool_apply_async(sleepy_time, arr_split, seconds)
out: 1 loop, best of 3: 1.58 s per loop
out: 1 loop, best of 3: 1.49 s per loop
out: 1 loop, best of 3: 4.16 s per loop
out: 1 loop, best of 3: 1.46 s per loop
```
Except for *mp_pool_apply*, the parallelized computations give 
a speedup of about a factor of three and we can declare a multiprocessing victory!
My results for *mp_pool_apply* appear to agree with comments I found 
[elsewhere](http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async).


## Memory Profiling
### Altering the reference-count
I would like to display something explicit to show how the copy-write is being invoked. Profiling memory can be a tricky 
task since in all operating systems, there are various forms of memory and one needs to be somewhat precise in order to
 get a sensible answer. In addition, in Python one cannot easily access the precise memory address of an object since the data
 store is more complicated than just a simple pointer. 
 
 Im going to use the [memory_profiler](https://pypi.org/project/memory_profiler/) library
which has a 
[multiprocessing method](https://bbengfort.github.io/observations/2017/03/20/contributing-a-multiprocess-memory-profiler.html).
This is a script I call *mp_map_async.py*
```python
num_rows = 3*10**7
arr = np.random.randint(10**3, size=(num_rows,1))
arr_split = np.array_split(arr, NBR_PROCESSES)

def replace_from_array(arr):
    time.sleep(0.1)
    arr1 = arr[0]

def mp_pool_map_async(func, chunks):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = pool.map_async(func, chunks)
    pool.close()
    pool.join()
    return out.get()

if __name__ == '__main__':
    arr = mp_pool_map_async(replace_from_array, arr_split)
```
Note that the method *replace_from_array* will increase by one the reference-count of the numpy array which the variable
*arr* is pointing to. This change in the reference count should trigger copy-on-write. I have included a small sleep
because of some minor plotting issue in the next section.

I call this script with 
```bash
$ mprof run -M python3 mp_a pply_async.py
```
which creates a *.dat* file in the working directory. One can either inspect this file visually
or plot it with 
```bash
$ mprof plot
```
giving the folloing output 
![mp_map_async]({{ "/assets/mp_map_async.png" | absolute_url }})
So this image captures a lot of imformation.
1. The parent process has a lot of overhead
2. Each subprocess take a while to load a copy of the array, this is due to copy-on-write being triggered.
3. Each subsequent process is delayed by about a second while it
waits for the array to be loaded into the previous process. 
4. I suspect the measurement of the memory usage of the parent process is including the sum of the 
memory usage child-processes...
to be confirmed.

### Not altering the reference-count

We can run a similar script which does not alter the reference count and does not trigger a copy-on-write
```
num_rows = 3*10**5
arr = np.random.randint(10**3, size=(num_rows,1))
arr_split = np.array_split(arr, NBR_PROCESSES)

seconds=10**(-5)
def sleepy_time(arr, seconds=seconds):
    [time.sleep(seconds) for _ in arr]
    return arr
    
def mp_pool_map_async(func, chunks):
    pool = mp.Pool(processes=NBR_PROCESSES)
    out = pool.map_async(func, chunks)
    pool.close()
    pool.join()
    return out.get()

if __name__ == '__main__':
    arr = mp_pool_map_async(sleepy_time, arr_split)
```    
Which gives the following output
![mp_map_sleepy]({{ "/assets/mp_map_sleepy.png" | absolute_url }})
There is no delay from copying the array


# Avoiding copy-on-write with a manager

Now here is a method which avoids copy-on-write which invokes a manager class. A manager is like a 
[helicopter parent](https://www.urbandictionary.com/define.php?term=helicopter%20parent), 
which may sometimes be considered a 
pejorative but not here. This is *mp_manager.py*
```python
num_rows = 3*10**7
arr = np.random.randint(10**3, size=(num_rows,1))
arr_split = np.array_split(arr, NBR_PROCESSES)

def replace_from_array(arr):
    time.sleep(0.2)
    arr1 = arr[0]

def mp_func(func, chunks, *args, procnums=None):
    
    def mp_func_chunk(func, chunk, out_dict, procnum, *args):
        out_dict[procnum] = func(chunk, *args)

    mgr = mp.Manager()
    out_dict = mgr.dict()
    
    if not procnums: 
        procnums = len(chunks)

    procs = []
    for i in range(procnums):
        p = mp.Process(target=mp_func_chunk, args=(func, chunks[i], out_dict, i, *args))
        procs.append(p)
        p.start()
    for proc in procs:
        proc.join()
        
    out = [out_dict[key] for key in sorted(out_dict.keys())]
    return out

if __name__ == '__main__':
    arr = mp_func(replace_from_array, arr_split)
```
This mp_func routine delivers sorted results in the same order as the input and allows for multiple arguments. To demonstrate that
it is indeed creating the expected number of processes we have:
```python
%timeit mp_func(sleepy_time, arr_split, seconds)
out: 1 loop, best of 3: 1.31 s per loop
```
## Memory Profiling
To profile the memory usage we call
```bash
$ mprof run -M python3 mp_manager.py
```
and view the following output
![mp_func_replace]({{ "/assets/mp_func_replace.png" | absolute_url }})
We can see that indeed there is no copying of the array being called.


# Wrapping Up
Multiprocessing aint going away, one should really stay on top of it and quite possibly this should involve integrating 
Dask to your workflow. I found it quite tricky to really nail down
exactly what copy-on-write is doing, this post is the result of me trying to make sense of this rather mysterious
concept.  I hope it offers at least some incremental progress for the regular Pythonistas out there. 
Using a manager you can easily avoid copy-on-write and create as many children as you can afford. 


<p style="text-align: center;">
Happy procreating!
</p>







# References
Anything I have correctly learnt on this topic has come from chit-chats with 
[Brookie Williams](https://github.com/brookisme) and [Sheer El-Showk](https://www.lore.ai/)
and the following interesting materials:
* Chapter 9 of [High Performance Python](http://shop.oreilly.com/product/0636920028963.do) by Gorelick-Ozsvald
* the response of *vartec* on [this stackoverflow thread](https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce)  

Anything I have misunderstood is my own folly.
