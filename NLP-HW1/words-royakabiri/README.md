# Objectives

The learning objectives of this assignment are to:
1. practice Python programming skills
2. get familiar with submitting assignments on GitHub Classroom

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [scikit-learn (version 0.20 or higher)](http://scikit-learn.org/)
* [pytest](https://docs.pytest.org/)

If you have not used Git, Python, Numpy, or scikit-learn before, this would be a
good time to go through some tutorials:

* [git tutorial](https://try.github.io/)
* [Python tutorial](https://docs.python.org/3/tutorial/)
* [numpy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
* [scikit-learn tutorial](http://scikit-learn.org/stable/tutorial/index.html)

You can find many other tutorials for these tools online.

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.
This is the first step of the
[standard GitHub workflow](https://guides.github.com/introduction/flow/):

1. Create a branch
2. Add commits
3. Open a Pull Request
4. Discuss and review your code

In this class, the first three steps are your responsibility;
the fourth step is how your instructor will grade your assignment.
Note that you *must create a branch* (you cannot simply commit to master) or you
will not be able to create a pull request as required for grading.

First, go to the repository that GitHub Classroom created for you,
`https://github.com/UA-LING-439-SP19/words-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.

Then, clone the repository to your local machine and checkout the branch you
just created:
```
git clone -b <branch> https://github.com/UA-LING-439-SP19/words-<your-username>.git
```
where `<branch>` is whatever you named your branch.
You are now ready to begin working on the assignment.

# Write your code

You will implement a module-level method, `most_common`, and a few methods of a
class, `WordVectors`.
A template has been provided for you in the file `words.py`.
In the template, each method has only a documentation string, with no code in
the body of the method yet.
For example, the `most_common` method looks like:
```python
def most_common(word_pos_path: Text,
                word_regex=".*",
                pos_regex=".*",
                n=10) -> List[Tuple[Text, int]]:
    """Finds the most common words and/or parts of speech in a file.

    :param word_pos_path: The path of a file containing part-of-speech tagged
    text. The file should be formatted as a sequence of tokens separated by
    whitespace. Each token should be a word and a part-of-speech tag, separated
    by a slash. For example: "The/at Hartsfield/np home/nr is/bez at/in 637/cd
    E./np Pelham/np Rd./nn-tl Aj/nn ./."

    :param word_regex: If None, do not include words in the output. If a regular
    expression string, all words included in the output must match the regular
    expression.

    :param pos_regex: If None, do not include part-of-speech tags in the output.
    If a regular expression string, all part-of-speech tags included in the
    output must match the regular expression.

    :param n: The number of most common words and/or parts of speech to return.

    :return: A list of (token, count) tuples for the most frequent words and/or
    parts-of-speech in the file. Note that, depending on word_regex and
    pos_regex (as described above), the returned tokens will contain either
    words, part-of-speech tags, or both.
    """
```
Right below the documentation string of each method, you should write code to
implement the method as it has been defined by the documentation.

The following objects and methods may come in handy:
* [collections.Counter]()
* [re.match]()
* [numpy.fromstring]()
* [numpy.average]()
* [sklearn.metrics.pairwise.cosine_similarity]()
* [heapq.nlargest]()

# Test your code

Tests have been provided for you in the `test_words.py` file.
The tests show how each of the methods are expected to be used.
For example, the `test_most_common_tokens` test method looks like:
```python
def test_most_common_tokens():
    assert words.most_common("brown_sample.txt", n=2) == [
        ('the/at', 5558),
        (',/,', 5133),
    ]
```
This tests that your code correctly counts tokens in the `brown_sample.txt`
text file and identifies the top two most frequent tokens and their counts.


To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_words.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.6.5, pytest-3.8.1, py-1.6.0, pluggy-0.7.1
rootdir: .../words, inifile:
collected 8 items                                                              

test_words.py FFFFFFFF                                                   [100%]

=================================== FAILURES ===================================
___________________________ test_most_common_tokens ____________________________

    def test_most_common_tokens():
>       assert words.most_common("brown_sample.txt", n=2) == [
            ('the/at', 5558),
            (',/,', 5133),
        ]
E       AssertionError: assert None == [('the/at', 5558), (',/,', 5133)]
E        +  where None = <function most_common at 0x106d82268>('brown_sample.txt', n=2)
E        +    where <function most_common at 0x106d82268> = words.most_common

test_words.py:9: AssertionError
...
=========================== 8 failed in 0.18 seconds ===========================

```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.6.5, pytest-3.8.1, py-1.6.0, pluggy-0.7.1
rootdir: .../words, inifile:
collected 8 items                                                              

test_words.py ........                                                   [100%]

=========================== 8 passed in 2.63 seconds ===========================
```

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.
Make a habit of checking the GitHub page for your repository to make
sure your changes have been correctly pushed there.
You may also want to check the "commits" page of your repository on GitHub:
there should be a green check mark beside your last commit, showing that your
code passes all of the given tests.
If there is a red X instead, your code is still failing some of the tests.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).
where the "base" branch is "master", and the "compare" branch is the branch you
created at the beginning of this assignment.
Then go to the "Files changed" tab, and make sure that you have only changed
the `words.py` file and that all your changes look as you would expect them to.
**Do not merge the pull request.**

Your instructor will grade the code of this pull request, and provide you
feedback in the form of comments on the pull request.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all tests will receive at least 80% of the possible
points.
To get the remaining 20% of the points, make sure that your code is using
appropriate data structures, existing library functions are used whenever
appropriate, code duplication is minimized, variables have meaningful names,
complex pieces of code are well documented, etc.
