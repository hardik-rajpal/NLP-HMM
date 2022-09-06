tagger.py contains all the code.

python (3.10 used) packages required:
    numpy
    matplotlib
    re (regex expressions)
    nltk (and brown corpus downloaded, otherwise follow instructions in error message.)

Notes:
1. Every time the tagger is trained, it gets saved into txt and npy files locally.
We can then use the loadTagger() function next time to avoid spending time on training.
2. To test a particular sentence on the tagger:
t = Tagger.getTrainedModel()
t.demoSent('The boy jumped over the wall.')