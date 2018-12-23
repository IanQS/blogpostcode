# High Level Goals:

1) Give you an example of a productionized pipeline where one "machine" is creating the datasets, and another is consuming them

2) Give you an idea of how you can convert your numpy/ text files into `TfRecords`

* note, there already exist TONs of resources on directly feeding `np` arrays into your graph so I'm not going into it

3) Discuss the advantages and disadvantages of using this method over something more "out of the box"

4) Discuss when you might want to pre-store 

# Long-term Goal:

Integrate knowledge from this exploration into [Sequences Tutorial](https://ianqs.github.io/tag/Sequences) by creating a production-ready pipeline



## Note:

1) This doesn't cover `tf.Estimator` pipelines - mostly because I don't know how those work/ what goes into them. I may update it in the future to account for it

2) This is a non-traditional view of pipelines where it is 2 pipelines running in tandem but in my experience better modularization leads to better parallelization (across teams)

- just add asserts on both ends / use [Abstract Base Classes](https://docs.python.org/3/library/abc.html) to ensure that the interface is respected