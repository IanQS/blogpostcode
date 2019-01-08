# High Level Goals:

1) Give you an example of a productionized pipeline where one "machine" is creating the datasets, and another is consuming them

2) Give you an idea of how you can convert your numpy/ text files into `TfRecords`

* note, there already exist TONs of resources on directly feeding `np` arrays into your graph so I'm not going into it

3) Discuss the advantages and disadvantages of using this method over something more "out of the box"

# Long-term Goal:

Integrate knowledge from this exploration into [Sequences Tutorial](https://ianqs.github.io/tag/Sequences) by creating a production-ready pipeline