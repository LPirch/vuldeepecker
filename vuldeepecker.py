"""
Interface to VulDeePecker project
"""
import sys
import os
from pathlib import Path
import pandas
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
from blstm import BLSTM
from rich.progress import track

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is a binary value (label) and a single word (string)
    This indicates whether or not there is vulnerability in that gadget
    and to which split it belongs

"""
def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        yield from parse_stream(file)


"""
Schematic gadget:

[number] [additional_info]
code line 1
code line 2
...
[label] [split]
--------------------------------

"""
def parse_stream(stream):
    gadget = []
    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue
        if gadget and stripped == "-"*33+"<EOF>":
            label = int(gadget[-1])
            gadget = gadget[:-1]
            yield clean_gadget(gadget), label
            gadget = []
        else:
            gadget.append(stripped)

"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
"""
def get_vectors_df(gadget_stream, vector_length=100):
    gadgets = []
    vectorizer = GadgetVectorizer(vector_length)
    print("Collecting gadgets...")
    for gadget, val in gadget_stream:
        vectorizer.add_gadget(gadget)
        row = {"gadget" : gadget, "val" : val}
        gadgets.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print("Training model...", end="\r")
    vectorizer.train_model()
    vectors = []
    for gadget in track(gadgets):
        vector = vectorizer.vectorize(gadget["gadget"])
        row = {"vector" : vector, "val" : gadget["val"]}
        vectors.append(row)
    df = pandas.DataFrame(vectors)
    return df


def process_data(gadgets, vector_filename, vector_length=100):
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(gadgets, vector_length)
        df.to_pickle(vector_filename)
    return df["vector"].values, df["val"].values

"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main():
    if len(sys.argv) != 7:
        print("Usage: python vuldeepecker.py [train_file] [val_file] [test_file] [cache_dir] [result_dir] [seed]")
        exit()
    train_file = Path(sys.argv[1])
    val_file = Path(sys.argv[2])
    test_file = Path(sys.argv[3])
    cache_dir = Path(sys.argv[4])
    result_dir = Path(sys.argv[5])
    seed = int(sys.argv[6])

    vector_length = 50
    # TODO subsample 90% of training data
    data = {
        "train": process_data(parse_file(train_file), cache_dir / "train_gadget_vectors.pkl", vector_length),
        "val": process_data(parse_file(val_file), cache_dir / "val_gadget_vectors.pkl", vector_length),
        "test": process_data(parse_file(test_file), cache_dir / "test_gadget_vectors.pkl", vector_length),
    }
    blstm = BLSTM(data["train"][0], data["train"][1], data["test"][0], data["test"][1], result_dir, name="vuldeepecker-blstm", subsample_train=0.9, seed=seed)
    blstm.train()
    blstm.test()

if __name__ == "__main__":
    main()