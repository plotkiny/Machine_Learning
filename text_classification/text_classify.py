#!usr/bin/env python

from __future__ import unicode_literals, print_function

import plac
import spacy
from pathlib import Path
from spacy.util import minibatch, compounding
from text_helper import acquire_data
from text_helper import evaluate
from text_helper import generative_model
from text_helper import load_data
from text_helper import nlp_methods


#train CNN model using both class labels
#TODO: change n_texts to percent text
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    file=("Input data", "option", "f", str))
def main(file=None, model=None, output_dir=None, n_iter=20, n_texts=5000):

    if file is None:
        IOError("Please provide an input file")

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    #pre-process the text features
    data = acquire_data(file)
    df = nlp_methods.preprocess_text(data, nlp)
    print("After preprocessing shape {}".format(df.shape))

    #exploratory data analysis
    print("Description")
    #exploratory.describe(df)
    #plot = exploratory.plot_class_balance(df, "class1")

    #TODO: Deal with class imbalance problem ,generative model can't predict the minority class!!
    #train generative model for classifying class1, added predicted class as a feature
    class1_pred = generative_model(df, "text", "class1")
    df.index = range(len(class1_pred))
    df["text"] = df.apply(lambda x: x[0] + " {}".format(class1_pred[x.name]), axis=1)

    #transform data to the right input shapes
    all_classes = list((set(df["class"])))
    class_dict = dict([(v,k) for k,v in enumerate(all_classes)])
    df = df[["text","class"]].replace(class_dict)
    data = [tuple(x) for x in df.values.tolist()]

    #load the data
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(data, class_dict, limit=n_texts)

    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))

    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    #add labels to text classifier
    for i in all_classes:
        textcat.add_label(i)

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    # test the trained model
    test_text = dev_texts[96]
    doc = nlp(test_text)
    mle = max(doc.cats, key=doc.cats.get)
    print("{}: Key: {} Value: {}".format(test_text, mle, doc.cats[mle]))

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)

if __name__ == '__main__':
    plac.call(main)


