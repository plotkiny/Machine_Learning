#!usr/bin/env/python


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(model, output_dir, perplexity=40, n_components=2, init='pca', n_iter=2500):

    """
    Creates and TSNE model and plots it"

    :param model: Pickled file of a dictionary of word-embeddings (Keys=word. Values=numpy high dimensional embedding).
    :param output_dir: Output directory to save the T-SNE figure
    :return: T-SNE plot of the words in a high-dimensional space
    """

    labels, tokens = [], []

    for word in model:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x, y = [], []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    plt.title('T-SNE Plot of the word-embeddings of the most common words')
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    plt.savefig(output_dir)




