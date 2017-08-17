#!usr/bin/env/python

import argparse, sys
from main.model import run_model
from main.data_processing import run_nlp


class CreatePipeline(object):

    def __init__(self, config, output):
        self.config = config
        self.output = output

    def nlp(self):
        run_nlp.main(self.config, self.output)

    def model(self, type):
        run_model.main(self.config, self.output, type)
        sys.stderr.write('Ended Model {}! \n'.format(type))

    def build(self, predict=False):
        run_nlp.main(self.output)
        run_model.main(self.config, self.output, 'train')
        sys.stderr.write('Ended Model Training. \n')

        if predict:
            run_model.main(self.config, self.output, 'predict')
            sys.stderr.write('Ended Model Prediction. \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments for running text summarization')
    parser.add_argument('-con', '--config', required=True, help="Jsoon dictionary configuration file")
    parser.add_argument('-out', '--output', required=True, help='Specify the output file path')
    parser.add_argument('-a','--all', action='store_true', help='Run both NLP text pre-processing and Seq2Seq model')
    parser.add_argument('-pre', '--pre_process', action='store_true', help="Perform NLP text pre-processing")
    parser.add_argument('-t', '--train', action='store_true', help='Train the Seq2Seq neural network model on processed data.')
    parser.add_argument('-p','--predict', action='store_true', help='Predict sentences using a trained Seq2Seq model.')
    args = parser.parse_args()

    #check some conditions
    if not args.output:
        parser.error('Please provide a path to save the output. \n ')

    elif args.train and args.predict:
        parser.error('You can either train or predict')

    elif all([args.predict == True and args.pre_process == True]):
        parser.error('You cannot both predict and train the model. Please choose one. \n')

    elif args.all and any([args.train == True and args.predict == True]):
        parser.error('Do you want to run the entire pipeline or just train/predict. \n')

    #instantiate the text_summarization class
    text_summarization = CreatePipeline(args.config, args.output)

    #logic for processing the input arguments
    if args.pre_process and all([args.train == False and args.predict == False]):
        text_summarization.nlp()

    elif args.train and not args.pre_process:
        type = 'train'
        text_summarization.model(type)

    elif args.train and args.pre_process:
        text_summarization.build()

    elif args.predict and not args.pre_process:
        type = 'predict'
        text_summarization.model(type)

    elif args.all:
        text_summarization.build()
