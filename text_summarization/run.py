#!usr/bin/env/python

import argparse
import run_nlp
import run_model

class CreatePipeline(object):

    def __init__(self, config, output):
        self.config = config
        self.output = output

    def nlp(self):
        run_nlp.main(self.config, self.output)

    def model(self, type):
        run_model.main(self.config, self.output, type)

    def build(self, predict=False):
        run_nlp.main(self.output)
        run_model.main(self.config, self.output, 'train')

        if predict:
            run_model.main(self.config, self.output, 'predict')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments for running text summarization')
    parser.add_argument('-con', '--config', required=True, help="Jsoon dictionary configuration file")
    parser.add_argument('-out', '--output', required=True, help='Specify the output file path')
    parser.add_argument('-a','--all', action='store_true', help='Run both NLP text pre-processing and Seq2Seq model')
    parser.add_argument('-prep', '--pre_process', action='store_true', help="Perform NLP text pre-processing")
    parser.add_argument('-t', '--train', action='store_true', help='Run both NLP text pre-processing and Seq2Seq model')
    parser.add_argument('-p','--predict', action='store_true', help='Use Seq2Seq model neural network on processed text')
    args = parser.parse_args()

    print('output {}'.format(args.output))
    print('all {}'.format(args.all))
    print('pre-process {}'.format(args.pre_process))
    print('train {}'.format(args.train))
    print('predict {}'.format(args.predict))

    #check some conditions
    if not args.output:
        parser.error("please provide a path to save the output ")

    elif args.train and args.predict:
        parser.error("you can either train or predict")

    elif args.all and any([args.train == True and args.predict == True]):
        parser.error("do you want to run the entire pipeline or just train/predict")

    #instantiate the text_summarization class
    text_summarization = CreatePipeline(args.config, args.output)

    #logic for processing the input arguments
    if args.pre_process and all([args.train == False and args.predict == False]):
        text_summarization.nlp()

    elif args.all:
        text_summarization.build()

    elif args.train and not args.pre_process:
        type = 'train'
        text_summarization.model(type)

    elif args.predict and not args.pre_process:
        type = 'predict'
        text_summarization.model(type)

    elif args.train and args.pre_process:
        text_summarization.build()


