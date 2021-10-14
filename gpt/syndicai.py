import gdown
import logging
from os import listdir
from source.generate_transformers import main, parse_args, get_model

class PythonPredictor:
    def __init__(self, config):
        parser = parse_args()
        args = parser.parse_args(['--model_type', 'gpt2', 
                                  '--model_name_or_path', './model/',
                                  '--k', '50',
                                  '--p', '0.95',
                                  '--length', '100',
                                  '--repetition_penalty', '5',
                                  '--num_return_sequences', '1',
                                  '--no_cuda'])
        
        url = 'https://drive.google.com/uc?id=1TsmlmEMGOVw9ftCbuYt7lxYrvRS31tT8'
        output = '/tmp/model_weights.pth'
        gdown.download(url, output)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        
        logger.info(listdir())
        logger.info(listdir('/tmp/'))

        model, tokenizer = get_model(args, output)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, payload):
        self.args.prompt = payload['text']
        output_text = main(self.model, self.tokenizer, self.args)
        return output_text

# class PythonPredictor:    
    # def __init__(self, config):        
        # pass    
    # def predict(self, payload):        
        # return payload["key"]  # prints "value"