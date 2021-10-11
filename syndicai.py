from source.generate_transformers import main, parse_args, get_model

class PythonPredictor:
    def __init__(self, config):
        parser = parse_args()
        args = parser.parse_args(['--model_type', 'gpt2', 
                                  '--model_name_or_path', './model',
                                  '--k', '50',
                                  '--p', '0.95',
                                  '--length', '100',
                                  '--repetition_penalty', '5',
                                  '--num_return_sequences', '1',
                                  '--no_cuda'])
        model, tokenizer = get_model(args)
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, payload):
        self.args.prompt = payload['text']
        output_text = main(self.model, self.tokenizer, self.args)
        return output_text