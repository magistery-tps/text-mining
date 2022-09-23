from transformers import BertTokenizer



def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


class Tokenizer:
    def __init__(
        self, 
        model='bert-base-cased',

        # To pad each sequence to the maximum length that you specify.
        padding        = 'max_length',

        # Maximum length of each sequence. In this example we use 10, 
        # but for our actual dataset we will use 512, which is the 
        # maximum length of a sequence allowed for BERT.
        max_length     = 10, 

        # If True, then the tokens in each sequence that exceed the 
        # maximum length will be truncated.
        truncation     = True,

        # the type of tensors that will be returned. Since we’re 
        # using Pytorch, then we use pt. If you use Tensorflow, 
        # then you need to use tf .
        return_tensors = "pt"
    ):
        self.__tokenizer      = BertTokenizer.from_pretrained(model)
        self.__padding        = padding
        self.__max_length     = clamp(0, max_length, 512)
        self.__truncation     = truncation
        self.__return_tensors = return_tensors

    def tokenize(self, text):
        return self.__tokenizer(
            text,
            padding        = self.__padding,
            max_length     = self.__max_length,
            truncation     = self.__truncation,
            return_tensors = self.__return_tensors
        )

    def to_text(self, bert_input):
        return self.__tokenizer.decode(bert_input.input_ids[0])