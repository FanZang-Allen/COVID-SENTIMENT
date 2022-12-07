from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
import torch
from fastai import *
from fastai.text import *
from transformers import AdamW
from functools import partial
from urllib.request import urlopen
import twitter
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

defaults.cpus=1

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
        
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids):
        # Return only the logits from the transfomer
        logits = self.transformer(input_ids)[0]   
        return logits

# read training data


def get_data(input_file_name):
    sem_df = pd.read_json(input_file_name, lines=True)
    sem_df.loc[sem_df["label"] == "positive", "sentiment"] = 1
    sem_df.loc[sem_df["label"] == "neutral", "sentiment"] = 0
    sem_df.loc[sem_df["label"] == "negative", "sentiment"] = -1
    
    df = sem_df[['cleaned_text', 'sentiment']].copy()

    return pd.DataFrame(df)


def train_model(train, test):
    model_class, tokenizer_class, config_class = BertForSequenceClassification, BertTokenizer, BertConfig
    model_name = 'bert-base-uncased'
    model_type = 'bert'

    transformer_tokenizer = tokenizer_class.from_pretrained(model_name)
    transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
    fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

    transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
    numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

    tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, 
                                        include_bos=False, 
                                        include_eos=False)
    transformer_processor = [tokenize_processor, numericalize_processor]


    
    pad_idx = transformer_tokenizer.pad_token_id

    databunch = (TextList.from_df(train, cols='cleaned_text', processor=transformer_processor)
                .split_by_rand_pct(0.1,seed=42)
                .label_from_df(cols='sentiment')
                .add_test(test)
                .databunch(bs=16, pad_first=False, pad_idx=pad_idx))

    CustomAdamW = partial(AdamW, correct_bias=False)

    config = config_class.from_pretrained(model_name)
    config.num_labels = 3
    transformer_model = model_class.from_pretrained(model_name, config = config)
    custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

    learner = Learner(databunch, 
                    custom_transformer_model, 
                    opt_func = CustomAdamW, 
                    metrics=[accuracy, error_rate])
    print("================ running lr find ================")
    res = learner.lr_find()
    fig = learner.recorder.plot(skip_end=10,suggestion=True)
    plt.savefig('plt.png')
    plt.show()

    

def main():
    input_file_name='../labeled_data/gold.json'
    df = get_data(input_file_name)
    train, test = train_test_split(df, test_size=0.2, random_state=25)
    train_model(train, test)

if __name__ == '__main__':
    # not really working due to the multithreading problem on windows and fastai properties.
    main()