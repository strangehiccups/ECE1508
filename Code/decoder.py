import torch
import torch.nn as nn
import transformers
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import CTCDecoderLM

class Decoder(nn.Module):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 blank_token: str,
                 lexicon: str=None,
                 lm: CTCDecoderLM=None,
                 nbest: int=1,
                 beam_size: int=50):
        self.tokenizer = tokenizer
        self.tokens = list(self.tokenizer.get_vocab().keys())
        self.blank_token = blank_token
        self.blank_token_id = self.tokenizer.convert_tokens_to_ids(self.blank_token)
        self.lexicon = lexicon
        self.lm = lm
        self.nbest = nbest
        self.beam_size = beam_size
        self.decoder = ctc_decoder(lexicon=self.lexicon,
                                   tokens=self.tokens,
                                   blank_token=self.blank_token,
                                   lm=self.lm,
                                   nbest=self.nbest,
                                   beam_size=self.beam_size)

    # Greedy CTC decode: argmax at each timestep, collapse repeats, remove blanks
    def decode_greedy(self,
                      specs: torch.Tensor, # (B, C, F, T)
                      spec_lengths: torch.Tensor, # (B,)
                      model: nn.Module) -> str:
        model.eval()
        with torch.no_grad():
            probs, _ = model(specs, spec_lengths)
        pred_ids = probs.argmax(dim=-1)
        # remove repeated and blank tokens
        collapsed = torch.unique_consecutive(pred_ids)
        collapsed = collapsed[collapsed != self.blank_token_id]
        return self.tokenizer.decode(collapsed)

    def decode_beam(self,
                    specs: torch.Tensor, # (B, C, F, T)
                    spec_lengths: torch.Tensor, # (B,)
                    model: nn.Module) -> str:
        model.eval()
        with torch.no_grad():
            probs, out_lens = model(specs, spec_lengths)
        results = self.decoder(probs, out_lens)
        # token ids to text
        batch_text = []
        for sample in results:
            best_sample = sample[0] # consider only the best beam
            if self.lexicon is None:
                token_ids = best_sample.tokens
                text = ''.join(self.tokens[i] for i in token_ids)
            else:
                text = ''.join(best_sample.words)
            batch_text.append(text)
        return batch_text