import torch
import torch.nn as nn
import transformers

from config import LM_WEIGHT, WORD_SCORE

import importlib
import subprocess
import sys
try:
    importlib.import_module("flashlight")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flashlight-text"])
try:
    importlib.import_module("torchaudio")
except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"])
else:
    from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import CTCDecoderLM

class Decoder(nn.Module):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        blank_token: str,
        lexicon: str=None,
        lm: CTCDecoderLM=None,
        lm_weight: float=LM_WEIGHT,
        word_score: float=WORD_SCORE,
        sil_token: str="|",
        nbest: int=1,
        beam_size: int=50
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens = list(self.tokenizer.get_vocab().keys())
        self.blank_token = blank_token
        self.blank_token_id = self.tokenizer.convert_tokens_to_ids(self.blank_token)
        self.lexicon = lexicon
        self.lm = lm
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.sil_token = sil_token # Required for CTCDecoder but not used since we are doing character-level decoding, so we can set it to any token (e.g. '|') that is not in the lexicon
        self.nbest = nbest
        self.beam_size = beam_size

        # not using cuda_ctc_decoder since it doesn't support lexicon or LM, 
        # but we can still run the regular ctc_decoder on GPU if available
        self.decoder = ctc_decoder(lexicon=self.lexicon,
                                    tokens=self.tokens,
                                    blank_token=self.blank_token,
                                    sil_token=self.sil_token,
                                    lm=self.lm,
                                    lm_weight=self.lm_weight,
                                    word_score=self.word_score,
                                    nbest=self.nbest,
                                    beam_size=self.beam_size)
        

    # Greedy CTC decode: argmax at each timestep, collapse repeats, remove blanks
    def decode_greedy(self,
                      specs: torch.Tensor, # (B, C, F, T)
                      spec_lengths: torch.Tensor, # (B,)
                      model: nn.Module) -> str:
        model.eval()
        with torch.no_grad():
            log_probs, _ = model(specs, spec_lengths)
        pred_ids = log_probs.argmax(dim=-1)
        # remove repeated and blank tokens
        collapsed = torch.unique_consecutive(pred_ids)
        collapsed = collapsed[collapsed != self.blank_token_id]
        return self.tokenizer.decode(collapsed)

    def decode_beam(self,
                    specs: torch.Tensor, # (B, C, F, T)
                    spec_lengths: torch.Tensor, # (B,)
                    model: nn.Module,
                    log_probs: torch.Tensor=None, # cached model output; if provided, skips model forward pass
                    out_lens: torch.Tensor=None) -> str:
        if log_probs is None or out_lens is None:
            model.eval()
            with torch.no_grad():
                log_probs, out_lens = model(specs, spec_lengths)
        if torch.cuda.is_available():
            results = self.decoder(log_probs.cpu(), out_lens.to(torch.int32))
        else:
            results = self.decoder(log_probs.cpu(), out_lens.cpu())
        # token ids to text
        batch_text = []
        for sample in results:
            best_sample = sample[0] # consider only the best beam
            if self.lexicon is None:
                token_ids = best_sample.tokens
                text = ''.join(self.tokens[i] for i in token_ids)
            else:
                text = ' '.join(best_sample.words)
            batch_text.append(text)
        return batch_text