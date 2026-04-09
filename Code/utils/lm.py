import torch
from torchmetrics.text import WordErrorRate

from config import LM_ARPA_PATH, LEXICON_PATH, LM_WEIGHT, WORD_SCORE


def sweep_lm_params(
    data_loader,
    model,
    device: torch.device,
    lexicon_path: str = LEXICON_PATH,
    lm_path: str = LM_ARPA_PATH,
    lm_weights=None,
    word_scores=None,
):
    # Import here to avoid circular imports and allow the module to load without flashlight
    from decoder import Decoder

    if lm_weights is None:
        lm_weights = [0.5, 1.0, 1.5]
    if word_scores is None:
        word_scores = [-1.0, 0.5]

    model.eval()
    cached_outputs = []
    with torch.no_grad():
        for batch in data_loader:
            specs = batch["padded_spectrograms"].to(device)
            seq_lens = batch["input_lengths"]
            log_probs, out_lens = model(specs, seq_lens)
            cached_outputs.append((log_probs.cpu(), out_lens.cpu()))

    best_wer = float("inf")
    best_params = {"lm_weight": LM_WEIGHT, "word_score": WORD_SCORE}
    results = []

    for lw in lm_weights:
        for ws in word_scores:
            sweep_decoder = Decoder(
                tokenizer=model.tokenizer,
                blank_token=model.tokenizer.pad_token,
                lexicon=lexicon_path,
                lm=lm_path,
                lm_weight=lw,
                word_score=ws,
            )
            wer_metric = WordErrorRate()
            for batch, (log_probs, out_lens) in zip(data_loader, cached_outputs):
                hyps = sweep_decoder.decode_beam(
                    specs=None,
                    spec_lengths=None,
                    model=model,
                    log_probs=log_probs,
                    out_lens=out_lens,
                )
                refs = [
                    model.tokenizer.decode(
                        batch["packed_transcripts"][
                            sum(batch["target_lengths"][:i]):
                            sum(batch["target_lengths"][:i + 1])
                        ].tolist()
                    )
                    for i in range(len(hyps))
                ]
                wer_metric.update(hyps, refs)

            wer = wer_metric.compute().item()
            results.append((lw, ws, wer))
            if wer < best_wer:
                best_wer = wer
                best_params = {"lm_weight": lw, "word_score": ws}
            print(f"lm_weight={lw:.1f} word_score={ws:+.1f} val_WER={wer:.4f}")

    print(f"Best val WER {best_wer:.4f} at {best_params}")
    return best_wer, best_params, results
