
# Backward-compatibility shim.
# New code should import directly from the source modules:
#   from sample import AudioSample, collate_fn
#   from ljspeech import LJSpeechDataset
from utils import AudioSample, collate_fn  # noqa: F401
from ljspeech import LJSpeechDataset        # noqa: F401
