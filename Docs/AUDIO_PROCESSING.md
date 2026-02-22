RAW AUDIO FILE (.wav, .mp3, etc.)
        │
        │  librosa.load() — reads binary file, decodes compression,
        │  resamples to target sample_rate
        ▼
WAVEFORM  (time-domain signal)
amplitude
   │   ╭╮    ╭─╮        ╭╮
   │  ╭╯╰╮  ╭╯ ╰╮  ╭╮  ╭╯╰╮
───┼──╯  ╰──╯   ╰──╯╰──╯  ╰───▶ time
   │
   │  Raw pressure values at each time step, e.g. 22050 values per second
   │
   │  STEP 1 — FRAMING
   │  Slice into overlapping short windows (n_fft=512 samples, ~23ms)
   │  hop_length=160 samples between each window start
   │
   │   |--frame 1--|
   │       |--frame 2--|
   │           |--frame 3--|  ...
   │
   │  STEP 2 — FFT (per frame)
   │  Each frame is converted from time-domain → frequency-domain
   │  via Fast Fourier Transform, revealing which frequencies
   │  are present and how strongly
   │
   │   amplitude
   │   │ █
   │   │ █   █
   │   │ █ █ █  █
   │   │ █ █ █  █   █
   │   └──────────────▶ frequency (Hz)
   │
   │  Doing this for every frame gives you a SPECTROGRAM
   │  (frequency × time matrix)
   │
   │  STEP 3 — MEL FILTERBANK
   │  Linear Hz scale → Mel scale by applying triangular filters
   │  Filters are narrow at low frequencies (where speech detail is)
   │  and wide at high frequencies (less perceptually important)
   │
   │   Hz scale:  |--|--|--|--|------|------|----------|
   │   Mel scale: |--|--|--|--|--|--|--|--|--|--|--|--|--|
   │                 (more resolution at low frequencies)
   │
   │  This compresses n_fft/2 frequency bins → n_mels=80 bins
   │
   │  STEP 4 — AMPLITUDE TO DB
   │  Convert power values to log scale (decibels)
   │  Compresses dynamic range, mimics human loudness perception
   │
   │  STEP 5 — NORMALIZE to [0, 1]
   │
   ▼
MEL SPECTROGRAM  (n_mels × time_frames matrix)

   80 ┤████████████████████████████  ← high freq (wide mel filters)
      │▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
 Mel  │░░░▓▓▓░░░▒▒▒▓▓▓▓░░░▒▒▒▒░░░░░  ← speech content lives here
 bins │▒▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓▓▒▒▓▓▓▓▓▓▒▒▒
    1 ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← low freq (narrow mel filters)
      └────────────────────────────▶ time (frames)



EG of FFT (basically finding the sine graphs that make up the audio signal)

7 second audio @ 22050 Hz = 154,350 samples

Frame 1:  samples 0   → 512    (0ms   → 23ms)
Frame 2:  samples 160 → 672    (7ms   → 30ms)
Frame 3:  samples 320 → 832    (14ms  → 37ms)
...

Frame ~963: samples ~154,000 → 154,512  (~6.98s → 7s)

         frame1  frame2  frame3  ... frame963
high freq  │  0.2 │  0.1 │  0.4 │     │
           │  0.8 │  0.9 │  0.7 │     │
           │  0.3 │  0.3 │  0.2 │     │
low freq   │  0.9 │  0.8 │  0.9 │     │
           └──────┴──────┴──────┴─────┘
              0ms   7ms   14ms      7s



MEL FILTERBANK (80 triangular filters)

Linear Hz:  0    500   1k    2k    4k    8k    11k
            |--|-|-|---|-----|---------|-----------|
                ↑ ↑ ↑                              
            narrow filters           wide filters
            (lots of detail          (less detail,
            for low freqs)           ear can't distinguish)

Each filter outputs 1 value = weighted average of the bins it covers
257 bins → 80 mel values per frame
So each column gets compressed:
         frame1  frame2  frame3  ... frame963
mel 80   │  0.1 │  0.1 │  0.2 │     │   ← averaged from ~50 raw bins
mel 79   │  0.1 │  0.2 │  0.3 │     │
  ...
mel 40   │  0.6 │  0.5 │  0.7 │     │   ← averaged from ~5 raw bins
  ...
mel  2   │  0.8 │  0.9 │  0.8 │     │   ← averaged from ~2 raw bins
mel  1   │  0.4 │  0.3 │  0.5 │     │   ← averaged from ~1 raw bin
           0ms   7ms   14ms      7s

80 rows, evenly spaced perceptually
