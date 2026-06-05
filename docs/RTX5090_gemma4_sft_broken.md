# From 5090 Run
Critical finding - the SFT model is generatively broken.

The model generated 2048 tokens in 38s (~53 tok/s, fast) but output is textbook repetition collapse:

这样可以帮助他建立更强的数学基础。这样可以帮助他建立更强的数学基础。

这样可以帮助他建立更强的数学基础。［~70+ repetitions, never emits EOS］

Going back to the morning's smoke test, it ALSO showed early degeneration patterns - the trailing `*（此处是观察点....）* *（此处是引导点....）* *（此处是总结点....）*` was the same loop just truncated at 200 tokens. I missed it.

Diagnosis: the model never emits EOS. Without a max_tokens cap, every teacher call runs to context Limit (~150K tokens, taking ~45 min each). The 88x stowdown I saw wasn't cold-start overhead - it was the model never finishing a single response.

What this means for the eval

We sit somewhere in the STATUS_REPORT §2.8 outcome matrix's "fine-tuned Gemma loses both" cell:

> Either undertraining or data-format issue / Document honestly in §Limitations; Locked headline stands at 72.24

Two paths forward

Path A - Run the eval with the cap, report the (likely bad) numbers as evidence of the failure mode. Negative result is still a paper contribution: documents that Pattern-A SFT on this size dataset caused output collapse. Locked headline 72.24 stays as A-grade submission.

Path B - Diagnose root cause first. Most likely candidates:

1. ﻿﻿﻿Pattern-A format trained the model to depend on specific structural markers - without them at inference,
2. ﻿﻿﻿05_K_M guantization damaged a small model - test by re-running smoke on the BF16 merged checkpoint (but won't fit on 5090 VRAM; CPU inference is very staw)
3. ﻿﻿﻿Chat template mismatch-training used a patched template; serving uses llama.cpp's llama.cpp's Possible structural mismatch.

Which way do you want to play it? My recommendation is Path A first (~45 min to get hard numbers showing the failure) then Path B (post-mortem after we have the data). The negative-result narrative is paper-worthy and matches what §2.8's outcome matrix already anticipates.