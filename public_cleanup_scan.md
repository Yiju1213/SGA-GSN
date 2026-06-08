# Public Cleanup Scan

Generated before cleanup for the `/SGA-GSN` public tree. This file intentionally records original Chinese snippets and legacy paths as audit evidence. It is not runtime code.

## Chinese Content Scan

| File | Lines | Type | Handling |
|---|---:|---|---|
| `utils/vtg3d_utils.py` | 87-160 | docstrings/comments | Translate alignment-error helper documentation and comments to English. |
| `utils/vtg3d_utils.py` | 378-452 | comments/debug comments | Translate useful comments; remove commented-out Chinese warning debug lines. |
| `utils/vtg3d_utils.py` | 475-528, 701 | comments | Translate subset/cache comments to English. |
| `tools/runner_3dvtg.py` | 137-138, 252-254, 292, 620, 628, 644 | comments | Translate binary-classification and batch-index comments to English. |
| `models/MultiResolutionCrossAttn.py` | 13-111 | docstring/comments | Translate grouping downsampler documentation and implementation comments. |
| `models/MultiResolutionCrossAttn.py` | 123-257 | comments/docstrings | Translate layer-parameter, branch-processing, residual, and cross-attention comments. |
| `models/MultiResolutionCrossAttn.py` | 320, 380-494 | comments | Translate validation, scale-transition, fusion, runtime-check comments. |
| `models/GraspStability_CNN_MCA.py` | 27, 47-61 | comments | Translate Q/K/V and attention comments. |
| `models/GraspStability_PPCT.py` | 203 | TODO comment | Translate Chinese TODO into a precise English public-release warning. |

## Hardcoded Path / Public Error Scan

| File | Line | Issue | Handling |
|---|---:|---|---|
| `tools/runner_3dvtg.py` | 54 | Legacy private experiment checkpoint comment under `/AdaPoinTr/experiments/...` | Delete the comment. |
| `tools/runner_3dvtg.py` | 55 | Shape checkpoint hardcoded to `/AdaPoinTr/ckpts/ap_ps55.pth` | Change to `os.path.join(os.getcwd(), "ckpts", "ap_ps55.pth")` so it matches `/SGA-GSN/ckpts/ap_ps55.pth` when run from the public root. |
| `README.md` | 26, 32-39 | `/SGA-GSN` and public data/checkpoint paths | Keep as public usage convention. |
| `utils/dist_utils.py` | 19 | English TODO unrelated to this cleanup | Keep unchanged. |

## Dependency Declaration Gap

| File | Issue | Handling |
|---|---|---|
| `requirements.txt` | Source imports `PIL` and `termcolor`, but `Pillow` and `termcolor` are not declared. | Add `Pillow` and `termcolor`. |
