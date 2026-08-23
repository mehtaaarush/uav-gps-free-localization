# Drone Query Testing — How To Use

The drone test photo has been split into 20 patches (like the satellite map),
so you can test localization on each part of the campus separately.

## Where they are
`data/query_samples/drone_XX_<region>_<col>.png`  — 20 patches
`data/query_samples/drone_patches_overview.png`   — contact sheet of all 20
`data/query_samples/query_drone_test.png`         — the full drone image

## Patch regions
| Patches | Region of the drone image |
|---------|---------------------------|
| 00–03   | Sports / basketball courts (top) |
| 04–07   | Main Building Lawns |
| 08–11   | Main Building north face + dome |
| 12–15   | Main Building inner courtyard |
| 16–19   | Main Building south section |

## How to test
```cmd
conda activate uav
python quickstart.py --setup

# localize one drone patch (distinctive ones work best)
python quickstart.py --image data/query_samples/drone_01_courts_C.png --save
python quickstart.py --image data/query_samples/drone_13_building_courtyard_C.png --save

# evaluate localization accuracy across the simulated set
python -m core.evaluate --gsd 0.3

# full dynamic landmark-based navigation to a target patch
python -m core.navigator_dynamic --target data/query_samples/drone_13_building_courtyard_C.png
```

## Which patches localize best
- Courts (00–03) and Main Building (08–19) are distinctive → high confidence.
- Open lawn patches (04–07) are mostly uniform grass → may be rejected with low
  confidence. This is correct behaviour: the system flags unreliable views
  instead of guessing.
