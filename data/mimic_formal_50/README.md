# mimic_formal_50

50 MIMIC-CXR studies, 25 normal and 25 abnormal. Labels were read from each study's
report and audited by the authors against 11 labels (Normal plus 10 CheXpert findings).

`toy_dataset.json` lists the study IDs and labels only. Images and report text are not
redistributed; fetch the images from your own credentialed copy of MIMIC-CXR-JPG with
`python scripts/prepare_mimic_images.py --mimic-jpg-root <path>`.
