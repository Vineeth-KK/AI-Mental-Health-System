# CareAI ML — Dataset Exploration Report

**Dataset:** nayma11/Mental-Health_Text-Classification_Dataset
**File:** mental_heath_unbanlanced.csv
**Total samples:** 49,612
**Columns:** ['Unique_ID', 'text', 'status', '_text_len', '_word_count']
**Text column:** `text`
**Label column:** `status`

## Shape
- Rows: 49,612
- Columns: 5

## Missing Values
{'Unique_ID': 9600}

## Duplicates
- Exact duplicate rows: 0 (0.0%)

## Label Distribution
- Normal: 18,391 (37.1%)
- Depression: 14,506 (29.2%)
- Suicidal: 11,212 (22.6%)
- Anxiety: 5,503 (11.1%)

## Text Length Statistics
- Mean chars: 399.2
- Median chars: 242.0
- Max chars: 38785.0
- Mean words: 78.2
- Min chars: 7.0

## Sample Texts Per Class

### Anxiety (5,503 samples)
> oh my gosh...

### Normal (18,391 samples)
> Gr gr dreaming of ex crush to be my game, God...

### Depression (14,506 samples)
> I recently went through a breakup and she said she still wants to be friends so I said I can try doing that but when she talks to me about things it a...

### Suicidal (11,212 samples)
> I am 20 year old with some good friends but I am just tired a had a problem with bullying when I was little and that hit me hard but the thing that an...

## Notes
- Labels verified against expected set: Normal, Depression, Anxiety, Suicidal
- After cleaning: 661 exact duplicates removed → 48,950 usable samples
- Label encoding: Normal=0, Depression=1, Anxiety=2, Suicidal=3
- Anxiety class is minority (10.9%) — class weights applied during training