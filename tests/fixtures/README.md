# Test fixtures

`raw/` is a tiny copy of the Snappfood raw format (`comment,label,label_id`), hand-written so
each cleaning step has something to remove. Expected results (asserted in `test_prepare.py`):

| File | Row(s) | Removed by |
|---|---|---|
| train.csv | `غذا عالي بود` (Arabic ي) | duplicate of `غذا عالی بود` after normalization |
| train.csv | `معمولی بود` HAPPY + SAD | conflicting labels (both copies dropped) |
| train.csv | `نظری ندارم` NEUTRAL | unknown label |
| train.csv | `!` and the empty comment | too short |
| train.csv | `پیک خیلی دیر رسید` | also in test.csv (test overlap) |

Final sizes: train 7 rows (4 positive, 3 negative), test 4 rows (2 / 2).
