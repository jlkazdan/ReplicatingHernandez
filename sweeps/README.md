# GPU Memory Usage

## Pretraining

### H200s

All at 2k max context length.

| Model Size | Batch Size (BS) | Memory Usage | Status    |
|:-----------|:---------------:|:------------:|:----------|
| **34M**    |       72        |    115 GB    | ✅ Tested  |
| **48M**    |       72        |    121 GB    | ✅ Tested  |
| **63M**    |       72        |    121 GB    | ✅ Tested  |
| **93M**    |       64        |    117 GB    | ✅ Tested  |
| **153M**   |       64        |    131 GB    | ✅ Tested  |
| **344M**   |       48        |     OOM      | ✅ Tested  |
| **344M**   |       46        |    130 GB    | ⏳ Pending |
| **499M**   |       40        |     OOM      | ✅ Tested  |
| **499M**   |       38        |     OOM      | ✅ Tested  |
| **499M**   |       38        |    129 GB    | ✅ Tested  |
| **660M**   |       32        |     OOM      | ✅ Tested  |
| **660M**   |       26        |    114 GB    | ⏳ Pending |
| **806M**   |       64        |     TBD      | ⏳ Pending |
| **806M**   |       64        |     TBD      | ⏳ Pending |

Note: For 344M, at 44 per device batch size, `gradient_accumulation_steps_unrounded` is 7.11, 
which is then rounded up to 8. We may want to rerun.

## Evaluating