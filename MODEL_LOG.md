# Model Log

| Date | Model id | Result | Notes |
| :-- | :-- | :-- | :-- |
| Earlier (about 2 weeks before 2026-10-25) | qwen/qwen3.8-27b:free | No longer free | Dropped because it stopped being free |
| Earlier | nvidia Nemotron 3 Ultra (free) | Unreliable | Upstream 503 overload errors, and tokens ran out fast |
| 2026-10-25 | google/gemma-4-31b-it:free | Failed | HTTP 429 from the shared free pool even after retries, no usage on our own account |
| 2026-10-25 | cohere/north-mini-code:free | Worked | First successful run with the timetable registry, output in results/cohere_north_mini_code |

**Note:** Free models change or disappear, so always record the model id and date for each run.
