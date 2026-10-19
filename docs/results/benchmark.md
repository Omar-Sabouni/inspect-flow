| Detector | Scenario | Gate verdict accuracy | Escapes | False rejects | Missed at gate | Detection recall | Detection precision | Defect class accuracy | Detect p50 (ms) | Detect p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classical | nominal | 100.0% ± 0.0 | 0 of 165 | 0 of 55 | 0 of 220 | 99.9% ± 0.1 | 100.0% ± 0.0 | 100.0% ± 0.0 | 2.52 | 3.33 |
| classical | poor lighting | 45.9% ± 7.9 | 0 of 165 | 28 of 55 | 91 of 220 | 64.4% ± 1.0 | 97.1% ± 0.6 | 45.5% ± 3.1 | 2.13 | 2.91 |
| cnn | nominal | 100.0% ± 0.0 | 0 of 165 | 0 of 55 | 0 of 220 | 100.0% ± 0.0 | 99.8% ± 0.1 | 100.0% ± 0.0 | 7.41 | 10.52 |
| cnn | poor lighting | 100.0% ± 0.0 | 0 of 165 | 0 of 55 | 0 of 220 | 100.0% ± 0.0 | 99.8% ± 0.1 | 100.0% ± 0.0 | 7.95 | 17.01 |

20 seeds (7–26), 240 frames each. Ratios are mean ± standard deviation across seeds, in percentage points. Escapes are out of defective parts at the gate, false rejects out of good parts, and misses out of all parts.

13th Gen Intel(R) Core(TM) i7-13700H, OpenCV 4.14.0 with 20 thread(s), Python 3.12.3. CNN model SHA-256 `22794cfabe08`.
