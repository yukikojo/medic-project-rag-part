# Archived Test Files

Tests moved here because their coverage is fully superseded by other test files in the parent directory.

| File | Archived Reason | Superseded By |
|------|----------------|---------------|
| `test_rag.py` | 60-70% content overlap with test_runner.py + test_comprehensive_10.py. Same query sets, same pipeline checks. | `test_runner.py` (Category A/C), `test_comprehensive_10.py` (TC-01/02/04/08/10), `benchmark_metrics.py` (performance) |
| `test_d_10cases.py` | Explicit 10-case subset of `test_runner.py` Category D (50 cases). Imports `COMPREHENSIVE_TEST_CASES` directly from test_runner. Per-component timing breakdown already covered by `benchmark_metrics.py` and `test_comprehensive_10.py` TC-10. | `test_runner.py` (Category D), `benchmark_metrics.py` (timing), `test_comprehensive_10.py` (TC-10) |

These files are kept for reference but are no longer part of the active test suite.
