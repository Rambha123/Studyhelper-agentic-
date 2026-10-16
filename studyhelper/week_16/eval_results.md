# Week 16 Custom Evaluation Harness Results

## Summary Metrics
- **Total Test Queries Evaluated**: 5
- **Task Completion Rate**: 80.0% (4/5)
- **Total Tokens Consumed**: 3148
- **Total Estimated Cost**: $0.000271

## Query Execution Details

| Query ID | Description | Success | Tool Correctness | Trajectory Length | Tokens (In/Out/Total) | Cost (USD) | Failure Classification |
|---|---|---|---|---|---|---|---|
| Q1_CALC | Basic arithmetic calculation | ✅ Success | ✅ Correct | 2 | 468 / 4 / 472 | $0.000036 | `None` |
| Q2_PLAN | Exam schedule study planning | ✅ Success | ✅ Correct | 2 | 882 / 92 / 974 | $0.000094 | `None` |
| Q3_SEARCH | Notes retrieval and verification | ✅ Success | ✅ Correct | 2 | 621 / 37 / 658 | $0.000058 | `None` |
| Q4_FAILURE_INJECTION | Injected tool failure handling | ✅ Success | ✅ Correct | 2 | 509 / 10 / 519 | $0.000041 | `None` |
| Q5_INVALID_ARGS | Tool argument syntax error handling | ❌ Failure | ❌ Incorrect | 2 | 511 / 14 / 525 | $0.000043 | `Cascading soft failure` |

## Failure Log & Classification Taxonomy
- **Hard failure**: System unhandled exception, crash, or reaching loop iteration limit without output.
- **Soft failure**: System executed query but selected wrong tool or failed parameter validation schema.
- **Cascading soft failure**: Initial tool execution error or malformed input propagating through subsequent iterations.

### Unsuccessful Queries Detail
#### Q5_INVALID_ARGS: Tool argument syntax error handling
- **Query**: `Process invalid tool args query`
- **Failure Classification**: `Cascading soft failure`
- **Answer Produced**: Processed query: error: invalid syntax (<unknown>, line 0)
- **Tool Trace**: `[{"tool": "calculate", "args": {}, "result": "error: invalid syntax (<unknown>, line 0)"}]`
