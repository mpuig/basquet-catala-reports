# Context Management for Basketball Analytics System

This document captures the essential context and development history from Claude Code sessions to enable future improvements and maintenance.

## Key Technical Decisions

### 1. Basketball Time Calculation Logic
**Issue**: Basketball clocks count DOWN, not up
**Solution**:
```python
# For period 1, min=5, sec=30 (5:30 remaining)
# Elapsed time = 600 - (5*60 + 30) = 270 seconds
elapsed_time = PERIOD_LENGTH_SEC - (minutes * 60 + seconds)
```
**Context**: This was a critical fix that resolved test failures in `test_match_moves.py`

### 2. DataFrame Indexing Pattern
**Issue**: Tests expected DataFrame with 'Player' column as index
**Solution**:
```python
# Wrong approach (caused KeyError)
df["Player"]

# Correct approach
df_indexed = df.set_index("Player")
assert df_indexed.loc["PLAYER_NAME", "+/-"] == expected_value
```
**Context**: Fixed in `test_advanced_stats.py` - this pattern should be used consistently

## Critical Code Patterns

### 1. Basketball Statistics Calculation
```python
def calculate_on_off_stats(match: Match, team_id: int) -> pd.DataFrame:
    """Calculate team performance with/without specific players."""
    # Pattern: Always validate team_id and handle edge cases
    # Return: DataFrame with Player as index for consistent access
```
**Key**: Always return DataFrames with proper indexing for test compatibility

### 2. Error Handling Pattern
```python
def load_json_file(file_path: Path) -> Optional[Dict]:
    if not file_path.exists():
        logger.debug(f"File not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load JSON from {file_path}: {e}")
        return None
```
**Principle**: Graceful degradation with appropriate logging levels

## Testing Insights

### Test Architecture Success Factors
1. **Fixture-based Testing**: Using pytest fixtures for consistent test data
2. **Anonymized Data**: Maintains privacy while preserving statistical relationships
3. **Modular Organization**: `tests/models/` directory for model-specific tests
4. **Comprehensive Coverage**: 73 tests covering all major functionality

### Common Test Patterns
```python
# Model validation
def test_model_creation():
    data = {...}
    model = Model.model_validate(data)
    assert isinstance(model, Model)

# DataFrame testing
def test_dataframe_operations():
    df = calculate_stats(...)
    df_indexed = df.set_index("Player")
    assert df_indexed.loc["PLAYER_NAME", "STAT"] == expected
```

## Data Structure Insights

### Group-Based Architecture
```python
# Primary data flow
groups = build_groups([17182, 18299], Path("./data"))
# Returns: List[Group] with complete team/match/stats hierarchy
```

### Statistical Calculation Order
1. **Load raw data** (moves, stats, schedules)
2. **Create models** (Match, Team, Player objects)
3. **Calculate basic stats** (aggregations, summaries)
4. **Advanced analytics** (on/off, lineups, pairwise)
5. **Generate reports** (HTML with navigation)

## Code Quality Standards

### Established Patterns
- **Type hints**: Throughout codebase for IDE support and documentation
- **Pydantic validation**: All external data validated through models
- **Error logging**: Consistent use of logger with appropriate levels
- **Test coverage**: Each new feature requires corresponding tests

### Dependency Management
- **uv**: Primary dependency manager (fast, modern)
- **pyproject.toml**: Central configuration for all tools
- **No requirements.txt**: Replaced with modern uv workflow

## Performance Considerations

### Efficient Data Loading
- **Conditional fetching**: Only download missing data
- **JSON parsing**: Direct file I/O, no unnecessary processing
- **DataFrame operations**: Use vectorized operations where possible

### Memory Management
- **Lazy loading**: Load data only when needed
- **Generator patterns**: For large datasets (if implemented)
- **Cleanup**: Explicit resource management in long-running processes

## Common Debugging Approaches

### Test Failures
1. **Check DataFrame indexing**: Ensure proper `.set_index()` usage
2. **Verify anonymized names**: Update test assertions after data anonymization
3. **Basketball time logic**: Remember countdown vs. elapsed time differences

### Workflow Issues
1. **Argument validation**: Check CLI parameter names match argparse
2. **File paths**: Verify relative vs. absolute path handling
3. **GitHub Pages**: Confirm source setting and .nojekyll placement

## Development Environment

### Recommended Setup
```bash
# Clone and setup
git clone https://github.com/mpuig/basquet-catala-reports.git
cd basquet-catala-reports

# Install with uv (recommended)
uv sync

# Run tests
python -m pytest tests/ -v

# Generate reports
python run.py --groups 17182 18299 --season 2024
```

### IDE Configuration
- **Type checking**: Enable for full type hint support
- **Linting**: ruff configuration in pyproject.toml
- **Testing**: pytest integration for test discovery

## Future Development Guidelines

### When Adding New Features
1. **Write tests first**: Test-driven development approach
2. **Update models**: Extend Pydantic models for new data structures
3. **Document changes**: Update ARCHITECTURE.md and relevant docs
4. **Anonymize data**: Any new test data must use fake names

### When Modifying Statistics
1. **Validate calculations**: Ensure basketball logic correctness
2. **Update tests**: Modify expected values in test assertions
3. **Check DataFrame patterns**: Maintain consistent indexing approach
4. **Performance testing**: Large datasets may require optimization

### When Changing Data Models
1. **Migration strategy**: Plan for existing data compatibility
2. **Validation updates**: Ensure Pydantic models remain accurate
3. **Test data updates**: Synchronize fixtures with model changes
4. **Documentation**: Update data structure diagrams

This context document should be updated whenever significant architectural decisions are made or patterns are established.
