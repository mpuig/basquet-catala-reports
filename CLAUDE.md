# Catalan Basketball Data Analysis Project

This project analyzes basketball match data from the Catalan Basketball Federation website (basquetcatala.cat). It provides tools for data fetching, processing, and reporting.

## Project Structure

- **Data Fetching**: `fetch_data.py` - Scrapes match schedules, downloads JSON data (play-by-play, stats), team/player seasonal statistics
- **Data Processing**: `process_data.py` - Calculates aggregate statistics, advanced metrics (Pairwise minutes, On/Off Net Rating, Lineup analysis), generates visualizations
- **Report Generation**: `generate_match_reports.py` - Creates individual HTML match reports with stats, plots, and optional AI-powered narratives

## Key Commands

### Data Fetching
```bash
python fetch_data.py --groups 17182 18299 --season 2024 --mode all --data-dir ./data
```

### Data Processing  
```bash
python process_data.py --team 69630 --groups 17182 18299 --season 2024 --data-dir ./data --plot-dir ./plots
```

### Report Generation
```bash
python generate_match_reports.py --team 69630 --groups 17182 18299 --season 2024 --data-dir ./data --output-dir ./reports
```

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`
- Optional: LLM API key for AI summaries (OPENAI_API_KEY)

## Data Structure

- `results_{group_id}.csv` - Basic schedules/results
- `match_moves/{match_id}.json` - Play-by-play events  
- `match_stats/{match_id}.json` - Aggregated match statistics
- `team_stats/` - Team seasonal statistics
- `player_stats/` - Player seasonal statistics

## Recent Changes - Single Entrypoint Architecture

The project has been refactored to provide a **single entrypoint** via the new `report_tools/reports.py` module:

### New Architecture
- **`build_groups(group_ids, data_dir)`** - Main entrypoint that returns structured Group objects containing:
  - Teams with schedules and statistics
  - Matches with moves and calculated stats  
  - Advanced metrics (On/Off, lineups, pairwise minutes, player evolution)

### Modular Structure
- `report_tools/data_loaders.py` - Data loading functions
- `report_tools/models/` - Data models (matches, teams, groups, players)
- `report_tools/advanced_stats.py` - Advanced basketball statistics
- `report_tools/stats.py` - Basic statistics processing
- `report_tools/reports.py` - **Main entrypoint with `build_groups()`**

### Key Functions
- `calculate_player_aggregate_stats()` - Player performance across matches
- `calculate_on_off_stats()` - Team performance with/without specific players
- `calculate_lineup_stats()` - 5-player lineup effectiveness
- `calculate_pairwise_minutes()` - Player combination analysis
- `calculate_player_evolution()` - Performance trends over time

### Usage Pattern
```python
from report_tools.reports import build_groups
groups = build_groups([17182, 18299], Path("./data"))
# Access all teams, matches, and calculated stats through group objects
```

The architecture allows generating detailed reports from a single function call rather than running separate scripts.

## Unified Run Script - COMPLETED ✅

The project now features a **unified `run.py` script** that serves as the single entrypoint for the entire basketball analytics pipeline:

### Features
- **Single command** replaces all separate scripts (fetch_data.py, process_data.py, generate_match_reports.py)
- **Group-based organization** instead of period-based filtering
- **Conditional data downloading** - only fetches missing data
- **Professional HTML reports** with hierarchical navigation
- **Advanced statistics** using the sophisticated report_tools architecture

### Usage
```bash
# Generate reports for specific team in a group
python run.py --group_ids 17182 --season 2024 --team_id 69630

# Generate reports for all teams in multiple groups  
python run.py --group_ids 17182 18299 --season 2024

# Force re-download of data
python run.py --group_ids 17182 --season 2024 --force-download
```

### Output Structure
```
reports/
├── index.html                    # Main dashboard
└── group_17182/
    ├── index.html               # Group overview
    ├── team_69630/
    │   └── index.html          # Team detailed report
    ├── team_69621/
    │   └── index.html
    └── ...
```

### Integration
- **Fully integrated data pipeline** - No external script dependencies
- **Direct web scraping** - Integrated fetch_data.py functionality directly into run.py
- **Sophisticated analytics** - Uses the `report_tools/reports.py` architecture
- **Professional HTML output** - Match-by-match breakdowns with responsive styling
- **Hierarchical navigation** - Breadcrumb navigation between dashboard → group → team
- **Advanced statistics** - Team analysis with match breakdowns and advanced metrics

## Test Status - All Fixed ✅

- **73 tests passing** with full test coverage
- **Updated test fixtures** - Fixed DataFrame indexing and basketball time calculations
- **Statistical accuracy** - Corrected +/- expected values to match actual calculations
- **Pydantic v2 compatibility** - Fixed deprecated `Config` classes, now using `ConfigDict`
- **Type safety** - Fixed string/integer type assertions in tests
- **Code quality** - Removed unused variables flagged by ruff linter
- **Test command**: `python -m pytest tests/ -v`

## Code Cleanup - COMPLETED ✅

The codebase has been optimized by removing unused modules and functions:

### Removed Files
- **`report_tools/llm.py`** - Unused LLM integration for AI-powered match summaries
- **`report_tools/templates.py`** - Unused HTML template strings (replaced by Chart.js frontend)
- **`report_tools/plotting.py`** - Unused matplotlib plotting functions (replaced by web-based charts)

### Architecture Optimization
- **Dual statistics architecture** - Maintained both `stats.py` and `stats_calculator.py` for validation
- **Clean imports** - Removed all unused imports across the codebase
- **Focused functionality** - ~600+ lines of dead code removed while preserving all essential features

### Data Organization
- **`data/samples/`** - Recent, lightweight test data for model validation (9-592KB files)
- **`tests/fixtures/`** - Historical data for integration tests (25-592KB files)
- **Separation maintained** - Different purposes: samples for focused testing, fixtures for broad coverage