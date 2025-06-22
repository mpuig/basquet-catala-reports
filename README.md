# Catalan Basketball Data Analysis

A comprehensive basketball analytics tool that fetches, processes, and analyzes match data from the Catalan Basketball Federation website (`basquetcatala.cat`). Features a unified pipeline for generating professional HTML reports with advanced basketball statistics.

## Features

*   **Unified Pipeline (`run.py`):**
    *   Single command to run the entire analytics pipeline
    *   Automatically fetches missing data and generates reports
    *   Group-based organization with hierarchical navigation
    *   Professional HTML output with responsive design

*   **Data Collection:**
    *   Scrapes match schedules for competition groups and seasons
    *   Downloads detailed play-by-play moves and match statistics
    *   Fetches team and player seasonal statistics
    *   Conditional downloading - only fetches missing data

*   **Advanced Basketball Analytics:**
    *   Player aggregate statistics and performance metrics
    *   On/Off Net Rating analysis (team performance with/without players)
    *   Lineup effectiveness analysis (5-player combinations)
    *   Pairwise minutes tracking (player combination analysis)
    *   Player evolution and performance trends over time
    *   Plus/minus calculations and efficiency ratings

*   **Professional Report Generation:**
    *   Individual match breakdowns with detailed statistics
    *   Team performance analysis across competitions
    *   Interactive charts and visualizations
    *   Hierarchical navigation: Dashboard → Group → Team → Match
    *   Responsive HTML design for all devices

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mpuig/basquet-catala-reports.git
    cd basquet-catala-reports
    ```

2.  **Install dependencies using uv (recommended):**
    ```bash
    uv sync
    ```
    
    Or using pip with virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -e .
    ```

3.  **Requirements:**
    *   Python 3.10 or higher
    *   All dependencies managed via `pyproject.toml`

## Usage

The project features a **unified `run.py` script** that serves as the single entrypoint for the entire basketball analytics pipeline.

### Basic Usage

**Generate reports for all teams in a competition group:**
```bash
python run.py --groups 17182 --season 2024
```

**Generate reports for all teams in multiple groups:**
```bash
python run.py --groups 17182 18299 --season 2024
```

### Command Line Options

```bash
python run.py --help
```

Key parameters:
- `--groups`: Competition group IDs (space-separated for multiple)
- `--season`: Season year (e.g., 2024)
- `--data-dir`: Directory to store data files (default: ./data)
- `--output-dir`: Directory to save HTML reports (default: ./reports)
- `--verbose`: Enable detailed logging output
- `--match`: Analyze a specific match ID (optional)

### Output Structure

The script generates a hierarchical report structure:

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

### Features

- **Automatic data management**: Only downloads missing data
- **Professional HTML reports**: Responsive design with navigation breadcrumbs
- **Advanced statistics**: On/Off ratings, lineup analysis, player evolution
- **Match-by-match breakdowns**: Detailed analysis for each game
- **Team comparisons**: Performance across different competitions

## Data Structure

The system automatically organizes data in the following structure:

```
data/
├── results_{group_id}.csv                              # Basic schedules and results
├── match_moves/{match_id}.json                         # Play-by-play events  
├── match_stats/{match_id}.json                         # Aggregated match statistics
├── team_stats/team_{team_id}_season_{season}.json      # Team seasonal statistics
└── player_stats/player_{player_id}_team_{team_id}.json # Player seasonal statistics
```

## Architecture

- **`run.py`**: Unified entrypoint and main pipeline orchestration
- **`report_tools/`**: Core analytics and data processing modules
  - `reports.py`: Main entrypoint with `build_groups()` function
  - `data_loaders.py`: Data loading and parsing functions
  - `models/`: Pydantic data models (matches, teams, players, groups)
  - `advanced_stats.py`: Advanced basketball statistics calculations
  - `stats.py`: Basic statistics processing
  - `stats_calculator.py`: Statistical computation utilities
- **`tests/`**: Comprehensive test suite (73 tests) with fixtures
- **Data privacy**: All personal names have been anonymized with fake Spanish/Catalan names

## Testing

Run the complete test suite:
```bash
python -m pytest tests/ -v
```

The project includes 73 comprehensive tests covering:
- Data loading and parsing
- Statistical calculations  
- Model validation
- Advanced analytics
- Edge cases and error handling

## Automated Report Generation

The project includes GitHub Actions automation to generate reports automatically:

### GitHub Actions Workflow

The workflow (`.github/workflows/generate-reports.yml`) runs:
- **On push to main branch**: Generates fresh reports for the latest code
- **Weekly on Sundays**: Keeps reports up-to-date with new match data
- **Manual trigger**: Can be run on-demand from GitHub Actions tab
- **Pull requests**: Validates that report generation still works

### GitHub Pages Deployment

Reports are automatically deployed to GitHub Pages when pushed to main branch:
- **Live reports**: Available at `https://mpuig.github.io/basquet-catala-reports/`
- **Automatic updates**: New reports deployed on every main branch push
- **Static hosting**: No server required, fast global CDN delivery

### Artifacts

Generated reports are also stored as GitHub Actions artifacts:
- **30-day retention**: Download reports directly from workflow runs
- **Version history**: Previous report versions available for comparison

## Contributing

This project uses modern Python tooling:
- **uv** for dependency management
- **Pydantic v2** for data validation
- **pytest** for testing
- **ruff** for linting
- **Type hints** throughout the codebase
- **GitHub Actions** for CI/CD and automated report generation
