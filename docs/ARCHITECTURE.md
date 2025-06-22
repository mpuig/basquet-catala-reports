# Basketball Analytics System Architecture

## Overview

The Basketball Analytics System is a comprehensive data pipeline for analyzing basketball match data from the Catalan Basketball Federation website (basquetcatala.cat). The system features a modern, modular architecture built with Python 3.11+ and designed for scalability, maintainability, and extensibility.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Data Pipeline  │───▶│  Report Output  │
│                 │    │                 │    │                 │
│ basquetcatala   │    │     run.py      │    │ HTML Reports    │
│     .cat        │    │                 │    │ GitHub Pages    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  report_tools/  │
                    │   (Core Logic)  │
                    └─────────────────┘
```

### Directory Structure

```
basquet-catala-reports/
├── run.py                          # Single entrypoint script
├── report_tools/                   # Core analytics modules
│   ├── reports.py                  # Main entrypoint with build_groups()
│   ├── data_loaders.py            # Data loading and parsing
│   ├── advanced_stats.py          # Advanced basketball statistics
│   ├── stats.py                   # Basic statistics processing
│   ├── stats_calculator.py        # Statistical computation utilities
│   ├── logger.py                  # Logging configuration
│   └── models/                    # Pydantic data models
│       ├── groups.py              # Group and competition models
│       ├── matches.py             # Match and move models
│       ├── players.py             # Player statistics models
│       └── teams.py               # Team statistics models
├── tests/                         # Comprehensive test suite (73 tests)
│   ├── fixtures/                  # Test data fixtures
│   ├── models/                    # Model-specific tests
│   ├── test_advanced_stats.py     # Advanced statistics tests
│   ├── test_data_loading.py       # Data loading tests
│   └── test_stats.py              # Basic statistics tests
├── data/                          # Raw data storage
│   ├── results_{group_id}.csv     # Match schedules
│   ├── match_moves/               # Play-by-play data
│   ├── match_stats/               # Match statistics
│   ├── team_stats/                # Team seasonal data
│   └── player_stats/              # Player seasonal data
├── reports/                       # Generated HTML reports
│   ├── index.html                 # Main dashboard
│   └── group_{id}/                # Group-specific reports
├── docs/                          # Documentation
└── .github/workflows/             # CI/CD automation
```

## Core Components

### 1. Single Entrypoint (`run.py`)

**Purpose**: Unified pipeline orchestration
**Key Features**:
- Single command to run entire analytics pipeline
- Automatic data fetching for missing data
- Group-based processing with dynamic team discovery
- Professional HTML report generation

**Usage**:
```bash
python run.py --groups 17182 18299 --season 2024
```

### 2. Report Tools Module (`report_tools/`)

#### 2.1 Main Entrypoint (`reports.py`)
- **`build_groups(group_ids, data_dir)`**: Primary function that returns structured Group objects
- Returns complete data structure with teams, matches, and calculated statistics
- Handles data loading, processing, and advanced metric calculations

#### 2.2 Data Loading (`data_loaders.py`)
- **Web scraping**: Match schedules from basquetcatala.cat
- **JSON parsing**: Play-by-play moves and match statistics
- **Data validation**: Using Pydantic models for type safety
- **Error handling**: Graceful handling of missing or corrupted data

#### 2.3 Advanced Statistics (`advanced_stats.py`)
- **Player Aggregate Stats**: Performance metrics across matches
- **On/Off Analysis**: Team performance with/without specific players
- **Lineup Analysis**: 5-player combination effectiveness
- **Pairwise Minutes**: Player combination tracking
- **Player Evolution**: Performance trends over time
- **Plus/Minus Calculations**: Advanced efficiency ratings

#### 2.4 Data Models (`models/`)
Built with **Pydantic v2** for robust data validation:

```python
# Core model hierarchy
Group
├── teams: List[Team]
├── matches: List[Match]
└── advanced_metrics: Dict

Team  
├── stats: TeamStats
├── players: List[Player]
└── matches: List[Match]

Match
├── moves: List[MatchMove]
├── stats: MatchStats
└── calculated_stats: Dict
```

### 3. Testing Infrastructure

**Comprehensive Test Suite**: 73 tests covering:
- **Model validation**: Pydantic model correctness
- **Data loading**: File parsing and error handling
- **Statistical calculations**: Mathematical accuracy
- **Advanced analytics**: Complex basketball metrics
- **Integration tests**: End-to-end pipeline validation

**Test Organization**:
- `tests/models/`: Model-specific validation tests
- `tests/fixtures/`: Anonymized test data
- `tests/test_*.py`: Feature-specific test suites

**Data Privacy**: All test data uses anonymized Spanish/Catalan fake names while preserving statistical relationships.

### 4. CI/CD Pipeline (GitHub Actions)

**Automated Workflow**:
- **Triggers**: Push to main, weekly schedule, manual trigger
- **Process**: Data fetching → Report generation → GitHub Pages deployment
- **Outputs**: Live website + downloadable artifacts

**Deployment**:
- **GitHub Pages**: https://mpuig.github.io/basquet-catala-reports/
- **Artifacts**: 30-day retention for report downloads
- **Scheduling**: Weekly updates to capture new match data

## Data Flow

### 1. Data Collection Phase
```
basquetcatala.cat → Web Scraping → JSON Files → data/
```
- Match schedules scraped and saved as CSV
- Detailed match data (moves, stats) downloaded as JSON
- Team and player seasonal statistics fetched

### 2. Data Processing Phase
```
JSON Files → Pydantic Models → Advanced Analytics → Statistics
```
- Raw JSON parsed into validated Pydantic models
- Advanced basketball statistics calculated
- Player and team performance metrics computed

### 3. Report Generation Phase
```
Statistics → HTML Templates → Reports → GitHub Pages
```
- Professional HTML reports with responsive design
- Hierarchical navigation structure
- Interactive charts and visualizations

## Key Design Principles

### 1. **Single Responsibility**
- Each module has a focused, well-defined purpose
- Clear separation between data loading, processing, and output

### 2. **Type Safety**
- Comprehensive use of Python type hints
- Pydantic v2 models for data validation
- Runtime type checking and error prevention

### 3. **Testability**
- Modular design enables isolated unit testing
- Comprehensive test coverage (73 tests)
- Test fixtures with anonymized data

### 4. **Maintainability**
- Modern Python tooling (uv, ruff, pytest)
- Clear documentation and code organization
- Standardized error handling and logging

### 5. **Scalability**
- Group-based processing for handling multiple competitions
- Efficient data structures and algorithms
- Conditional data fetching to minimize network requests

## Technology Stack

- **Language**: Python 3.11+
- **Dependency Management**: uv (modern, fast)
- **Data Validation**: Pydantic v2
- **Data Processing**: pandas, numpy
- **Web Scraping**: requests, BeautifulSoup4
- **Testing**: pytest (73 comprehensive tests)
- **Linting**: ruff
- **CI/CD**: GitHub Actions
- **Deployment**: GitHub Pages

## Basketball Analytics Features

### Statistical Calculations
- **Basic Stats**: Points, rebounds, assists, shooting percentages
- **Advanced Metrics**: Plus/minus, on/off ratings, lineup effectiveness
- **Time-based Analysis**: Player minutes, combination tracking
- **Evolution Tracking**: Performance trends over time

### Report Features
- **Hierarchical Navigation**: Dashboard → Group → Team → Match
- **Responsive Design**: Works on all devices
- **Interactive Elements**: Clickable navigation, expandable sections
- **Professional Styling**: Clean, readable HTML output

## Future Enhancement Areas

1. **Real-time Data**: WebSocket integration for live match updates
2. **Machine Learning**: Predictive analytics and player recommendations
3. **Internationalization**: Multi-language support
4. **API Layer**: REST API for programmatic access
5. **Database Integration**: Persistent storage for historical analysis
6. **Enhanced Visualizations**: Interactive charts and graphs
7. **Mobile App**: Native mobile application for reports