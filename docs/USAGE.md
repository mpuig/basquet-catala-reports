# Basketball Analytics System Usage Guide

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/mpuig/basquet-catala-reports.git
cd basquet-catala-reports

# Install dependencies (recommended: use uv)
uv sync

# Alternative: pip installation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Basic Usage
```bash
# Generate reports for one competition group
python run.py --groups 17182 --season 2024

# Generate reports for multiple groups
python run.py --groups 17182 18299 --season 2024

# Enable verbose logging
python run.py --groups 17182 --season 2024 --verbose
```

## Command Line Options

### Required Parameters
- `--groups`: Competition group IDs (space-separated for multiple groups)
- `--season`: Season year (default: 2024)

### Optional Parameters
- `--data-dir`: Directory to store raw data files (default: ./data)
- `--output-dir`: Directory to save HTML reports (default: ./reports)
- `--verbose`: Enable detailed logging output
- `--match`: Analyze a specific match ID (optional, for debugging)

### Examples
```bash
# Basic usage
python run.py --groups 17182 18299 --season 2024

# Custom directories
python run.py --groups 17182 --season 2024 --data-dir /path/to/data --output-dir /path/to/reports

# Debug a specific match
python run.py --groups 17182 --season 2024 --match 158215 --verbose

# Help
python run.py --help
```

## Understanding the Output

### Report Structure
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

### Navigation Flow
1. **Main Dashboard** (`reports/index.html`): Overview of all groups
2. **Group Page** (`group_*/index.html`): All teams in the competition
3. **Team Page** (`team_*/index.html`): Detailed team analysis with match breakdowns

### Report Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Hierarchical Navigation**: Breadcrumb navigation between levels
- **Advanced Statistics**: On/Off ratings, lineup analysis, player evolution
- **Match Breakdowns**: Detailed analysis for each game

## Data Collection Process

### Automatic Data Fetching
The system automatically:
1. **Scrapes match schedules** from basquetcatala.cat
2. **Downloads missing match data** (play-by-play moves, statistics)
3. **Fetches team/player statistics** for seasonal analysis
4. **Saves data locally** to avoid re-downloading

### Data Storage Structure
```
data/
├── results_{group_id}.csv              # Match schedules and results
├── match_moves/{match_id}.json         # Play-by-play events
├── match_stats/{match_id}.json         # Aggregated match statistics  
├── team_stats/team_{id}_season_{season}.json    # Team seasonal data
└── player_stats/player_{id}_team_{id}.json      # Player seasonal data
```

### Data Freshness
- **Conditional Downloads**: Only fetches missing or outdated data
- **Incremental Updates**: New matches automatically included
- **Manual Override**: Delete data files to force re-download

## Basketball Statistics Explained

### Basic Statistics
- **Points, Rebounds, Assists**: Standard basketball metrics
- **Shooting Percentages**: Field goals, 3-pointers, free throws
- **Minutes Played**: Time on court per player
- **Fouls and Turnovers**: Defensive statistics

### Advanced Analytics

#### On/Off Analysis
Shows team performance when specific players are on vs. off the court:
- **Net Rating**: Point differential per 100 possessions
- **Usage**: How much a player impacts team performance
- **Efficiency**: Points scored/allowed per minute of play

#### Lineup Analysis  
Evaluates 5-player combinations:
- **Lineup Net Rating**: Point differential for specific 5-player groups
- **Minutes Together**: How much time combinations play together
- **Effectiveness**: Which lineups perform best in different situations

#### Pairwise Minutes
Tracks which players play together:
- **Combination Analysis**: Two-player performance metrics
- **Chemistry Indicators**: Which player pairs work well together
- **Rotation Patterns**: Understanding coaching decisions

#### Player Evolution
Performance trends over time:
- **Rolling Averages**: Performance trends across games
- **Consistency Metrics**: Game-to-game variation
- **Improvement Tracking**: Season-long development

## Troubleshooting

### Common Issues

#### 1. No Data Available
```
Error: No matches found for group 17182
```
**Solution**: 
- Check if group ID is correct
- Verify season parameter matches available data
- Ensure internet connection for data fetching

#### 2. Missing Reports
```
Reports directory is empty
```
**Solution**:
- Check data was downloaded successfully (look in `data/` directory)
- Verify all required arguments provided
- Run with `--verbose` flag for detailed logging

#### 3. Incomplete Statistics
```
Warning: Some advanced statistics unavailable
```
**Solution**:
- Ensure sufficient match data exists (need multiple games for trends)
- Check that play-by-play data is available (some matches may only have basic stats)

### Data Issues

#### Invalid Match Data
If specific matches show errors:
```bash
# Debug specific match
python run.py --groups 17182 --season 2024 --match 158215 --verbose
```

#### Corrupted Data Files
Delete the problematic files to force re-download:
```bash
# Remove specific match data
rm data/match_moves/158215.json
rm data/match_stats/158215.json

# Remove all data for fresh start
rm -rf data/
```

### Performance Issues

#### Slow Generation
For large datasets:
- **Limit groups**: Process one group at a time
- **Check disk space**: Ensure sufficient storage for data and reports
- **Network speed**: Slow internet affects data fetching

#### Memory Usage
For systems with limited RAM:
- **Process smaller groups**: Avoid processing too many teams simultaneously
- **Close other applications**: Free up system memory
- **Consider batch processing**: Generate reports for subsets of teams

## Development and Testing

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/models/ -v          # Model tests
python -m pytest tests/test_advanced_stats.py -v  # Statistics tests

# Run with coverage
python -m pytest tests/ --cov=report_tools --cov-report=html
```

### Code Quality
```bash
# Lint code
ruff check .

# Format code  
ruff format .

# Type checking (if mypy installed)
mypy report_tools/
```

### Adding New Features
1. **Write tests first**: Create test cases in appropriate `tests/` subdirectory
2. **Implement feature**: Add functionality to relevant `report_tools/` module
3. **Update documentation**: Modify this file and ARCHITECTURE.md as needed
4. **Test thoroughly**: Ensure all tests pass before committing

## Integration with External Systems

### GitHub Actions (Automated Reports)
The repository includes automated report generation:
- **Weekly Schedule**: Runs every Sunday to capture new match data
- **Manual Trigger**: Can be run on-demand from GitHub Actions
- **GitHub Pages**: Reports automatically published to live website

### API Integration (Future)
The modular architecture supports future API development:
```python
# Example programmatic usage
from report_tools.reports import build_groups

groups = build_groups([17182, 18299], Path("./data"))
for group in groups:
    print(f"Group: {group.name}")
    for team in group.teams:
        print(f"  Team: {team.name} ({len(team.matches)} matches)")
```

## Best Practices

### Data Management
- **Regular Backups**: Save `data/` directory for historical analysis
- **Version Control**: Don't commit large data files to git
- **Cleanup**: Periodically remove old data to save disk space

### Report Generation
- **Consistent Timing**: Run reports after match days for fresh data
- **Quality Checks**: Review generated reports for data accuracy
- **Performance Monitoring**: Track generation time for optimization

### Privacy and Ethics
- **Data Anonymization**: All personal names in tests are anonymized
- **Public Data Only**: System only uses publicly available match data
- **Respect Rate Limits**: Avoid excessive requests to source website

## Getting Help

### Documentation Resources
- **Architecture**: See `docs/ARCHITECTURE.md` for technical details
- **Context**: See `docs/CONTEXT_MANAGEMENT.md` for development history
- **Code**: Well-documented functions and classes throughout codebase

### Support Channels
- **GitHub Issues**: Report bugs or request features
- **Code Comments**: Inline documentation for complex logic
- **Test Examples**: Look at test files for usage patterns

### Contributing
1. **Fork the repository**: Create your own copy for modifications
2. **Create feature branch**: Work on isolated feature branches
3. **Write tests**: All new features need corresponding tests
4. **Submit pull request**: Include clear description of changes

This system is actively maintained and documented. For specific questions about basketball statistics calculations or technical implementation details, refer to the extensive test suite and inline code documentation.