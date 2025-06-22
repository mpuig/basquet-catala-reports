# Catalan Basketball Data Analysis

This project provides a set of Python scripts to fetch, process, and analyze basketball match data from the Catalan Basketball Federation website (`basquetcatala.cat`). It calculates various statistics and generates reports and visualizations.

## Features

*   **Data Fetching (`fetch_data.py`):**
    *   Scrapes match schedules for specified competition groups and seasons.
    *   Downloads detailed JSON data for each match:
        *   Play-by-play moves (`match_moves/`).
        *   Aggregated match statistics (`match_stats/`).
    *   Downloads team and player seasonal statistics (`team_stats/`, `player_stats/`).
    *   Saves basic schedule info to CSV (`results_{group_id}.csv`).
*   **Data Processing (`process_data.py`):**
    *   Calculates aggregate player and team statistics for a target team across specified groups.
    *   Computes advanced metrics: Pairwise minutes, On/Off Net Rating, Lineup Net Rating & Usage, Rolling Averages (Evolution).
    *   Generates summary tables (Pandas DataFrames).
    *   Creates visualizations (using Seaborn/Matplotlib): Evolution trends, pairwise heatmaps, player On/Off charts, lineup performance, points comparison, player comparison radar plots.
*   **Match Report Generation (`generate_match_reports.py`):**
    *   Creates individual HTML reports for each match played by a target team.
    *   Includes team stat comparisons, player aggregates, On/Off ratings, top lineups for the specific match.
    *   Embeds plots: Score timeline, pairwise heatmap, player On/Off chart, lineup chart.
    *   (Optional) Generates an AI-powered narrative summary using LiteLLM (requires API key).
    *   Creates an `index.html` linking to all generated reports.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Python Version:**
    *   Requires Python 3.10 or higher.
3.  **Install Dependencies:**
    *   It's recommended to use a virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`
        ```
    *   Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
4.  **(Optional) LLM API Key for AI Summaries:**
    *   If you want to use the AI summary feature in `generate_match_reports.py`, you need to set an environment variable for your chosen LLM provider (e.g., OpenAI).
    *   Example for OpenAI:
        ```bash
        export OPENAI_API_KEY='your-api-key-here'
        ```
        *Note: The default model is currently set to `gpt-3.5-turbo`.*

## Usage

The typical workflow is:
1.  Fetch data using `fetch_data.py`.
2.  Analyze the data in aggregate using `process_data.py` OR generate individual reports using `generate_match_reports.py`.

**1. Fetch Data (`fetch_data.py`)**

*   **Fetch schedule only for specific groups:**
    ```bash
    python fetch_data.py --groups 17182 18299 --season 2024 --mode schedule
    ```
*   **Fetch schedule and all detailed JSON data (moves, stats, teams, players):**
    ```bash
    python fetch_data.py --groups 17182 18299 --season 2024 --mode all --data-dir ./data
    ```
*   **See all options:**
    ```bash
    python fetch_data.py --help
    ```

**2. Process Data (`process_data.py`)**

*   **Calculate and display aggregate stats for a team across groups:**
    ```bash
    python process_data.py --team 69630 --groups 17182 18299 --season 2024 --data-dir ./data --plot-dir ./plots
    ```
*   **Analyze a single match:**
    ```bash
    python process_data.py --team 69630 --match <id> --season 2024 --data-dir ./data --plot-dir ./plots
    ```
*   **Exclude specific players (by UUID) from analysis:**
    ```bash
    python process_data.py --team 69630 --groups 17182 --exclude-players <uuid1> <uuid2> --data-dir ./data
    ```
*   **Compare a player's stats between two groups (requires fetching player data):**
    ```bash
    python process_data.py --team 69630 --groups 17182 18299 --compare-player <player_uuid> --plot-dir ./plots
    ```
*   **See all options:**
    ```bash
    python process_data.py --help
    ```

**3. Generate Match Reports (`generate_match_reports.py`)**

*   **Generate reports for a team across specified groups:**
    ```bash
    python generate_match_reports.py --team 69630 --groups 17182 18299 --season 2024 --data-dir ./data --output-dir ./reports
    ```
*   **See all options:**
    ```bash
    python generate_match_reports.py --help
    ```

## Data Structure

*   Raw data fetched by `fetch_data.py` is typically stored in a `data/` directory (configurable via `--data-dir`).
    *   `results_{group_id}.csv`: Basic schedule and results.
    *   `match_moves/{match_id}.json`: Play-by-play events.
    *   `match_stats/{match_id}.json`: Aggregated statistics per match.
    *   `team_stats/team_{team_id}_season_{season_id}.json`: Team statistics for a season.
    *   `player_stats/player_{player_id}_team_{team_id}.json`: Player statistics for a season within a specific team.

## Output

*   **`fetch_data.py`**: CSV schedule files and JSON data files in the specified data directory.
*   **`process_data.py`**: Console output with statistics tables and PNG plot files saved to the specified plot directory (e.g., `plots/`).
*   **`generate_match_reports.py`**: Individual HTML reports (`match_{match_id}/report.html`), supporting files (plots, markdown summaries), and an `index.html` file within the specified output directory (e.g., `reports/`).
