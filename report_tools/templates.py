REPORT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Report: {{ match_id }} - {{ local_name }} vs {{ visitor_name }}</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; margin-bottom: 20px; width: auto; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #eee; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .plot { margin-bottom: 30px; text-align: center; }
        .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .plot-caption { font-size: 0.9em; color: #555; margin-top: 5px; }
        .stats-section { margin-bottom: 30px; }
        .player-columns { display: flex; justify-content: space-between; gap: 20px; }
        .player-column { width: 50%; }
    </style>
</head>
<body>
    <h1>Match Report: {{ local_name }} vs {{ visitor_name }}</h1>
    <p><strong>Match ID:</strong> <a href="https://www.basquetcatala.cat/estadistiques/{{ season }}/{{ match_id }}" target="_blank" title="View official stats page">{{ match_id }}</a> | <strong>Date:</strong> {{ match_date }} | <strong>Group:</strong> {{ group_name }}</p>

    <div class="summary">
        {{ team_comparison_html | safe }}
    </div>

    {% if llm_summary %}
    <h2>AI Generated Summary</h2>
    <div class="summary">
        <p>{{ llm_summary }}</p>
    </div>
    {% endif %}

    <div class="player-columns stats-section">
        <div class="player-column">
            <h2>{{ local_name }} - Player Aggregates</h2>
            {{ local_table_html | safe }}
        </div>
        <div class="player-column">
            <h2>{{ visitor_name }} - Player Aggregates</h2>
            {{ visitor_table_html | safe }}
        </div>
    </div>

    <div class="stats-section">
        <h2>Advanced Stats</h2>

        <h3>On/Off Net Rating</h3>
        {{ on_off_table_html | safe }}

        <h3>Top Lineups (by Net Rating)</h3>
        {{ lineup_table_html | safe }}
    </div>


    <h2>Charts</h2>

    <!-- Placeholder for charts -->
    {% if score_timeline_path %}
    <div class="plot">
        <h3>Score Timeline</h3>
        <img src="{{ score_timeline_path }}" alt="Score Timeline">
        <p class="plot-caption">Score progression throughout the match.</p>
    </div>
    {% endif %}

    {% if pairwise_heatmap_path %}
    <div class="plot">
        <h3>Pairwise Minutes Heatmap</h3>
        <img src="{{ pairwise_heatmap_path }}" alt="Pairwise Minutes Heatmap">
         <p class="plot-caption">Minutes players spent on court together.</p>
   </div>
    {% endif %}

    {% if on_net_chart_path %}
    <div class="plot">
        <h3>Player On-Court Net Rating</h3>
        <img src="{{ on_net_chart_path }}" alt="Player On Net Rating Chart">
        <p class="plot-caption">Team point differential per 40 mins while player was on court.</p>
    </div>
    {% endif %}

     {% if lineup_chart_path %}
    <div class="plot">
        <h3>Top Lineup Net Rating</h3>
        <img src="{{ lineup_chart_path }}" alt="Lineup Net Rating Chart">
        <p class="plot-caption">Point differential per 40 mins for top lineups.</p>
    </div>
    {% endif %}


</body>
</html>
"""
INDEX_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Match Report Index</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        a { text-decoration: none; color: #0066cc; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Match Report Index (Team: {{ target_team_id }})</h1>

    {% if reports %}
    <table>
        <thead>
            <tr>
                <th>Date</th>
                <th>Group</th>
                <th>Local</th>
                <th>Visitor</th>
                <th>Score</th>
                <th>Report Link</th>
            </tr>
        </thead>
        <tbody>
            {% for report in reports %} {# Use already sorted reports #}
            <tr>
                <td>{{ report.match_date }}</td>
                <td>{{ report.group_name }}</td>
                <td>{{ report.local_name }}</td>
                <td>{{ report.visitor_name }}</td>
                <td>{{ report.score }}</td>
                <td><a href="{{ report.report_path }}">{{ report.match_id }}</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p>No match reports were generated for Team {{ target_team_id }}.</p>
    {% endif %}

</body>
</html>
"""
