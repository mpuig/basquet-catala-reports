import requests
from bs4 import BeautifulSoup
import csv
import sys
import re # Import regex module
import os # Import os module for directory operations
import json # Import json module for JSON operations

def fetch_and_inspect_tables(url):
    """Fetches HTML and prints details of all tables found."""
    try:
        print(f"Fetching {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print("HTML fetched successfully. Inspecting tables...")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    all_tables = soup.find_all('table')

    if all_tables:
        print("\nTables found on page:")
        for i, table in enumerate(all_tables):
            table_id = table.get('id')
            table_classes = table.get('class')
            print(f"  Table {i+1}: ID='{table_id}', Class='{table_classes}'")
    else:
        print("No tables found on the page.")

def fetch_and_parse_data(url):
    """Fetches HTML, parses data, extracts group ID, and returns (matches, group_id)."""
    group_id = "UNKNOWN" # Default value
    try:
        # Extract group_id from URL (assuming format .../resultats/GROUPID/...) 
        parts = url.strip('/').split('/')
        if len(parts) >= 2 and parts[-2].isdigit():
             group_id = parts[-2]
        else:
             print(f"Warning: Could not reliably extract group ID from URL: {url}", file=sys.stderr)

        print(f"Fetching {url} (Group ID: {group_id})...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        print("HTML fetched successfully. Parsing...")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}", file=sys.stderr)
        return None, None # Return None for both matches and group_id on error
    except Exception as e: # Catch potential errors during group ID extraction
        print(f"Error processing URL or extracting group ID: {e}", file=sys.stderr)
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    matches = []
    current_jornada = ""

    # Find all elements that could be Jornada headers or match containers
    # The structure seems to be: Jornada Header DIV -> Container DIV -> rowsJornada DIV -> Match rows
    # Let's find all Jornada headers first
    jornada_headers = soup.find_all('div', class_=re.compile(r'bg-2 pd-5 ff-1 fs-16 c-5'))

    if not jornada_headers:
        print("Could not find Jornada header divs.", file=sys.stderr)
        # Fallback or alternative finding method might be needed if class names change
        # For now, let's try finding match rows directly if headers fail
        all_match_rows = soup.find_all('div', class_=re.compile(r'rowJornada'))
        if not all_match_rows:
            print("Could not find any match rows directly either. Aborting parse.", file=sys.stderr)
            return None, group_id # Return group_id even if parsing fails later
        else:
            print("Parsing match rows directly without Jornada headers.", file=sys.stderr)
            # Handle parsing without Jornada context (Jornada field will be empty)
            process_match_rows(all_match_rows, "N/A", matches)
    else:
        for header in jornada_headers:
            # Extract Jornada number/text
            jornada_text = header.get_text(strip=True)

            # --- Skip the initial "Equips del grup" section --- 
            if jornada_text == "Equips del grup":
                print(f"Skipping section: {jornada_text}...")
                continue # Move to the next header
            # --- End skip ---

            jornada_match = re.search(r'Jornada\s*(\d+)', jornada_text)
            current_jornada = jornada_match.group(1) if jornada_match else jornada_text # Use full text if number not found
            print(f"Processing {current_jornada}...")

            # Find the container sibling that holds the matches for this jornada
            # It seems to be the next div.container.m-bottom
            # -- OLD: match_container = header.find_next_sibling('div', class_='container m-bottom') --
            # ++ NEW: Try finding the next occurrence of the container after the header ++
            match_container = header.find_next('div', class_='container m-bottom')

            if match_container:
                rows_jornada_div = match_container.find('div', class_='rowsJornada')
                if rows_jornada_div:
                    # Find individual match rows within this Jornada
                    # These can be direct children div with id='fila' or the nested div.rowJornada
                    # Let's find all divs that contain match details directly under rowsJornada
                    # This includes both 'fila' divs and the 'Descansa' row
                    potential_rows = rows_jornada_div.find_all('div', recursive=False)
                    process_match_rows(potential_rows, current_jornada, matches)
                else:
                    print(f"Could not find 'div.rowsJornada' for {current_jornada}", file=sys.stderr)
            else:
                print(f"Could not find match container for {current_jornada}", file=sys.stderr)

    print(f"Parsed {len(matches)} matches in total.")
    return matches, group_id # Return both matches and group_id

def process_match_rows(rows, current_jornada, matches):
    """Helper function to process a list of potential match rows."""
    for row_wrapper in rows:
        # A row could be a 'Descansa' row directly under rowsJornada
        # or a div with id='fila' containing the actual rowJornada div.
        is_descansa = False
        descansa_team = None
        descansa_team_id = "N/A"

        # Check for 'Descansa' case first
        if 'rowJornada' in row_wrapper.get('class', []) and 'col-md-12' in row_wrapper.get('class', []):
            cols = row_wrapper.find_all('div', recursive=False)
            if len(cols) == 3 and "Descansa" in cols[2].get_text(strip=True):
                is_descansa = True
                team_link = cols[0].find('a', class_='teamNameLink')
                descansa_team = team_link.get_text(strip=True) if team_link else cols[0].get_text(strip=True)
                if team_link and team_link.has_attr('href'):
                    try:
                        descansa_team_id = team_link['href'].strip('/').split('/')[-1]
                    except IndexError:
                        descansa_team_id = "ERROR_EXTRACTING_ID"
                print(f"  Found Descansa: {descansa_team} (ID: {descansa_team_id})")
                # Add N/A for opponent ID, score, match_id
                matches.append([current_jornada, "N/A", descansa_team, descansa_team_id, "Descansa", "N/A", "N/A", "N/A"])
                continue

        # If not descansa, look for the actual match row inside a potential wrapper (like div#fila)
        match_row = row_wrapper
        if row_wrapper.get('id') == 'fila':
            match_row = row_wrapper.find('div', class_=re.compile(r'rowJornada'))

        if not match_row or 'rowJornada' not in match_row.get('class', []):
            # print(f"Skipping row wrapper, doesn't seem to be a match row or descansa: {row_wrapper.prettify()[:200]}")
            continue

        # --- Extract data from a standard match row --- 
        local_team, local_team_id, visitor_team, visitor_team_id, date_time, score, match_id = "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A" # Added team IDs

        # Find team names, date/time within the central column (col-md-10 or col-xs-12)
        main_col = match_row.find('div', class_=re.compile(r'col-md-10'))
        if not main_col:
            print(f"  Skipping row, could not find main column (col-md-10). Row: {match_row.prettify()[:200]}")
            continue
        
        # First inner row contains team names and date/time
        info_row = main_col.find('div', class_=re.compile(r'rowJornada col-md-12'))
        if info_row:
            cols = info_row.find_all('div', class_=re.compile(r'col-md-4'))
            if len(cols) == 3:
                # Local Team & ID
                local_team_link = cols[0].find('a', class_='teamNameLink')
                local_team = local_team_link.get_text(strip=True) if local_team_link else cols[0].get_text(strip=True)
                if local_team_link and local_team_link.has_attr('href'):
                    try:
                        local_team_id = local_team_link['href'].strip('/').split('/')[-1]
                    except IndexError:
                        local_team_id = "ERROR_EXTRACTING_ID"
                
                # Date/Time
                time_div = cols[1].find('div', id='time2')
                date_time = time_div.get_text(strip=True) if time_div else cols[1].get_text(strip=True)
                
                # Visitor Team & ID
                visitor_team_link = cols[2].find('a', class_='teamNameLink')
                visitor_team = visitor_team_link.get_text(strip=True) if visitor_team_link else cols[2].get_text(strip=True)
                if visitor_team_link and visitor_team_link.has_attr('href'):
                    try:
                        visitor_team_id = visitor_team_link['href'].strip('/').split('/')[-1]
                    except IndexError:
                        visitor_team_id = "ERROR_EXTRACTING_ID"
            else:
                 print(f"  Skipping info row, unexpected number of columns ({len(cols)}). Row: {info_row.prettify()[:200]}")
                 continue # Skip if structure is not as expected

        # Second inner row contains scores AND stats link
        score_row = main_col.find('div', class_=re.compile(r'rowJornada fs-38'))
        if score_row:
            cols = score_row.find_all('div', class_=re.compile(r'col-md-4'))
            if len(cols) == 3:
                # Score
                local_score = cols[0].get_text(strip=True)
                visitor_score = cols[2].get_text(strip=True)
                if local_score and visitor_score:
                    score = f"{local_score}-{visitor_score}"
                elif local_score:
                     score = local_score # Handle cases maybe only one score exists?
                elif visitor_score:
                     score = visitor_score
                # Stats URL -> Match ID (in the middle column, cols[1])
                stats_img = cols[1].find('img', title='Estadística')
                if stats_img:
                    stats_link = stats_img.find_parent('a')
                    if stats_link and stats_link.has_attr('href'):
                        stats_url = stats_link['href']
                        # Extract last part of the URL path as ID
                        if stats_url:
                            try:
                                match_id = stats_url.strip('/').split('/')[-1]
                            except IndexError:
                                print(f"  Warning: Could not extract ID from URL: {stats_url}", file=sys.stderr)
                                match_id = "ERROR_EXTRACTING_ID" # Mark if extraction fails
            # else: score and match_id remain "N/A" if structure wrong

        if local_team != "N/A": # Only add if we found at least a local team
             print(f"  Found Match: {local_team}({local_team_id}) vs {visitor_team}({visitor_team_id}) on {date_time} -> {score} (ID: {match_id})")
             matches.append([current_jornada, date_time, local_team, local_team_id, visitor_team, visitor_team_id, score, match_id])
        # else: print(f"  Skipping row, could not extract essential data. Row: {match_row.prettify()[:200]}")

def fetch_and_save_match_moves(match_id, output_dir="data/match_moves"):
    """Fetches match moves (play-by-play) JSON for a given match_id and saves it."""
    if not match_id or match_id in ["N/A", "ERROR_EXTRACTING_ID"]:
        # print(f"Skipping match moves fetch for invalid match_id: {match_id}")
        return

    # Endpoint for play-by-play moves
    json_url = f"https://msstats.optimalwayconsulting.com/v1/fcbq/getJsonWithMatchMoves/{match_id}?currentSeason=true"
    filename = os.path.join(output_dir, f"{match_id}.json")

    # Avoid re-downloading if file already exists
    if os.path.exists(filename):
        print(f"Match moves JSON already exists for match_id {match_id}: {filename}")
        return

    print(f"Fetching match moves for match_id {match_id} from {json_url}...")
    try:
        response = requests.get(json_url, timeout=20) 
        response.raise_for_status()

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the raw text content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Successfully saved match moves JSON to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching match moves for match_id {match_id} from {json_url}: {e}", file=sys.stderr)
    except IOError as e:
        print(f"Error saving match moves JSON file {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred fetching/saving match moves for match_id {match_id}: {e}", file=sys.stderr)

def fetch_and_save_team_stats(team_id, season_id, output_dir="data/team_stats"):
    """Fetches team season stats JSON for a given team_id and season_id and saves it."""
    if not team_id or not isinstance(team_id, str) or not team_id.isdigit():
        print(f"Skipping team stats fetch for invalid team_id: {team_id}")
        return
    if not season_id:
        print(f"Skipping team stats fetch for missing season_id.")
        return

    stats_url = f"https://msstats.optimalwayconsulting.com/v1/fcbq/team-stats/team/{team_id}/season/{season_id}"
    filename = os.path.join(output_dir, f"team_{team_id}_season_{season_id}.json")

    # Avoid re-downloading if file already exists
    if os.path.exists(filename):
        print(f"Team stats JSON already exists for team {team_id}, season {season_id}: {filename}")
        return

    print(f"Fetching team stats for team {team_id}, season {season_id} from {stats_url}...")
    try:
        response = requests.get(stats_url, timeout=15)
        response.raise_for_status() 

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Attempt to parse JSON to ensure it's valid before saving
        # (Optional, but good practice)
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"  Warning: Response for team {team_id} season {season_id} is not valid JSON. Saving raw text.", file=sys.stderr)
            json_data = response.text # Save raw text if not JSON
        
        # Save the content (either parsed JSON or raw text)
        with open(filename, 'w', encoding='utf-8') as f:
             # If we successfully parsed JSON, dump it prettily, otherwise write text
            if isinstance(json_data, (dict, list)):
                 import json # Local import for json dump
                 json.dump(json_data, f, ensure_ascii=False, indent=4)
            else:
                 f.write(json_data) # Write raw text

        print(f"Successfully saved team stats to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching team stats for team {team_id}, season {season_id}: {e}", file=sys.stderr)
    except IOError as e:
        print(f"Error saving team stats file {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred fetching/saving team stats for team {team_id}, season {season_id}: {e}", file=sys.stderr)

def fetch_and_save_player_stats(player_id, team_id, output_dir="data/player_stats"):
    """Fetches player stats JSON for a given player_id (UUID) and team_id."""
    if not player_id or not isinstance(player_id, str):
        print(f"Skipping player stats fetch for invalid player_id: {player_id}")
        return
    if not team_id or not isinstance(team_id, str) or not team_id.isdigit():
        print(f"Skipping player stats fetch for invalid team_id: {team_id}")
        return

    # Construct the URL
    player_stats_url = f"https://msstats.optimalwayconsulting.com/v1/fcbq/player-stats/federated/{player_id}/team/{team_id}"
    filename = os.path.join(output_dir, f"player_{player_id}_team_{team_id}.json")

    # Avoid re-downloading
    if os.path.exists(filename):
        # print(f"Player stats JSON already exists for player {player_id}, team {team_id}: {filename}")
        return

    print(f"Fetching player stats for player {player_id}, team {team_id}...")
    try:
        response = requests.get(player_stats_url, timeout=15)
        response.raise_for_status()
        os.makedirs(output_dir, exist_ok=True)

        # Parse and save JSON
        try:
            json_data = response.json()
            with open(filename, 'w', encoding='utf-8') as f:
                import json
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            print(f"  Successfully saved player stats to {filename}")
        except requests.exceptions.JSONDecodeError:
            print(f"  Warning: Response for player {player_id} team {team_id} is not valid JSON. Skipping save.", file=sys.stderr)

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching player stats for player {player_id} team {team_id}: {e}", file=sys.stderr)
    except IOError as e:
        print(f"  Error saving player stats file {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  An unexpected error occurred for player {player_id} team {team_id}: {e}", file=sys.stderr)

def fetch_and_save_match_stats(match_id, output_dir="data/match_stats"):
    """Fetches aggregated match stats JSON for a given match_id and saves it."""
    if not match_id or match_id in ["N/A", "ERROR_EXTRACTING_ID"]:
        # print(f"Skipping aggregated stats fetch for invalid match_id: {match_id}")
        return

    # Note the different endpoint: getJsonWithMatchStats
    stats_url = f"https://msstats.optimalwayconsulting.com/v1/fcbq/getJsonWithMatchStats/{match_id}?currentSeason=true"
    # filename = os.path.join(output_dir, f"agg_{match_id}.json") # Add prefix to distinguish
    filename = os.path.join(output_dir, f"{match_id}.json") # Use standard name in new folder

    # Avoid re-downloading if file already exists
    if os.path.exists(filename):
        print(f"Match stats JSON already exists for match_id {match_id}: {filename}")
        return

    print(f"Fetching match stats for match_id {match_id} from {stats_url}...")
    try:
        response = requests.get(stats_url, timeout=20)
        response.raise_for_status()

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Attempt to parse JSON before saving
        try:
            json_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"  Warning: Response for match stats {match_id} is not valid JSON. Saving raw text.", file=sys.stderr)
            json_data = response.text

        # Save the content
        with open(filename, 'w', encoding='utf-8') as f:
            if isinstance(json_data, (dict, list)):
                import json
                json.dump(json_data, f, ensure_ascii=False, indent=4)
            else:
                f.write(json_data)

        print(f"Successfully saved match stats to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching match stats for match_id {match_id}: {e}", file=sys.stderr)
    except IOError as e:
        print(f"Error saving match stats file {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred fetching/saving match stats for {match_id}: {e}", file=sys.stderr)

def write_to_csv(data, group_id, output_dir="data"):
    """Writes the parsed data to a CSV file named with the group ID."""
    if not data:
        print("No data to write.", file=sys.stderr)
        return
    if not group_id:
        print("Cannot write CSV: Group ID is missing.", file=sys.stderr)
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # filename = os.path.join(output_dir, f"group_{group_id}_results.csv")
    filename = os.path.join(output_dir, f"results_{group_id}.csv") # New filename format

    print(f"Writing {len(data)} matches to {filename}...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Update header row
            writer.writerow([
                'Jornada', 'Date/Time', 'Local Team', 'Local Team ID', 
                'Visitor Team', 'Visitor Team ID', 'Score', 'Match ID'
            ])
            # Write data rows
            writer.writerows(data)
        print(f"Data successfully written to {filename}")
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}", file=sys.stderr)

if __name__ == "__main__":
    # --- Configuration ---
    # group_id = "18299" # <-- New Group ID
    group_ids_to_process = ["17182", "18299"] # List of group IDs
    base_output_dir = "data"
    season_id_to_fetch = "2024" # Define the season ID for team stats
    # --- End Configuration ---

    for current_group_id in group_ids_to_process:
        print(f"\n=== Processing Group: {current_group_id} ===")
        
        # Construct the URL based on the current group ID
        url = f"https://www.basquetcatala.cat/competicions/resultats/{current_group_id}/0"

        print(f"\n--- Step 1: Fetching and Parsing Schedule for Group {current_group_id} ---")
        # Get matches and group_id from the function
        extracted_data, extracted_group_id = fetch_and_parse_data(url)

        # Check if data extraction was successful for *this* group
        if extracted_data is not None and extracted_group_id is not None and extracted_group_id != "UNKNOWN":
            print(f"\n--- Step 2: Writing Schedule CSV for Group {extracted_group_id} ---")
            # Use the *extracted* group ID for saving, as it's verified from the URL structure
            write_to_csv(extracted_data, extracted_group_id, output_dir=base_output_dir)

            # Define indices and directories (could be outside loop, but clear here)
            match_id_index = 7 
            moves_output_dir = os.path.join(base_output_dir, "match_moves")
            stats_output_dir = os.path.join(base_output_dir, "match_stats")
            team_stats_output_dir = os.path.join(base_output_dir, "team_stats")
            
            valid_matches = [row for row in extracted_data if len(row) > match_id_index and row[match_id_index] not in ["N/A", "ERROR_EXTRACTING_ID"]]
            total_valid_matches = len(valid_matches)
            print(f"Found {total_valid_matches} matches with valid IDs in group {extracted_group_id}.")

            print(f"\n--- Step 3: Fetching Match Moves JSONs for Group {extracted_group_id} ---")
            count = 0
            for i, match_info in enumerate(extracted_data):
                if len(match_info) > match_id_index:
                    match_id = match_info[match_id_index]
                    if match_id and match_id not in ["N/A", "ERROR_EXTRACTING_ID"]:
                        count += 1
                        print(f"Processing Match Moves {count}/{total_valid_matches} (Row {i+1})...")
                        fetch_and_save_match_moves(match_id, output_dir=moves_output_dir)
                # else: Warning printed during parsing if needed
            print(f"\n--- Match Moves JSON Fetching Complete for Group {extracted_group_id} ---")

            print(f"\n--- Step 4: Fetching Match Stats JSONs for Group {extracted_group_id} ---")
            count = 0 # Reset counter
            for i, match_info in enumerate(extracted_data):
                if len(match_info) > match_id_index:
                    match_id = match_info[match_id_index]
                    if match_id and match_id not in ["N/A", "ERROR_EXTRACTING_ID"]:
                        count += 1
                        print(f"Processing Match Stats {count}/{total_valid_matches} (Row {i+1})...")
                        fetch_and_save_match_stats(match_id, output_dir=stats_output_dir)
                # else: Warning printed during parsing if needed
            print(f"\n--- Match Stats Fetching Complete for Group {extracted_group_id} ---")

            print(f"\n--- Step 5: Fetching Team Stats JSONs for Group {extracted_group_id} ---")
            print(f"Fetching stats for season: {season_id_to_fetch}")
            # Collect unique team IDs *from this group's schedule data*
            unique_team_ids = set()
            local_team_id_index = 3
            visitor_team_id_index = 5
            for row in extracted_data:
                if len(row) > max(local_team_id_index, visitor_team_id_index):
                    local_id = row[local_team_id_index]
                    visitor_id = row[visitor_team_id_index]
                    if local_id and isinstance(local_id, str) and local_id.isdigit():
                        unique_team_ids.add(local_id)
                    if visitor_id and isinstance(visitor_id, str) and visitor_id.isdigit():
                        unique_team_ids.add(visitor_id)
                
            print(f"Found {len(unique_team_ids)} unique team IDs in group {extracted_group_id}.")
            team_count = 0
            total_teams = len(unique_team_ids)
            for team_id in sorted(list(unique_team_ids)): # Sort for consistent order
                team_count += 1
                print(f"Processing Team Stats {team_count}/{total_teams} (ID: {team_id})...")
                fetch_and_save_team_stats(team_id, season_id_to_fetch, output_dir=team_stats_output_dir)
            
            print(f"\n--- Team Stats Fetching Complete for Group {extracted_group_id} ---")

        else:
            print(f"Failed to extract schedule data or group ID for group {current_group_id}. Skipping JSON fetching for this group.")
            continue # Move to the next group ID
    
    print("\n--- All Group Processing Complete ---")

    # --- Step 6: Fetch Player Stats (using downloaded team stats) ---
    print("\n--- Step 6: Fetching Player Stats from Team Data ---")
    team_stats_dir = os.path.join(base_output_dir, "team_stats")
    player_stats_output_dir = os.path.join(base_output_dir, "player_stats")
    processed_players_count = 0
    skipped_players_count = 0

    try:
        team_files = [f for f in os.listdir(team_stats_dir) if f.endswith(f'_season_{season_id_to_fetch}.json')]
    except FileNotFoundError:
        print(f"Error: Team stats directory not found: {team_stats_dir}", file=sys.stderr)
        team_files = []

    print(f"Found {len(team_files)} team stats files for season {season_id_to_fetch}.")

    for filename in team_files:
        team_filepath = os.path.join(team_stats_dir, filename)
        try:
            # Extract team_id from filename (e.g., team_70390_season_2024.json -> 70390)
            parts = filename.split('_')
            if len(parts) >= 3 and parts[0] == 'team':
                team_id = parts[1]
            else:
                print(f"  Warning: Could not extract team_id from filename {filename}. Skipping file.", file=sys.stderr)
                continue
            
            print(f"Processing players from team {team_id} ({filename})...")
            with open(team_filepath, 'r', encoding='utf-8') as f:
                team_data = json.load(f)
            
            # --- Find the player list/dictionary --- 
            # This requires knowing the structure of team_stats json.
            # Assumption 1: It's a dictionary where keys are player UUIDs.
            # Assumption 2: It might be a dictionary with a key like 'players' holding a list.
            # Let's try checking common structures.
            players_to_process = []
            if isinstance(team_data, dict):
                # Check if keys look like UUIDs (basic check: length 36, contains dashes)
                potential_uuids = [k for k in team_data.keys() if isinstance(k, str) and len(k) == 36 and '-' in k]
                if len(potential_uuids) > 0 and len(potential_uuids) == len(team_data):
                    print(f"  Found {len(potential_uuids)} players (assuming keys are UUIDs)...")
                    players_to_process = potential_uuids # Keys are the player UUIDs
                elif 'players' in team_data and isinstance(team_data['players'], list):
                     print(f"  Found 'players' list with {len(team_data['players'])} entries...")
                     # Assuming each item in the list is a dict with a 'uuid' key
                     for player_entry in team_data['players']:
                         if isinstance(player_entry, dict) and 'uuid' in player_entry:
                             players_to_process.append(player_entry['uuid'])
                         else:
                             print(f"    Warning: Found item in players list without 'uuid': {player_entry}", file=sys.stderr)
                elif 'generalStats' in team_data and 'uuid' in team_data['generalStats']:
                    # Check if the file structure is like the *player* stats example 
                    # (This means the team_stats URL might be wrong or returns single player data?)
                    print(f"  Warning: File {filename} looks like player stats, not team stats. Extracting single UUID.", file=sys.stderr)
                    players_to_process.append(team_data['generalStats']['uuid'])
                # Add more checks if needed based on actual structure
            
            if not players_to_process:
                print(f"  Warning: Could not find player UUIDs in expected format within {filename}. Skipping.", file=sys.stderr)
                continue
                
            # --- Fetch stats for each player --- 
            for player_id in players_to_process:
                if player_id:
                    fetch_and_save_player_stats(player_id, team_id, output_dir=player_stats_output_dir)
                    processed_players_count += 1
                else:
                    skipped_players_count += 1
                    print(f"  Warning: Encountered empty player_id for team {team_id}. Skipping.", file=sys.stderr)

        except FileNotFoundError:
            print(f"  Error: Could not find team stats file {team_filepath}", file=sys.stderr)
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {team_filepath}", file=sys.stderr)
        except Exception as e:
            print(f"  Error processing file {team_filepath}: {e}", file=sys.stderr)

    print(f"\n--- Player Stats Fetching Complete --- ({processed_players_count} players processed, {skipped_players_count} skipped) ---")
    # --- End Step 6 ---

    # else:
    #     print("Failed to extract schedule data or group ID. Cannot proceed to JSON fetching.")
