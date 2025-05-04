import csv
import json
import os
import sys
from collections import defaultdict # Import defaultdict

def load_schedule_csv(csv_filepath):
    """Loads schedule data from a CSV file into a list of dictionaries."""
    schedule = []
    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Check if required header 'Match ID' exists
            if 'Match ID' not in reader.fieldnames:
                print(f"Error: CSV file {csv_filepath} missing required header 'Match ID'.", file=sys.stderr)
                return None
            for row in reader:
                schedule.append(row)
        print(f"Successfully loaded {len(schedule)} rows from {csv_filepath}")
        return schedule
    except FileNotFoundError:
        print(f"Error: Schedule CSV file not found at {csv_filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}", file=sys.stderr)
        return None

def load_match_moves_json(match_id, json_dir="data/match_moves"):
    """Loads and parses the match moves (play-by-play) JSON data for a single match ID."""
    if not match_id or match_id in ["N/A", "ERROR_EXTRACTING_ID"]:
        return None # Skip invalid IDs

    json_filepath = os.path.join(json_dir, f"{match_id}.json")

    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            match_data = json.load(f)
            # print(f"  Successfully loaded JSON for {match_id}")
            return match_data
    except FileNotFoundError:
        print(f"  Warning: Match Moves JSON file not found for match_id {match_id} at {json_filepath}", file=sys.stderr)
        return None
    except json.JSONDecodeError:
        print(f"  Warning: Could not decode Match Moves JSON for match_id {match_id} from {json_filepath}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Warning: Error reading Match Moves JSON file {json_filepath}: {e}", file=sys.stderr)
        return None

def get_absolute_seconds(period, minute, second, period_length=600):
    # Assuming period is 1-based
    # Time in JSON is time *remaining* in the period
    seconds_elapsed_in_period = period_length - ((minute * 60) + second)
    seconds_from_prior_periods = (period - 1) * period_length
    return seconds_from_prior_periods + seconds_elapsed_in_period

def calculate_team_stats(target_team_id_str, schedule_data, all_games_data):
    """Calculates detailed statistics (including approx minutes) for a specific team."""
    print(f"\nCalculating stats for Team ID: {target_team_id_str} ...")
    
    # Aggregate stats
    player_stats = defaultdict(lambda: defaultdict(int))
    player_games_played = defaultdict(int)
    player_total_seconds = defaultdict(float) # Accumulate seconds played across games
    player_numbers = {} # Store {player_name: player_number} - Aggregate
    pairwise_seconds = defaultdict(lambda: defaultdict(float)) # Store {p1: {p2: seconds}} - Aggregate
    
    # Per-game storage
    all_games_pairwise_minutes = {} # {match_id: {'minutes': {...}, 'players': [...], 'numbers': {...}}}

    team_points = 0
    team_fouls = 0
    games_processed_count = 0
    games_analyzed_count = 0
    period_length_seconds = 600 # Assuming 10 minutes = 600 seconds per period

    points_map = {
        "Cistella de 1": 1,
        "Cistella de 2": 2,
        "Cistella de 3": 3,
    }
    foul_keywords = ["Personal", "Tècnica", "Antiesportiva", "Desqualificant"]

    print(f"Analyzing schedule to find matches and internal IDs for team {target_team_id_str}...")
    for match_info in schedule_data:
        match_id = match_info.get('Match ID')
        local_team_id_str = match_info.get('Local Team ID')
        visitor_team_id_str = match_info.get('Visitor Team ID')
        target_team_name = None

        if local_team_id_str == target_team_id_str:
            target_team_name = match_info.get('Local Team')
        elif visitor_team_id_str == target_team_id_str:
            target_team_name = match_info.get('Visitor Team')
        
        if target_team_name and match_id and match_id in all_games_data:
            games_processed_count += 1
            match_events = all_games_data[match_id]
            internal_target_id = None
            
            # --- Per-Game Tracking --- 
            current_game_player_seconds = defaultdict(float) # Total seconds for players in *this* game
            current_game_pairwise_seconds = defaultdict(lambda: defaultdict(float)) # Pairwise seconds for *this* game
            current_game_players = set() # Players who appeared in *this* game
            current_game_player_numbers = {} # Numbers for players in *this* game
            # {player_name: {'status': 'in'/'out', 'last_event_abs_seconds': seconds}}
            player_status_this_game = {} 
            on_court = set() # Players currently assumed to be on court
            current_period = 0
            last_update_abs_seconds = 0 # Track time for processing period ends/updates
            # --- End Per-Game Tracking --- 
            
            # Find internal ID
            for event in match_events:
                if event.get('teamAction') is True and event.get('actorName') == target_team_name:
                    internal_target_id = event.get('idTeam')
                    break 
            
            if internal_target_id is not None:
                games_analyzed_count += 1
                players_in_this_game = set() # Track players who played in this specific game

                # Sort events by timestamp to process chronologically for minutes calculation
                # If timestamp format is consistent (YYYYMMDDHHMMSS), string sort works
                # Handle potential None or non-string timestamps
                try:
                    sorted_events = sorted(match_events, key=lambda x: x.get('timestamp') or '')
                except TypeError:
                    print(f"  Warning: Could not sort events by timestamp for match {match_id}, minute calculation might be inaccurate.")
                    sorted_events = match_events # Process in original order if sort fails
                    
                for event in sorted_events:
                    event_period = event.get('period')
                    event_min = event.get('min')
                    event_sec = event.get('sec')
                    
                    # Ensure time components are valid integers
                    if not all(isinstance(t, int) for t in [event_period, event_min, event_sec]):
                        # print(f"  Skipping event due to invalid time data: {event}")
                        continue
                    
                    event_abs_seconds = get_absolute_seconds(event_period, event_min, event_sec, period_length_seconds)
                    duration_since_last_update = event_abs_seconds - last_update_abs_seconds

                    # --- Update Pairwise Time (Current Game and Aggregate) before changing on_court ---
                    if duration_since_last_update > 0 and len(on_court) > 1:
                        sorted_on_court = sorted(list(on_court)) # Ensure consistent pairing order
                        for i in range(len(sorted_on_court)):
                            p1 = sorted_on_court[i]
                            for j in range(i, len(sorted_on_court)): # Start from i to include self and avoid duplicates
                                p2 = sorted_on_court[j]
                                # Update current game pairwise
                                current_game_pairwise_seconds[p1][p2] += duration_since_last_update
                                # Update aggregate pairwise
                                pairwise_seconds[p1][p2] += duration_since_last_update
                                if p1 != p2:
                                    current_game_pairwise_seconds[p2][p1] += duration_since_last_update
                                    pairwise_seconds[p2][p1] += duration_since_last_update
                    # --- End Update Pairwise Time ---

                    move = event.get('move', '')
                    actor_name = event.get('actorName')
                    is_team_event = event.get('teamAction', False)
                    
                    # --- Period End/Start Logic --- 
                    if event_period > current_period:
                        # End of previous period detected
                        period_end_abs_seconds = current_period * period_length_seconds
                        duration_before_period_end = period_end_abs_seconds - last_update_abs_seconds

                        # --- Update Pairwise Time (Current Game and Aggregate) for end of period ---
                        if duration_before_period_end > 0 and len(on_court) > 1:
                            sorted_on_court = sorted(list(on_court))
                            for i in range(len(sorted_on_court)):
                                p1 = sorted_on_court[i]
                                for j in range(i, len(sorted_on_court)):
                                    p2 = sorted_on_court[j]
                                    # Update current game pairwise
                                    current_game_pairwise_seconds[p1][p2] += duration_before_period_end
                                    # Update aggregate pairwise
                                    pairwise_seconds[p1][p2] += duration_before_period_end
                                    if p1 != p2:
                                        current_game_pairwise_seconds[p2][p1] += duration_before_period_end
                                        pairwise_seconds[p2][p1] += duration_before_period_end
                        # --- End Update Pairwise Time ---
                        
                        # For players still 'in' at the end of the period, credit individual time until period end
                        for player, status_info in player_status_this_game.items():
                            if status_info['status'] == 'in':
                                ind_duration = period_end_abs_seconds - status_info['last_event_abs_seconds']
                                if ind_duration > 0:
                                     # current_game_player_seconds[player] += ind_duration # Individual total per game
                                     current_game_player_seconds[player] += ind_duration
                                     # player_total_seconds[player] += ind_duration # Aggregate total
                        # Reset for new period
                        player_status_this_game = {} 
                        on_court = set()
                        current_period = event_period
                        last_update_abs_seconds = (current_period - 1) * period_length_seconds # Start of new period
                        # print(f"--- Start Period {current_period} ---")
                   
                    # Check if event belongs to the target team
                    if event.get('idTeam') == internal_target_id:
                        if actor_name and not is_team_event:
                            # players_in_this_game.add(actor_name) # Add player to set for this game
                            current_game_players.add(actor_name) # Track players in this game
                           
                            # --- Store Player Number (Current Game and Aggregate) ---
                            player_num = event.get('actorShirtNumber') # Correct key
                            if player_num is not None:
                                if actor_name not in player_numbers: # Store first encountered globally
                                    player_numbers[actor_name] = player_num
                                if actor_name not in current_game_player_numbers: # Store first encountered in game
                                    current_game_player_numbers[actor_name] = player_num
                            # --- End Store Player Number ---
                           
                            # --- Minutes Calculation Logic --- 
                            # Infer starter if player has an action but isn't tracked as 'in' or 'out' yet in this period
                            if actor_name not in player_status_this_game and len(on_court) < 5: 
                                # Assume started period if they act before being subbed in
                                period_start_abs_seconds = (current_period - 1) * period_length_seconds
                                player_status_this_game[actor_name] = {'status': 'in', 'last_event_abs_seconds': period_start_abs_seconds}
                                on_court.add(actor_name)
                                # print(f"    Infer Start P{current_period}: {actor_name} at {period_start_abs_seconds}s")
                               
                            # Handle substitutions
                            if move == "Entra al camp":
                                if actor_name not in player_status_this_game or player_status_this_game[actor_name]['status'] == 'out':
                                    player_status_this_game[actor_name] = {'status': 'in', 'last_event_abs_seconds': event_abs_seconds}
                                    on_court.add(actor_name) # Add AFTER updating pairwise time based on previous state
                                    # print(f"    Sub In: {actor_name} at {event_abs_seconds}s")
                                elif actor_name in player_status_this_game and player_status_this_game[actor_name]['status'] == 'in':
                                    # Already marked as in, maybe redundant event, ignore time start
                                    pass 
                            elif move == "Surt del camp":
                                if actor_name in player_status_this_game and player_status_this_game[actor_name]['status'] == 'in':
                                    duration = event_abs_seconds - player_status_this_game[actor_name]['last_event_abs_seconds']
                                    if duration > 0:
                                        # current_game_player_seconds[actor_name] += duration # Individual total per game
                                        current_game_player_seconds[actor_name] += duration
                                        # player_total_seconds[actor_name] += duration # Aggregate total
                                    # print(f"    Sub Out: {actor_name} at {event_abs_seconds}s played {duration}s")
                                    player_status_this_game[actor_name] = {'status': 'out', 'last_event_abs_seconds': event_abs_seconds}
                                    if actor_name in on_court: on_court.remove(actor_name) # Remove AFTER updating pairwise time
                                else:
                                    # Player subbed out who wasn't tracked as in? Ignore.
                                    pass
                           
                            # --- End Minutes Logic ---
                           
                            # Calculate points and types
                            points_scored = points_map.get(move, 0)
                            if points_scored > 0:
                                team_points += points_scored
                                player_stats[actor_name]['points'] += points_scored
                                if points_scored == 1: player_stats[actor_name]['t1'] += 1
                                elif points_scored == 2: player_stats[actor_name]['t2'] += 1
                                elif points_scored == 3: player_stats[actor_name]['t3'] += 1
                           
                            # Calculate fouls
                            if any(keyword in move for keyword in foul_keywords):
                                team_fouls += 1
                                player_stats[actor_name]['fouls'] += 1
                   
                    # Update last event time for period end calculations
                    # last_event_abs_seconds = max(last_event_abs_seconds, event_abs_seconds)
                    last_update_abs_seconds = event_abs_seconds # Update after processing event and potential court changes
                   
                    # Handle final period end after loop if necessary (e.g., if JSON ends before period end event)
                # --- End Event Loop for Game --- 
               
                 # --- Update Pairwise Time (Current Game and Aggregate) for end of game data ---
                game_end_abs_seconds = max(last_update_abs_seconds, current_period * period_length_seconds)
                duration_before_game_end = game_end_abs_seconds - last_update_abs_seconds
                if duration_before_game_end > 0 and len(on_court) > 1:
                    sorted_on_court = sorted(list(on_court))
                    for i in range(len(sorted_on_court)):
                        p1 = sorted_on_court[i]
                        for j in range(i, len(sorted_on_court)):
                            p2 = sorted_on_court[j]
                            # Update current game pairwise
                            current_game_pairwise_seconds[p1][p2] += duration_before_game_end
                            # Update aggregate pairwise
                            pairwise_seconds[p1][p2] += duration_before_game_end
                            if p1 != p2:
                                current_game_pairwise_seconds[p2][p1] += duration_before_game_end
                                pairwise_seconds[p2][p1] += duration_before_game_end
                # --- End Update Pairwise Time ---

                # Add individual time for players still on court at the very end of the game data
                for player, status_info in player_status_this_game.items():
                    if status_info['status'] == 'in':
                        ind_duration = game_end_abs_seconds - status_info['last_event_abs_seconds']
                        if ind_duration > 0:
                             # current_game_player_seconds[player] += ind_duration # Individual total per game
                             current_game_player_seconds[player] += ind_duration
                             # player_total_seconds[player] += ind_duration # Aggregate total

                # Add this game's individual seconds to the overall total and increment GP
                for player_name, seconds in current_game_player_seconds.items():
                    player_total_seconds[player_name] += seconds # Aggregate individual time
                # for player_name in players_in_this_game:
                for player_name in current_game_players: # Use per-game player list
                    player_games_played[player_name] += 1
                    
                # --- Finalize and Store Per-Game Pairwise Minutes ---
                game_pairwise_minutes = defaultdict(lambda: defaultdict(int))
                game_player_list = sorted(list(current_game_players))
                game_total_minutes = {p: int(round(s / 60.0)) for p, s in current_game_player_seconds.items()}

                for p1 in game_player_list:
                    for p2 in game_player_list:
                        if p1 == p2:
                            game_pairwise_minutes[p1][p2] = game_total_minutes.get(p1, 0)
                        else:
                            # Use .get safely for potentially missing pairs if a player played 0 mins with another
                            game_pairwise_minutes[p1][p2] = int(round(current_game_pairwise_seconds.get(p1, {}).get(p2, 0) / 60.0))
                
                all_games_pairwise_minutes[match_id] = {
                    'minutes': game_pairwise_minutes, 
                    'players': game_player_list, 
                    'numbers': current_game_player_numbers
                }
                # --- End Finalize Per-Game ---

            else:
                print(f"  Warning: Could not determine internal ID for team '{target_team_name}' in match {match_id}. Stats for this game may be incomplete.", file=sys.stderr)

        elif target_team_name and match_id and match_id not in all_games_data:
            print(f"  Note: JSON data for match_id {match_id} (involving {target_team_name}) was not loaded.", file=sys.stderr)

    print(f"Found {games_processed_count} matches involving team {target_team_id_str} with loaded JSON data.")
    print(f"Successfully analyzed events for {games_analyzed_count} of these games after finding internal ID.")

    # Add GP and Mins to player_stats before returning
    final_player_stats = defaultdict(lambda: defaultdict(int))
    all_player_names = set(player_stats.keys()) | set(player_games_played.keys())
    for name in all_player_names:
        final_player_stats[name] = player_stats[name]
        final_player_stats[name]['gp'] = player_games_played[name]
        # Convert minutes to integer after rounding
        final_player_stats[name]['mins'] = int(round(player_total_seconds[name] / 60.0))
        # Add player number
        final_player_stats[name]['number'] = player_numbers.get(name, '??') # Use '??' if number not found
    
    # Convert *aggregate* pairwise seconds to minutes
    aggregate_pairwise_minutes = defaultdict(lambda: defaultdict(int))
    all_player_names_aggregate = set(pairwise_seconds.keys()) # Get all players involved in aggregate pairwise
    for p1_agg in all_player_names_aggregate:
        for p2_agg in all_player_names_aggregate:
             # Ensure the diagonal reflects total player minutes for consistency
            if p1_agg == p2_agg:
                 # Use final_player_stats which has the rounded aggregate minutes
                aggregate_pairwise_minutes[p1_agg][p2_agg] = final_player_stats.get(p1_agg, {}).get('mins', 0)
            else:
                aggregate_pairwise_minutes[p1_agg][p2_agg] = int(round(pairwise_seconds.get(p1_agg, {}).get(p2_agg, 0) / 60.0))

    # Prepare results
    stats = {
        "team_id": target_team_id_str,
        "total_points": team_points,
        "total_fouls": team_fouls,
        "player_stats": final_player_stats,
        # "pairwise_minutes": pairwise_minutes # Add pairwise data - Now use aggregate
        "aggregate_pairwise_minutes": aggregate_pairwise_minutes,
        "all_games_pairwise_minutes": all_games_pairwise_minutes # Add per-game data
    }
    return stats

if __name__ == "__main__":
    # --- Configuration ---
    group_ids_to_process = ["17182", "18299"] # List of group IDs to process
    base_data_dir = "data"
    # json_subdir = "match_stats" # Updated subdirectory name
    moves_subdir = "match_moves" # Renamed subdirectory
    # json_base_directory = os.path.join(base_data_dir, json_subdir)
    moves_base_directory = os.path.join(base_data_dir, moves_subdir)
    # --- End Configuration ---

    print("--- Starting Data Processing for Multiple Groups ---")

    all_schedule_data = {}
    all_json_data = {}
    total_json_loaded = 0
    total_json_skipped = 0

    # --- Loop through each group ID ---
    for group_id in group_ids_to_process:
        print(f"\n=== Processing Group: {group_id} ===\n")
        schedule_csv_file = os.path.join(base_data_dir, f"group_{group_id}_results.csv")
        current_group_schedule = None
        current_group_json = {}
        loaded_count = 0
        skipped_count = 0

        # 1. Load the schedule CSV for the current group
        print(f"Step 1: Loading schedule from {schedule_csv_file}...")
        current_group_schedule = load_schedule_csv(schedule_csv_file)
        
        if current_group_schedule is None:
            print(f"Warning: Could not load schedule for group {group_id}. Skipping JSON loading for this group.", file=sys.stderr)
            all_schedule_data[group_id] = None # Mark as unloaded
            continue # Move to the next group ID
        else:
            all_schedule_data[group_id] = current_group_schedule
        
        # 2. Load Match Moves JSON for each match in the current group
        print(f"Step 2: Loading Match Moves JSON data from {moves_base_directory}...")
        for i, match_info in enumerate(current_group_schedule):
            match_id = match_info.get('Match ID')
            # Re-enable print for debugging
            print(f"  Processing row {i+1}/{len(current_group_schedule)}: Match ID '{match_id}' ...", end='')
            if match_id and match_id not in ["N/A", "ERROR_EXTRACTING_ID"]:
                # game_json_data = load_match_json(match_id, json_base_directory)
                game_moves_json_data = load_match_moves_json(match_id, moves_base_directory)
                # if game_json_data is not None:
                if game_moves_json_data is not None:
                    # current_group_json[match_id] = game_json_data
                    current_group_json[match_id] = game_moves_json_data
                    loaded_count += 1
                    # Re-enable print for debugging
                    print(" Loaded.")
                else:
                    skipped_count += 1
                    # Re-enable print for debugging
                    print(" Skipped (Moves JSON load error/missing).")
            else:
                skipped_count += 1
                # Re-enable print for debugging
                print(" Skipped (Invalid/Missing Match ID).")

        all_json_data[group_id] = current_group_json
        total_json_loaded += loaded_count
        total_json_skipped += skipped_count
        print(f"Finished loading Moves JSON for group {group_id}. Loaded: {loaded_count}, Skipped: {skipped_count}")

    # --- End Group Loop ---

    print("\n--- Overall Processing Complete ---")
    print(f"Successfully loaded schedule data for {len([s for s in all_schedule_data.values() if s is not None])} groups.")
    print(f"Successfully loaded Match Moves JSON data for {total_json_loaded} matches across all groups.")
    print(f"Skipped {total_json_skipped} Match Moves JSON entries total.")

    # 3. Calculate and Print Stats for the target team in EACH processed group
    target_team_id_to_analyze = "69630" # Team ID to analyze

    print(f"\n--- Calculating Statistics for Team {target_team_id_to_analyze} in Each Group --- ")

    for group_id in group_ids_to_process:
        print(f"\n--- Group: {group_id} ---")
        if group_id in all_schedule_data and group_id in all_json_data and all_schedule_data[group_id] is not None:
            print(f"Calculating stats for Team {target_team_id_to_analyze} using data from Group {group_id} ...")
            # Pass the specific group's loaded data to the stats function
            team_stats = calculate_team_stats(
                target_team_id_to_analyze, 
                all_schedule_data[group_id], 
                all_json_data[group_id]
            )

            if team_stats:
                print(f"\n--- Statistics for Team {team_stats['team_id']} (from Group {group_id}) ---")
                print(f"Total Points Scored: {team_stats['total_points']}")
                print(f"Total Fouls Committed: {team_stats['total_fouls']}")
                
                # --- Print Player Stats Table --- 
                player_stats_dict = team_stats.get('player_stats')
                if player_stats_dict:
                    # Sort players by points descending
                    sorted_players = sorted(player_stats_dict.items(), key=lambda item: item[1].get('points', 0), reverse=True)
                    
                    # Determine column widths
                    # Calculate max length needed for "(NN) Name" format
                    max_name_len = 0
                    for name, stats_data in sorted_players:
                        num_str = str(stats_data.get('number', '??'))
                        formatted_name = f"({num_str}) {name}"
                        max_name_len = max(max_name_len, len(formatted_name))

                    name_col_width = max(max_name_len, len("Player")) + 2 # Add padding
                    gp_col_width = 4 # Width for GP
                    mins_col_width = 6 # Width for Mins
                    num_col_width = 6 # Width for PTS, T3, T2, T1, Fouls

                    # Print header
                    header = f"{'Player'.ljust(name_col_width)}" + \
                             f"{'GP'.rjust(gp_col_width)}" + \
                             f"{'Mins'.rjust(mins_col_width)}" + \
                             f"{'PTS'.rjust(num_col_width)}" + \
                             f"{'T3'.rjust(num_col_width)}" + \
                             f"{'T2'.rjust(num_col_width)}" + \
                             f"{'T1'.rjust(num_col_width)}" + \
                             f"{'Fouls'.rjust(num_col_width)}"
                    print("\nPlayer Stats:")
                    print(header)
                    print("-" * len(header))
                    
                    # Print player rows
                    for player_name, stats in sorted_players:
                        player_num_str = str(stats.get('number', '??'))
                        formatted_player_name = f"({player_num_str}) {player_name}"
                        row = f"{formatted_player_name.ljust(name_col_width)}" + \
                              f"{str(stats.get('gp', 0)).rjust(gp_col_width)}" + \
                              f"{str(stats.get('mins', 0)).rjust(mins_col_width)}" + \
                              f"{str(stats.get('points', 0)).rjust(num_col_width)}" + \
                              f"{str(stats.get('t3', 0)).rjust(num_col_width)}" + \
                              f"{str(stats.get('t2', 0)).rjust(num_col_width)}" + \
                              f"{str(stats.get('t1', 0)).rjust(num_col_width)}" + \
                              f"{str(stats.get('fouls', 0)).rjust(num_col_width)}"
                        print(row)
                       
                    # --- Print Aggregate Pairwise Minutes Table --- 
                    aggregate_pairwise_minutes_data = team_stats.get('aggregate_pairwise_minutes')
                    if aggregate_pairwise_minutes_data and sorted_players:
                        print("\n--- Aggregate Minutes Played Together ---")
                        
                        # Use the same sorted player list and numbers from individual stats
                        player_names_ordered = [name for name, stats in sorted_players]
                        player_numbers_ordered = [str(stats.get('number', '??')) for name, stats in sorted_players]
                        
                        # Column width based on player numbers (assume max 3 digits + padding)
                        pair_col_width = 5 
                        # Header row width (player name column + space for each player number col)
                        max_name_width_pair_table = max(len(f"({num}) {name}") for num, name in zip(player_numbers_ordered, player_names_ordered)) + 2

                        # Print header (Player Numbers)
                        header1 = " ".ljust(max_name_width_pair_table) # Empty space for player name column
                        for num in player_numbers_ordered:
                            header1 += str(num).rjust(pair_col_width)
                        print(header1)
                        print("-" * len(header1))

                        # Print player rows
                        for i, p1_name in enumerate(player_names_ordered):
                            p1_num = player_numbers_ordered[i]
                            row_label = f"({p1_num}) {p1_name}"
                            row = row_label.ljust(max_name_width_pair_table)
                            for j, p2_name in enumerate(player_names_ordered):
                                # Fetch minutes played together, default to 0 if no entry
                                mins_together = aggregate_pairwise_minutes_data.get(p1_name, {}).get(p2_name, 0)
                                row += str(mins_together).rjust(pair_col_width)
                            print(row)

                    else:
                        print("\nCould not generate aggregate pairwise minutes table.")
                    # --- End Print Aggregate Pairwise Table --- 

                    # --- Print Top 5 Player Aggregate Pairwise Minutes --- 
                    if aggregate_pairwise_minutes_data and sorted_players and len(sorted_players) >= 1:
                        print("\n--- Top 5 Player Aggregate Pairwise Minutes ---")
                        top_5_players = sorted_players[:5] # Get the top 5 players

                        for top_player_name, top_player_stats in top_5_players:
                            top_player_num = str(top_player_stats.get('number', '??'))
                            print(f"\nPlayer: ({top_player_num}) {top_player_name} (Total Mins: {top_player_stats.get('mins', 0)})")
                            print("Played With:")
                            
                            # Sort teammates by minutes played *with* the top player
                            teammates_minutes = []
                            for other_player_name, other_player_stats in sorted_players:
                                if other_player_name == top_player_name:
                                    continue # Skip self
                                
                                mins_together = aggregate_pairwise_minutes_data.get(top_player_name, {}).get(other_player_name, 0)
                                other_player_num = str(other_player_stats.get('number', '??'))
                                teammates_minutes.append({
                                    'name': other_player_name, 
                                    'number': other_player_num, 
                                    'mins': mins_together
                                })
                            
                            # Sort by minutes descending
                            sorted_teammates = sorted(teammates_minutes, key=lambda x: x['mins'], reverse=True)
                            
                            # Print sorted list
                            max_teammate_name_len = 0
                            if sorted_teammates:
                                max_teammate_name_len = max(len(f"  ({t['number']}) {t['name']}") for t in sorted_teammates)
                            
                            name_width = max(max_teammate_name_len, len("  Player")) + 2
                            mins_width = 6

                            print(f"  {'Player'.ljust(name_width)}{ 'Mins'.rjust(mins_width)}")
                            print(f"  {'-' * name_width}{'-' * mins_width}-")

                            for teammate in sorted_teammates:
                                teammate_label = f"  ({teammate['number']}) {teammate['name']}"
                                print(f"{teammate_label.ljust(name_width+2)}{str(teammate['mins']).rjust(mins_width)}")
                    
                    # --- End Top 5 Player Aggregate Pairwise --- 

                    # --- Print Per-Game Pairwise Minutes ---
                    all_games_pairwise_data = team_stats.get('all_games_pairwise_minutes')
                    current_schedule_data = all_schedule_data.get(group_id, [])
                    schedule_lookup = {m.get('Match ID'): m for m in current_schedule_data if m.get('Match ID')}

                    if all_games_pairwise_data:
                        print("\n--- Per-Game Pairwise Minutes ---")

                        # Sort games by Match ID for consistent order, though schedule order might be better if available
                        # Let's try sorting by schedule order if possible
                        game_ids_in_schedule_order = [m.get('Match ID') for m in current_schedule_data if m.get('Match ID') in all_games_pairwise_data]
                        processed_game_ids = set(game_ids_in_schedule_order)
                        # Add any remaining games not found in schedule (shouldn't happen ideally)
                        remaining_game_ids = sorted([gid for gid in all_games_pairwise_data if gid not in processed_game_ids])
                        ordered_game_ids = game_ids_in_schedule_order + remaining_game_ids

                        # for match_id, game_data in all_games_pairwise_data.items():
                        for match_id in ordered_game_ids:
                            game_data = all_games_pairwise_data[match_id]
                            game_minutes = game_data['minutes']
                            game_players_list = game_data['players'] # Already sorted alphabetically now
                            game_player_numbers = game_data['numbers']

                            match_details = schedule_lookup.get(match_id, {})
                            local_team = match_details.get('Local Team', 'Unknown')
                            visitor_team = match_details.get('Visitor Team', 'Unknown')
                            game_date = match_details.get('Date/Time', 'Unknown Date') # Use 'Date/Time' from the CSV header

                            print(f"\n=== Game: {match_id} ({game_date}) - {local_team} vs {visitor_team} ===")

                            if not game_players_list:
                                print("No players recorded for this game.")
                                continue

                            # Prepare player numbers and names for this game's table
                            current_game_p_nums = [str(game_player_numbers.get(p, '??')) for p in game_players_list]
                            current_game_p_names = game_players_list

                            # Calculate widths for this specific game's table
                            game_pair_col_width = 5 
                            max_game_name_width = 0
                            if current_game_p_nums:
                                max_game_name_width = max(len(f"({num}) {name}") for num, name in zip(current_game_p_nums, current_game_p_names)) + 2

                            # Print header
                            game_header = " ".ljust(max_game_name_width) 
                            for num in current_game_p_nums:
                                game_header += str(num).rjust(game_pair_col_width)
                            print(game_header)
                            print("-" * len(game_header))

                            # Print rows
                            for i, p1_name in enumerate(current_game_p_names):
                                p1_num = current_game_p_nums[i]
                                row_label = f"({p1_num}) {p1_name}"
                                row = row_label.ljust(max_game_name_width)
                                for j, p2_name in enumerate(current_game_p_names):
                                    mins_together = game_minutes.get(p1_name, {}).get(p2_name, 0)
                                    row += str(mins_together).rjust(game_pair_col_width)
                                print(row)
                    
                    # --- End Per-Game Pairwise ---

                else:
                    print("\nNo player statistics data found for this team in this group.")
            else:
                print(f"Could not calculate stats for team {target_team_id_to_analyze} in group {group_id} (may not exist in this group or calculation failed).")

        else:
            print(f"Skipping stats calculation: Data for group {group_id} was not successfully loaded.")

    print(f"\nOverall Processing Complete ---")
    print(f"Successfully loaded schedule data for {len([s for s in all_schedule_data.values() if s is not None])} groups.")
    print(f"Successfully loaded Match Moves JSON data for {total_json_loaded} matches across all groups.")
    print(f"Skipped {total_json_skipped} Match Moves JSON entries total.")

    # Now you have all_schedule_data and all_json_data available
    # for further comparison or analysis between groups.
    # Example: print(f"\nNumber of games loaded for group 17182: {len(all_json_data.get('17182', {}))}\")\n    # Example: print(f\"Number of games loaded for group 18299: {len(all_json_data.get('18299', {}))}\")\n
