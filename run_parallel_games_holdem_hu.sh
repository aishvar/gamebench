#!/bin/bash
# Usage: ./run_parallel_games_holdem_hu.sh [count]
# Example: ./run_parallel_games_holdem_hu.sh 5

COUNT=${1:-2}  # default to 2 games if not specified

# stdin for heads_up_texas_hold_em_game.py:
#   line1: Player 1 choice (N, R or number)
#   line2: Player 2 choice (N, R or number)
#   line3: How many paired rounds? [1]

CONFIG_INPUT=$'R\nR\n4'

TEMP_DIR="./.tmp_hu_game_status"
mkdir -p "$TEMP_DIR"
rm -f "$TEMP_DIR"/game_*.done

echo "🎮 Launching $COUNT Heads‑Up Hold’em games in separate Terminal tabs…"

for i in $(seq 1 $COUNT); do
  osascript <<EOF
tell application "Terminal"
  activate
  do script "cd \"$(pwd)\"; echo '▶️ Starting HU Game #$i'; python3 heads_up_texas_hold_em_game.py <<< \"$CONFIG_INPUT\"; echo '✅ HU Game #$i done'; echo done > '$TEMP_DIR/game_$i.done'; exec bash"
  set custom title of selected tab of front window to "HU Game #$i"
end tell
EOF
done

echo "🚀 All $COUNT games launched. Monitoring progress…"

while true; do
  completed=$(ls -1 "$TEMP_DIR"/*.done 2>/dev/null | wc -l | xargs)
  echo -ne "\r⏳ Games completed: $completed / $COUNT"
  [ "$completed" -ge "$COUNT" ] && break
  sleep 2
done

echo -e "\n✅ All $COUNT games completed."
rm -rf "$TEMP_DIR"
