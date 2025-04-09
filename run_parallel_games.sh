#!/bin/bash
# Usage: ./run_parallel_games.sh [count]
# Example: ./run_parallel_games.sh 10

COUNT=${1:-2}  # Default to 2 if not provided

CONFIG_INPUT=$'2\nR\nC\nopenai\ngpt-4.5-preview\n1'

#CONFIG_INPUT=$'2\nR\n19\n1'

TEMP_DIR="./.tmp_game_status"

mkdir -p "$TEMP_DIR"
rm -f "$TEMP_DIR"/game_*.done

echo "🎮 Launching $COUNT Liar's Poker games in separate Terminal tabs..."

for i in $(seq 1 $COUNT); do
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \"$(pwd)\"; echo '▶️ Starting Game #$i'; python3 liars_poker_game.py <<< \"$CONFIG_INPUT\"; echo '✅ Game #$i done'; echo done > '$TEMP_DIR/game_$i.done'; exec bash"
    set custom title of selected tab of front window to "Game #$i"
end tell
EOF
done

echo "🚀 All $COUNT games launched. Monitoring progress..."

completed=0
while [ $completed -lt $COUNT ]; do
  completed=$(ls -1 "$TEMP_DIR"/*.done 2>/dev/null | wc -l | xargs)
  echo -ne "\r⏳ Games completed: $completed / $COUNT"
  sleep 2
done

echo -e "\n✅ All $COUNT games completed."
rm -rf "$TEMP_DIR"