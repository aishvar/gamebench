import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# Load the data from the JSON file
with open('model_ratings.json', 'r') as file:
    data = json.load(file)

# The "model_ratings" object is expected to be { "modelA": rating, "modelB": rating, ... }
rating_data = data.get('model_ratings', {})
num_rounds = len(data.get('processed_timestamps', []))

models = []
rating_values = []

# Loop through the ratings, mapping raw model names to nicer display names
for model_full, rating in rating_data.items():
    if 'Naive' in model_full:
        model_name = 'Naive 50/50 (Bot)'
    if 'Intelligent' in model_full:
        model_name = 'Intelligent (Bot)'
    elif 'optimus-alpha' in model_full:
        model_name = 'Optimus Alpha'
    elif '/' in model_full:
        # For models with provider/model format
        provider, model_part = model_full.split('/', 1)
        
        if 'deepseek-chat-v3-0324' in model_part:
            model_name = 'DeepSeek-V3 2025-03-24'
        elif 'deepseek-r1-distill-llama-70b' in model_part:
            model_name = 'DeepSeek R1 Llama 70B'
        elif 'deepseek-r1-distill-qwen-32b' in model_part:
            model_name = 'DeepSeek R1 Qwen 32B'
        elif 'deepseek-r1' in model_part:
            model_name = 'DeepSeek R1'
        elif 'gemini-2.5-pro-preview' in model_part:
            model_name = 'Gemini 2.5 Pro Preview'
        elif 'gemma-3-27b' in model_part:
            model_name = 'Gemma 3 27B'
        elif 'llama-4-maverick' in model_part:
            model_name = 'Llama 4 Maverick'
        elif 'llama-4-scout' in model_part:
            model_name = 'Llama 4 Scout'
        elif 'llama-3.1-8b' in model_part:
            model_name = 'Llama 3.1 8B'
        elif 'llama-3.3-70b' in model_part:
            model_name = 'Llama 3.3 70B'
        elif 'mistral-small' in model_part:
            model_name = 'Mistral Small 3.1'
        elif 'qwq-32b' in model_part:
            model_name = 'Qwen QwQ-32B'
        elif 'quasar-alpha' in model_part:
            model_name = 'Quasar Alpha'
        elif 'command-a' in model_part:
            model_name = 'Cohere Command A'
        else:
            model_name = model_part
    else:
        # For models without provider prefix
        if 'claude-3-7-sonnet' in model_full:
            model_name = 'Claude 3.7 Sonnet'
        elif 'claude-3-5-sonnet-20241022' in model_full:
            model_name = 'Claude 3.5 Sonnet 2024-10-22'
        elif 'gpt-4o-2024-11-20' in model_full:
            model_name = 'GPT-4o 2024-11-20'
        elif 'o3-mini' in model_full:
            model_name = 'o3-mini (high)'
        elif 'o1-2024' in model_full:
            model_name = 'o1 (high)'
        elif 'gemini-2.0-flash' in model_full:
            model_name = 'Gemini 2.0 Flash'
        else:
            model_name = model_full
    
    # Exclude specific models from the leaderboard
    if model_name in ('Optimus Alpha', 'Quasar Alpha'):
        continue
    models.append(model_name)
    rating_values.append(rating)

# Create a DataFrame and sort by rating (ascending => highest at bottom)
df = pd.DataFrame({
    'model': models,
    'rating': rating_values
})
df = df.sort_values('rating', ascending=True).reset_index(drop=True)

# Assign colors based on model family
colors = []
for model in df['model']:
    if 'DeepSeek' in model:
        colors.append('#5CAE80')  # Green for DeepSeek
    elif 'Gemini' in model or 'Gemma' in model:
        colors.append('#D174AC')  # Purple/Pink for Google models
    elif 'Llama' in model:
        colors.append('#F79F5E')  # Orange for Meta/Llama models
    elif 'Mistral' in model:
        colors.append('#9BE076')  # Light green for Mistral
    elif 'Claude' in model:
        colors.append('#FF6B6B')  # Red for Claude
    elif 'GPT' in model:
        colors.append('#7AB3EF')  # Blue for OpenAI
    elif 'Qwen' in model or 'QwQ' in model:
        colors.append('#FFC107')  # Yellow for Qwen
    elif 'Quasar' in model:
        colors.append('#9C27B0')  # Purple for Quasar
    else:
        colors.append('#7AB3EF')  # Default blue
df['color'] = colors

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#F8F9FB')  # Light background
ax.set_axisbelow(True)

# Plot horizontal bars
y_pos = np.arange(len(df))
for i, row in df.iterrows():
    ax.barh(y_pos[i], row['rating'], align='center', color=row['color'],
            height=0.7, alpha=0.8)

# Label each bar
for i, row in df.iterrows():
    # Model name on the left
    ax.text(-20, y_pos[i], row['model'], ha='right', va='center', fontsize=10)
    
    # Just display the rating on the bar
    label_text = f"({row['rating']:.1f})"
    ax.text(row['rating'] + 5, y_pos[i], label_text, va='center', fontsize=9, color='black')

# Customize axes
ax.set_yticks([])
ax.set_xlabel('Elo rating', fontsize=12)
ax.set_title('GameBench: Liars Poker - Elo Rankings', fontsize=16)

# Set x-axis limits and grid
x_max = max(df['rating']) + 80
ax.set_xlim(0, x_max)
ax.set_xticks(np.arange(0, int(x_max) + 1, 200))
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add game format info at bottom
plt.figtext(0.1, 0.02, 
            "Liar's Poker Format\n"
            "• 2 LLMs start each round\n"
            "• Players either raise the bet or call the prior player on a bluff\n"
            "• Winner is the one who calls a bluff correctly or isn't caught bluffing\n"
            "• Elo ratings are updated from each winner-loser pair\n",
            fontsize=8)

# Add totals and attribution
plt.figtext(0.9, 0.02, 
            f"Total Rounds: {num_rounds}\nBy Aishvar Radhakrishnan\ngamebench.ai", 
            fontsize=8, ha='right')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('model_rankings.png', dpi=300, bbox_inches='tight')
plt.show()