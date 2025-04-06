import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# Load the data from the JSON file
with open('model_ratings.json', 'r') as file:
    data = json.load(file)

# Extract model data
models = []
mu_values = []
sigma_values = []
games = [] #to add later 

for model_full, ratings in data['model_ratings'].items():
    # Extract and format model name for cleaner display
    if 'deepseek-chat-v3' in model_full:
        model_name = 'DeepSeek-V3'
    elif 'deepseek-r1' in model_full:
        model_name = 'DeepSeek R1'
    elif 'gemini-2.5-pro' in model_full:
        model_name = 'Gemini 2.5 Pro Exp 03-25'
    elif 'gemma-3-27b' in model_full:
        model_name = 'Gemma 3 27B'
    elif 'llama-4-maverick' in model_full:
        model_name = 'Llama 4 Maverick'
    elif 'llama-4-scout' in model_full:
        model_name = 'Llama 4 Scout'
    elif 'mistral-small' in model_full:
        model_name = 'Mistral Small 3.1'
    else:
        model_name = model_full.split('/')[1].split(':')[0]
    
    # Extract values from the JSON
    mu = ratings['mu']
    sigma = ratings['sigma']
    
    
    models.append(model_name)
    mu_values.append(mu)
    sigma_values.append(sigma)

# Create DataFrame and sort by mu (descending)
df = pd.DataFrame({
    'model': models,
    'mu': mu_values,
    'sigma': sigma_values
})
df = df.sort_values('mu', ascending=False).reset_index(drop=True)

# Assign colors based on model family (similar to reference image)
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
    else:
        colors.append('#7AB3EF')  # Blue for others
df['color'] = colors

# Get number of tournaments from the timestamps
num_tournaments = len(data.get('processed_timestamps', []))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#F8F9FB')  # Light background like in original
ax.set_axisbelow(True)

# Plot horizontal bars
y_pos = np.arange(len(df))
for i, row in df.iterrows():
    ax.barh(y_pos[i], row['mu'], align='center', color=row['color'], 
            height=0.7, alpha=0.8)
    
    # Add error bars
    ax.errorbar(row['mu'], y_pos[i], xerr=row['sigma'], fmt='none', ecolor='black', 
                capsize=5, capthick=1.5, elinewidth=1.5)

# Add model names and stats
for i, row in df.iterrows():
    # Model name on the left
    ax.text(-0.5, y_pos[i], row['model'], ha='right', va='center', fontsize=10)
    
    # Stats on the bar
    stats_text = f"({row['mu']:.1f} ±{row['sigma']:.1f})"
    ax.text(0.2, y_pos[i], stats_text, va='center', fontsize=9, 
            color='white' if row['mu'] > 25 else 'black')

# Customize axes
ax.set_yticks([])
ax.set_xlabel('μ (± σ)', fontsize=12)
ax.set_title('GameBench: Liars Poker - TrueSkill Rankings', fontsize=16)

# Set x-axis limits and grid
x_max = max(df['mu'] + df['sigma']) + 2
ax.set_xlim(0, x_max)
ax.set_xticks(range(0, int(x_max) + 1, 5))
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add game information at bottom
plt.figtext(0.1, 0.02, 
            "Liar's Poker Format\n"
            "• >2 LLMs start each round\n"
            "• Players can either represent and guess a certain amount of numbers or call the prior player on a bluff\n"
            "• The winner is the player who either calls a bluff correctly or has a bluff incorrectly called against them\n"
            "• TrueSkill ratings are based on each game's winner and loser(s)'",
            fontsize=8)

# Add totals and attribution
plt.figtext(0.9, 0.02, f"Total Tournaments: {num_tournaments}\nBy Aishvar Radhakrishnan", fontsize=8, ha='right')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('model_rankings.png', dpi=300, bbox_inches='tight')
plt.show()
