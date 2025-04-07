import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# Load the data from the JSON file
with open('model_ratings.json', 'r') as file:
    data = json.load(file)

# Multiplier for mu and sigma values
multiplier = 40

# Extract model data
models = []
mu_values = []
sigma_values = []

for model_full, ratings in data['model_ratings'].items():
    # Extract and format model name for cleaner display
    if 'Naive' in model_full:
        model_name='Naive 50/50'
    elif '/' in model_full:
        # For models with provider/model format
        provider, model_part = model_full.split('/', 1)
        
        if 'deepseek-chat-v3' in model_part:
            model_name = 'DeepSeek-V3'
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
        else:
            model_name = model_part
    else:
        # For models without provider prefix
        if 'claude-3-7-sonnet' in model_full:
            model_name = 'Claude 3.7 Sonnet'
        elif 'claude-3-5-sonnet' in model_full:
            model_name = 'Claude 3.5 Sonnet'
        elif 'gpt-4o-2024-11-20' in model_full:
            model_name = 'GPT-4o 2024-11-20'
        elif 'gemini-2.0-flash' in model_full:
            model_name = 'Gemini 2.0 Flash'
        else:
            model_name = model_full
    
    # Extract values from the JSON and multiply by 40
    mu = ratings['mu'] * multiplier
    sigma = ratings['sigma'] * multiplier
    
    models.append(model_name)
    mu_values.append(mu)
    sigma_values.append(sigma)

# Create DataFrame and sort by mu (ascending to put highest values at the bottom)
df = pd.DataFrame({
    'model': models,
    'mu': mu_values,
    'sigma': sigma_values
})
df = df.sort_values('mu', ascending=True).reset_index(drop=True)  # Sort ascending

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

# Get number of rounds from the timestamps
num_rounds = len(data.get('processed_timestamps', []))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor('#F8F9FB')  # Light background
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
    ax.text(-20, y_pos[i], row['model'], ha='right', va='center', fontsize=10)
    
    # Stats on the bar with multiplied values
    stats_text = f"({row['mu']:.1f} ±{row['sigma']:.1f})"
    ax.text(8, y_pos[i], stats_text, va='center', fontsize=9, 
            color='white' if row['mu'] > 1000 else 'black')

# Customize axes
ax.set_yticks([])
ax.set_xlabel('μ (± σ)', fontsize=12)
ax.set_title('GameBench: Liars Poker - TrueSkill Rankings', fontsize=16)

# Set x-axis limits and grid (adjusted for multiplied values)
x_max = max(df['mu'] + df['sigma']) + 80
ax.set_xlim(0, x_max)
ax.set_xticks(np.arange(0, int(x_max) + 1, 200))
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add game information at bottom
plt.figtext(0.1, 0.02, 
            "Liar's Poker Format\n"
            "• 2 LLMs start each round\n"
            "• Players can either represent and guess a certain amount of numbers or call the prior player on a bluff\n"
            "• The winner is the player who either calls a bluff correctly or has a bluff incorrectly called against them\n"
            "• TrueSkill ratings are based on each game's winner and loser",
            fontsize=8)

# Add totals and attribution
plt.figtext(0.9, 0.02, 
            f"Total Rounds: {num_rounds}\nBy Aishvar Radhakrishnan\ngamebench.ai", 
            fontsize=8, ha='right')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('model_rankings.png', dpi=300, bbox_inches='tight')
plt.show()