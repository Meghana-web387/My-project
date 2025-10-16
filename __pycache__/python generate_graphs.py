import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_classification_report_chart():
    """
    Generates a grouped bar chart for the classification report metrics
    and saves it to the static folder.
    """
    # 1. Manually parse the data from the text you provided
    data = {
        'A+': {'precision': 0.92, 'recall': 0.88, 'f1-score': 0.90, 'support': 200},
        'A-': {'precision': 0.81, 'recall': 0.93, 'f1-score': 0.86, 'support': 200},
        'B+': {'precision': 0.95, 'recall': 0.86, 'f1-score': 0.91, 'support': 200},
        'B-': {'precision': 0.96, 'recall': 0.94, 'f1-score': 0.95, 'support': 200},
        'AB+': {'precision': 0.89, 'recall': 0.85, 'f1-score': 0.87, 'support': 200},
        'AB-': {'precision': 0.95, 'recall': 0.84, 'f1-score': 0.89, 'support': 200},
        'O+': {'precision': 0.78, 'recall': 0.93, 'f1-score': 0.85, 'support': 200},
        'O-': {'precision': 0.87, 'recall': 0.85, 'f1-score': 0.86, 'support': 200},
    }

    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Drop the 'support' column for the chart, as it's not a performance score
    df = df.drop(columns=['support'])
    
    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up bar plot parameters
    blood_groups = df.index
    metrics = df.columns
    x = np.arange(len(blood_groups))  # the label locations
    width = 0.25  # the width of the bars

    # Create bars for each metric
    rects1 = ax.bar(x - width, df['precision'], width, label='Precision', color='#3F51B5')
    rects2 = ax.bar(x, df['recall'], width, label='Recall', color='#4CAF50')
    rects3 = ax.bar(x + width, df['f1-score'], width, label='F1-Score', color='#E91E63')

    # Add text labels on top of each bar
    def autolabel(rects):
        """Attach a text label above each bar in rects, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Add some text for titles, labels, and tick marks
    ax.set_xlabel('Blood Group', fontsize=12, labelpad=15)
    ax.set_ylabel('Score', fontsize=12, labelpad=15)
    ax.set_title('Model Classification Report', fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(blood_groups, fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a border around the plot
    plt.box(on=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3. Save the plot to the static folder
    static_folder = 'static'
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    
    file_path = os.path.join(static_folder, 'classification_report_bar_chart.png')
    plt.tight_layout()
    plt.savefig(file_path)
    
    print(f"Graph saved successfully to '{file_path}'")
    plt.close(fig) # Close the plot to free up memory

if __name__ == '__main__':
    create_classification_report_chart()
