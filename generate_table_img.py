import json
import matplotlib.pyplot as plt
from pathlib import Path

def generate_table_image(json_path: Path, output_path: Path):
    """
    Renders the benchmarking results as a professional IEEE-style table image.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    metrics = data["Metric"]
    values = data["Value"]
    
    # Prepare data for table
    table_data = [[m, v] for m, v in zip(metrics, values)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off') # Hide axes
    
    col_labels = ['Performance Metric / Specification', 'Verified Experimental Value']
    
    # Create Table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left',
        colColours=['#f2f2f2', '#f2f2f2'],
        bbox=[0, 0, 1, 1]
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=[0, 1])
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black')
        cell.set_edgecolor('#dcdcdc')
        cell.set_linewidth(1)
        
    plt.title("Table 1: Meta-Model Performance & Framework Specifications", 
              fontsize=16, pad=20, weight='bold', family='serif')
    
    # Save Figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Formal table image saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    json_path = Path("docs/research_plots/ieee_benchmark_results.json")
    output_path = Path("docs/research_plots/table_1_model_summary.png")
    
    if json_path.exists():
        generate_table_image(json_path, output_path)
    else:
        print(f"❌ Error: {json_path} not found. Run benchmark_mlcli.py first.")
