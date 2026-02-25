import matplotlib.pyplot as plt

def plot_pareto_frontier(results, filename='pareto_frontier.png'):
    """
    Plot Pareto frontier of Accuracy vs Model Size.
    results: list of dicts with 'size_mb', 'accuracy', 'label'
    """
    sizes = [r['size_mb'] for r in results]
    accs = [r['accuracy'] for r in results]
    labels = [r.get('label', '') for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, accs, c='blue', alpha=0.6)
    
    # Highlight Pareto optimal points (simple heuristic: highest acc for given size range)
    # For now just plotting all points.
    
    for i, label in enumerate(labels):
        plt.annotate(label, (sizes[i], accs[i]))
        
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.title('Pareto Frontier: Accuracy vs Size')
    plt.grid(True)
    plt.savefig(filename)
    # plt.close() # Keep open if needed, or close to save memory
