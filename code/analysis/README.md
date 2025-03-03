# analysis

To run the code in this folder, please ensure you have downloaded all data for the [data folder](../../data/).  
You may need to slightly adjust file paths as needed.

### Scripts Overview:
- **`parsing_tree.py`**: Calculates syntax metrics based on parsing results in the **CoNLL-U** format.
- **`measure.py`**: Computes all syntax metrics reported in the paper and performs the **Mann-Kendall trend tests**.
- **`analysis.py`**: Generates tables (e.g., majority vote trends tables) and plots using the outputs from `measure.py`.
- **`plot_trend.py`**: Plots diachronic trend figures (**Figures 14 & 15**); the `trends_to_plot.json` file used is an output of `analysis.py`.





