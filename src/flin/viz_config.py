# Imports for plotting configuration
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Register the Formula 1 custom font
f1_path = "fonts/formula1-regular-web-0.ttf"
fm.fontManager.addfont(f1_path)
f1_name = fm.FontProperties(fname=f1_path).get_name()

# Define background (BG) and foreground (FG) colors
BG, FG = '#06003A', 'white'

# Matplotlib rcParams for consistent styling
dark_params = {
    'figure.facecolor': BG,
    'axes.facecolor'  : BG,
    'axes.edgecolor'  : FG,
    'axes.labelcolor' : FG,
    'xtick.color'     : FG,
    'ytick.color'     : FG,
    'text.color'      : FG,
    'grid.color'      : FG,
    'grid.alpha'      : 0.1,
    'font.family'     : 'sans-serif',
    'font.sans-serif' : [f1_name, "DejaVu Sans", "Arial"],
    'font.weight'     : 'regular',

    # boxplot line styling
    'boxplot.boxprops.color'             : FG,
    'boxplot.boxprops.linewidth'         : 1.0,
    'boxplot.capprops.color'             : FG,
    'boxplot.capprops.linewidth'         : 1.0,
    'boxplot.whiskerprops.color'         : FG,
    'boxplot.whiskerprops.linewidth'     : 1.0,
    'boxplot.flierprops.markeredgecolor' : FG,
    'boxplot.flierprops.markerfacecolor' : FG,
    'boxplot.medianprops.color'          : FG,
    'boxplot.medianprops.linewidth'      : 1.0,
}

# Apply the theme to Seaborn and Matplotlib
sns.set_theme(style="whitegrid", palette="husl", rc=dark_params)
plt.rcParams.update(dark_params)
