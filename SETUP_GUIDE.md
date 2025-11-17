# Setup Guide for Teammates

## Quick Start (5 minutes)

### Prerequisites
- Python 3.7 or higher
- VS Code (or any Python IDE)

### Step 1: Get the Files
Copy these files to your computer:
```
cassandra_bugs.csv              # The dataset
bug_resolution_analysis.py      # Data preprocessing
train_models.py                 # Model training
visualize_results.py            # Create charts
predict_new_bug.py              # Prediction tool
analyze_bugs.py                 # Initial exploration
README_THESIS.md                # Full documentation
THESIS_SUMMARY.md               # Quick summary
SETUP_GUIDE.md                  # This file
```

### Step 2: Install Required Libraries
Open terminal in VS Code (Ctrl+` or View > Terminal) and run:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**Or use this single command:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Step 3: Run the Analysis
Execute these commands in order:

```bash
# 1. Analyze and prepare data
python bug_resolution_analysis.py

# 2. Train models
python train_models.py

# 3. Create visualizations
python visualize_results.py

# 4. Try predictions (interactive)
python predict_new_bug.py
```

---

## Detailed Setup Instructions

### For Windows Users

1. **Open VS Code**
2. **Open the folder** containing all files (File > Open Folder)
3. **Open Terminal** (Ctrl+` or Terminal > New Terminal)
4. **Check Python version:**
   ```cmd
   python --version
   ```
   Should show Python 3.7 or higher

5. **Install libraries:**
   ```cmd
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

6. **Run scripts:**
   ```cmd
   python bug_resolution_analysis.py
   python train_models.py
   python visualize_results.py
   ```

### For Mac/Linux Users

Same as Windows, but use:
```bash
python3 --version
pip3 install pandas numpy scikit-learn matplotlib seaborn joblib
python3 bug_resolution_analysis.py
```

---

## What Each Script Does

### 1. `bug_resolution_analysis.py`
**Purpose:** Prepares data for modeling
**Runtime:** ~10 seconds
**Output:**
- `processed_bugs.csv` - Cleaned dataset with features
- Console output with statistics

**What it shows:**
- Total bugs analyzed
- Resolution time statistics
- Feature correlations
- Category breakdowns

### 2. `train_models.py`
**Purpose:** Trains 6 different ML models
**Runtime:** ~30 seconds
**Output:**
- `best_model.pkl` - Trained model (Gradient Boosting)
- `scaler.pkl` - Feature scaler
- `model_results.csv` - Performance comparison

**What it shows:**
- Model performance (MAE, RMSE, R²)
- Feature importance
- Best model selection

### 3. `visualize_results.py`
**Purpose:** Creates 7 visualization charts
**Runtime:** ~20 seconds
**Output:** 7 PNG files:
- `1_resolution_distribution.png` - Time distribution
- `2_actual_vs_predicted.png` - Model accuracy
- `3_residuals.png` - Error analysis
- `4_feature_importance.png` - What matters most
- `5_model_comparison.png` - Model rankings
- `6_temporal_trends.png` - Trends over time
- `7_category_analysis.png` - Bug categories

### 4. `predict_new_bug.py`
**Purpose:** Predict resolution time for new bugs
**Runtime:** Instant
**Output:** Interactive predictions

**How to use:**
1. Run the script
2. Enter bug details when prompted
3. Get prediction in days

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pandas'"
**Solution:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Problem: "python: command not found"
**Solution:**
Try `python3` instead:
```bash
python3 bug_resolution_analysis.py
```

### Problem: "Permission denied"
**Solution (Windows):**
Run as administrator or use:
```cmd
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Problem: Charts don't display
**Solution:**
Charts are saved as PNG files in the same folder. Open them with any image viewer.

### Problem: "UnicodeEncodeError"
**Solution:**
Already fixed! All Unicode characters replaced with ASCII.

### Problem: Slow execution
**Solution:**
Normal! Training 6 models on 3,413 bugs takes time. Grab coffee ☕

---

## File Structure After Running

```
your-folder/
├── cassandra_bugs.csv              # Original data
├── processed_bugs.csv              # Processed data (generated)
├── best_model.pkl                  # Trained model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── model_results.csv               # Results table (generated)
├── 1_resolution_distribution.png   # Chart (generated)
├── 2_actual_vs_predicted.png       # Chart (generated)
├── 3_residuals.png                 # Chart (generated)
├── 4_feature_importance.png        # Chart (generated)
├── 5_model_comparison.png          # Chart (generated)
├── 6_temporal_trends.png           # Chart (generated)
├── 7_category_analysis.png         # Chart (generated)
├── bug_resolution_analysis.py      # Script
├── train_models.py                 # Script
├── visualize_results.py            # Script
├── predict_new_bug.py              # Script
├── analyze_bugs.py                 # Script
├── README_THESIS.md                # Documentation
├── THESIS_SUMMARY.md               # Summary
└── SETUP_GUIDE.md                  # This file
```

---

## Running in VS Code (Step by Step)

### Method 1: Using Terminal
1. Open folder in VS Code
2. Open Terminal (Ctrl+`)
3. Type commands and press Enter

### Method 2: Using Run Button
1. Open any `.py` file
2. Click the ▶️ Run button (top right)
3. Or press F5

### Method 3: Right-click
1. Right-click on `.py` file
2. Select "Run Python File in Terminal"

---

## Expected Output

### After `bug_resolution_analysis.py`:
```
Loading data...
Total resolved bugs: 3413
Resolution time range: 0 - 1610 days

FEATURE ENGINEERING
Features created:
  - Text features: Summary_Length, Description_Length, Has_Description
  - Temporal features: Year, Month, DayOfWeek
  - Category flags: 8 categories
  - Encoded features: Priority, Status

EXPLORATORY ANALYSIS
1. Resolution Time by Priority:
                mean  median  count
Priority
High      109.061224    27.0     49
Low        84.064748    15.0    139
Normal     69.497161    12.0   3170
Urgent     39.745455     8.0     55

[OK] Data preparation complete!
```

### After `train_models.py`:
```
BUG RESOLUTION TIME PREDICTION - MODEL TRAINING

Loading processed data...
Training samples: 2730
Test samples: 683

MODEL TRAINING & EVALUATION

Gradient Boosting:
  Train MAE: 47.51 days
  Test MAE:  61.51 days
  Train RMSE: 70.14 days
  Test RMSE:  92.59 days
  Train R²: 0.479
  Test R²:  0.038

[OK] Best model (Gradient Boosting) saved to 'best_model.pkl'
[OK] Training complete!
```

### After `visualize_results.py`:
```
GENERATING VISUALIZATIONS

1. Creating resolution time distribution plot...
   [OK] Saved: 1_resolution_distribution.png
2. Creating actual vs predicted plot...
   [OK] Saved: 2_actual_vs_predicted.png
...
[OK] All visualizations created successfully!
```

---

## Tips for VS Code

### Useful Extensions
- **Python** (by Microsoft) - Essential
- **Pylance** - Better IntelliSense
- **Jupyter** - If you want notebooks

### Keyboard Shortcuts
- `Ctrl+` ` - Toggle terminal
- `F5` - Run with debugging
- `Ctrl+F5` - Run without debugging
- `Ctrl+Shift+P` - Command palette

### View Multiple Files
- Split editor: `Ctrl+\`
- Switch between files: `Ctrl+Tab`

---

## Common Questions

### Q: Do I need to run scripts in order?
**A:** Yes! Follow this order:
1. `bug_resolution_analysis.py` (creates processed_bugs.csv)
2. `train_models.py` (creates best_model.pkl)
3. `visualize_results.py` (needs model results)
4. `predict_new_bug.py` (needs trained model)

### Q: Can I run them multiple times?
**A:** Yes! They'll overwrite previous outputs.

### Q: How long does it take?
**A:** Total: ~1 minute
- Analysis: 10 seconds
- Training: 30 seconds
- Visualization: 20 seconds

### Q: What if I only have the CSV file?
**A:** You need all Python scripts too! Make sure you have:
- `cassandra_bugs.csv` (data)
- All `.py` files (scripts)

### Q: Can I modify the code?
**A:** Absolutely! The code is well-commented. Feel free to:
- Add more features
- Try different models
- Change visualizations
- Adjust parameters

### Q: What Python version do I need?
**A:** Python 3.7 or higher. Check with:
```bash
python --version
```

---

## Getting Help

### If something doesn't work:
1. **Check Python version:** `python --version`
2. **Reinstall libraries:** `pip install --upgrade pandas numpy scikit-learn matplotlib seaborn joblib`
3. **Check file location:** Make sure `cassandra_bugs.csv` is in the same folder
4. **Read error message:** It usually tells you what's wrong
5. **Google the error:** Copy-paste error message into Google

### Common Error Messages

**"FileNotFoundError: cassandra_bugs.csv"**
→ CSV file not in same folder as scripts

**"ModuleNotFoundError: No module named 'X'"**
→ Run: `pip install X`

**"SyntaxError"**
→ Python version too old. Need 3.7+

---

## Success Checklist

After running everything, you should have:
- ✅ `processed_bugs.csv` file
- ✅ `best_model.pkl` file
- ✅ `model_results.csv` file
- ✅ 7 PNG chart files
- ✅ Console output showing results
- ✅ No error messages

---

## Next Steps

Once everything runs successfully:
1. **Review the charts** - Open all PNG files
2. **Read THESIS_SUMMARY.md** - Understand findings
3. **Try predictions** - Run `predict_new_bug.py`
4. **Read README_THESIS.md** - Full documentation
5. **Start writing!** - Use findings for thesis

---

## Contact

If you're stuck, check:
1. This guide (SETUP_GUIDE.md)
2. Full documentation (README_THESIS.md)
3. Summary (THESIS_SUMMARY.md)

---

## That's It!

You're ready to go. Just:
1. Install libraries
2. Run scripts in order
3. View results

**Total time: 5 minutes setup + 1 minute execution = 6 minutes to complete analysis!**

Good luck! 🚀
