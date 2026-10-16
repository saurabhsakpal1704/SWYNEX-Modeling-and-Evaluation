# SWYNEX — Task 3: Modeling and Evaluation

## Project
A reproducible baseline classification project using the public UCI Adult Census Income dataset. It predicts whether the dataset's income label is `>50K` or `<=50K` and evaluates logistic regression against a majority-class baseline.

## Files
- `SWYNEX_Modeling_and_Evaluation.ipynb` — complete notebook with explanation, training, evaluation, plots, and limitations.
- `modeling_and_evaluation.py` — runnable Python script version.
- `requirements.txt` — Python dependencies.
- `reports/` — created when the project runs.
- `models/` — trained model created when the project runs.

## Run locally (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/. During setup, tick **Add Python to PATH**.
2. Extract this project ZIP to a folder you can find, such as `Documents\SWYNEX-Modeling-and-Evaluation`.
3. Open that folder in File Explorer. Click the address bar, type `cmd`, and press Enter.
4. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run the script:
   ```bash
   python modeling_and_evaluation.py
   ```
   It downloads the public dataset and prints the evaluation metrics. An internet connection is required.
7. To open the notebook:
   ```bash
   jupyter notebook
   ```
   In the browser tab that opens, select `SWYNEX_Modeling_and_Evaluation.ipynb`, then choose **Kernel → Restart & Run All**. Check that every cell completes and that the metric table and plots appear.

## Run in Google Colab (no local installation)
1. Open https://colab.research.google.com/.
2. Choose **New notebook**.
3. In the left sidebar, click the folder icon (Files), then the upload icon.
4. Upload `SWYNEX_Modeling_and_Evaluation.ipynb` from the extracted project folder.
5. Open the uploaded notebook and select **Runtime → Run all**.
6. Wait for all cells to finish. The dataset download requires internet access.

## Upload to GitHub
1. Sign in to https://github.com/ and click the **+** icon → **New repository**.
2. Repository name: `SWYNEX-Modeling-and-Evaluation`.
3. Choose **Public** if the internship reviewer needs to open the notebook without special access. Do not add secrets or private data.
4. Click **Create repository**.
5. On the repository page, choose **Add file → Upload files**.
6. Extract the ZIP first, then drag the project files and folders into the upload area. Upload the notebook, Python script, README, requirements, and `.gitignore`. Do not upload `.venv/` or generated `.joblib` files.
7. Click **Commit changes**.
8. Open the notebook in the GitHub repository. Copy its URL, which should look like:
   `https://github.com/YOUR-USERNAME/SWYNEX-Modeling-and-Evaluation/blob/main/SWYNEX_Modeling_and_Evaluation.ipynb`
   Replace `YOUR-USERNAME` with your actual GitHub username. Do not submit this example URL as if it were your real repository.

## Submit on SWYNEX
1. Return to the Task 3 submission panel.
2. In **Task submission link**, paste the real GitHub notebook URL (or the repository URL if the platform accepts it).
3. In **LinkedIn explanation post link**, paste the URL of your published LinkedIn post describing what you built/learned.
4. Before submitting, open both links in a separate browser tab and confirm they are accessible.
5. Click **Submit task** once both required fields are complete.
6. Wait for the platform's reviewer to check the submission; the screenshot says the next task unlocks after review.

## LinkedIn post template
Edit this so it accurately reflects what you actually ran and learned, then publish it and use the real post URL:

> Completed Task 3 of my SWYNEX internship: Modeling and Evaluation. I built a baseline logistic-regression classification pipeline using the public UCI Adult Census Income dataset, handled missing values, encoded categorical features, and evaluated the model using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. I also compared it with a majority-class baseline and documented limitations of the dataset and model. The results and notebook are available here: [paste your GitHub notebook URL]
>
> @SWYNEX Technologies #SWYNEX #Internship #MachineLearning #Python #DataScience

Only state that you completed the task after you have run the notebook and checked the outputs. Tag the official company page only if you have verified it is the correct page.

## Notes
- The dataset is downloaded from UCI at runtime; the first run requires internet access.
- The notebook's metric values are generated when you run it. Do not write a made-up accuracy in the report.
- This is an educational demonstration. The data includes demographic attributes and historical bias; do not use the model to make real decisions about people.
