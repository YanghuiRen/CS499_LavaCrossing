### This is the Repo for CS499 Minigrid Final Project

**Prerequisites:**
- Python 3.9+ (tested with Python 3.13 on macOS, but earlier versions like 3.10, 3.11 should also work)
- `pip`
- `git`

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <URL_OF_YOUR_GITHUB_REPO>
    cd CS499_LavaCrossing  # Or whatever your repo directory is named
    ```

2.  **Create and activate a virtual environment:**

    *   **Using `venv` (standard Python, recommended for cross-platform consistency):**
        ```bash
        # Navigate to the directory 옆에 where you want to store the venv, e.g., inside Final_project but outside CS499_LavaCrossing
        # cd ../  # If you are inside CS499_LavaCrossing
        # python3 -m venv name_of_your_venv  # e.g., python3 -m venv crossing_lava_env
        # source name_of_your_venv/bin/activate  # On macOS/Linux
        # name_of_your_venv\Scripts\activate  # On Windows
        ```
        Or, create it inside the project directory (it will be ignored by git if `venv/` or `.*env/` is in `.gitignore`):
        ```bash
        # Inside CS499_LavaCrossing directory
        python3 -m venv venv
        source venv/bin/activate      # On macOS/Linux
        # venv\Scripts\activate         # On Windows (cmd.exe)
        # venv\Scripts\Activate.ps1     # On Windows (PowerShell) - you might need to set execution policy
        ```

    *   **Using Conda (if preferred by team members):**
        ```bash
        conda create -n crossing_lava_env python=3.10  # Or your preferred Python version
        conda activate crossing_lava_env
        ```

3.  **Install dependencies:**
    Once your virtual environment is activated, navigate to the root of the `CS499_LavaCrossing` repository (if not already there) and run:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary packages, including `gymnasium` and `minigrid`.

## Running the Test Script

To verify that the environment is set up correctly, you can run the test script:
```bash
# Ensure your virtual environment is activated
# Navigate to the CS499_LavaCrossing directory
python src/test.py
```

### Algorithms to be Implemented


### Timeline (Tentative):
- Week 7: Algorithems Implementation
- Week 8: Training Part A (Q-Learning)
- Week 9: Training Part B (Q-Learning-λ)
- Week 10: Report

### Team Members:
- Megan Dorn
- Brandyn Tucknott
- Yanghui Ren

## License
MIT: <https://rem.mit-license.org>