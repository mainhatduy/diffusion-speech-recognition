# Diffusion Speech Recognition

A speech recognition project using Diffusion Models. To run the project quickly and efficiently, you need to set up the appropriate virtual environment.

---

## 🚀 Virtual Environment Setup Guide on Linux

Here is a detailed guide on how to create and activate a local virtual environment directly on your Linux operating system without relying on any external machine or device.

### 📦 Method 1: Using Default Python `venv` (Independent & Recommended for Linux)

This method uses the Python built-in library, ensuring simplicity, independence, and maximum compatibility across all Linux machines.

#### **Step 1: Install prerequisites for Linux**
On Linux distributions like Ubuntu/Debian, the `venv` package is often not included by default with Python. Install it using the following command:
```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

#### **Step 2: Clone the project and navigate to the directory**
```bash
git clone <url-of-github-repo>
cd diffusion-speech-recognition
```

#### **Step 3: Create a virtual environment**
Initialize a virtual environment folder named `.venv` in the project directory:
```bash
python3 -m venv .venv
```

#### **Step 4: Activate the virtual environment**
Run the following command to activate the virtual environment:
```bash
source .venv/bin/activate
```
> [!TIP]
> Once successfully activated, your terminal will display the `(.venv)` prefix at the beginning of the prompt.
> To exit the virtual environment after you are done working, simply run: `deactivate`.

#### **Step 5: Upgrade pip and install libraries**
Run the following command to upgrade the `pip` installation tool and download all required libraries from the `requirements.txt` file:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### ⚡ Method 2: Using `uv` (High Speed)

If you want a blazing fast installation process using Astral's modern tool `uv`:

#### **Step 1: Install `uv`**
* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows:**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### **Step 2: Sync the environment**
Navigate to the project directory and run:
```bash
uv sync
```
`uv` will automatically download the appropriate Python version (if missing), create the `.venv` directory, and install all required libraries.

---

## 💻 Running the Project

After completing the installation steps above:

* **If installed using Python `venv` (Method 1):**
  Activate the virtual environment and run the main file:
  ```bash
  source .venv/bin/activate
  python main.py
  ```

* **If installed using `uv` (Method 2):**
  Run directly without manually activating the virtual environment:
  ```bash
  uv run python main.py
  ```

---

## 📂 Git Push Guidelines

When pushing source code to GitHub, ensure the following configuration and dependency definition files are committed:
* `requirements.txt`
* `pyproject.toml`
* `uv.lock`
* `.python-version`

> [!IMPORTANT]
> **Never** commit the `.venv/` directory. This folder contains gigabytes of local downloaded library source code and is already configured to be ignored in the project's `.gitignore` file.
