import os
import sys
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "ui", "app.py")

    subprocess.call(
        f'"{sys._base_executable}" -m streamlit run "{app_path}"',
        shell=True
    )

if __name__ == "__main__":
    main()