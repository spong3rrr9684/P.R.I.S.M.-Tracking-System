import urllib.request
import os
import sys

def run_auto_update():
    print("\n--- P.R.I.S.M. AUTO-UPDATER ---")
    
    # SAFEGUARD: Do not auto-update if this is the Developer's machine
    # We check if the hidden '.git' folder exists in this directory. 
    # Normal users who download the .zip won't have a '.git' folder.
    if os.path.exists(".git"):
        print("[INFO] Developer Environment Detected (.git found).")
        print("[INFO] Skipping Auto-Update to protect your un-pushed local code.")
        print("-------------------------------\n")
        return

    print("Checking GitHub for the latest P.R.I.S.M. code...")

    # The raw GitHub URL for your repository
    base_url = "https://raw.githubusercontent.com/spong3rrr9684/P.R.I.S.M.-Tracking-System/main/"
    
    # The python scripts that make up the core app
    files_to_update = [
        "main.py",
        "renderer.py",
        "tracker.py",
        "hud_modes.py",
        "utils.py",
        "state.py",
        "voice_assistant.py",
        "config.py",
        "ui_components.py"
    ]

    updated_count = 0
    for filename in files_to_update:
        url = base_url + filename
        try:
            # Download the raw text from GitHub
            response = urllib.request.urlopen(url, timeout=5)
            if response.status == 200:
                cloud_data = response.read().decode('utf-8')
                
                # Check if local file exists and compare contents to avoid unnecessary writes
                local_data = ""
                if os.path.exists(filename):
                    with open(filename, 'r', encoding='utf-8') as f:
                        local_data = f.read()

                # Only overwrite if the cloud version is different
                if cloud_data != local_data:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(cloud_data)
                    print(f"  [+] Updated {filename} to the latest version.")
                    updated_count += 1
                else:
                    # They already have the latest version of this file
                    pass
            else:
                print(f"  [!] Failed to check {filename} (HTTP {response.status})")
        except Exception as e:
            # If they don't have internet, just skip and launch the local version
            print(f"  [WARNING] Could not connect to GitHub. Skipping updates.")
            break

    if updated_count > 0:
        print(f"[SUCCESS] Downloaded {updated_count} new updates from GitHub!")
    else:
        print("[OK] Your software is 100% up to date.")
    print("-------------------------------\n")

if __name__ == "__main__":
    run_auto_update()
