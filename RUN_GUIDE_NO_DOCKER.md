# 🌞 Solar Panel Project - Beginner Run Guide (No Docker)

This guide explains how to run the Solar Panel Image Processing project directly on your Raspberry Pi. This method is faster than Docker but requires setting up the environment once.

---

## 🛠️ Step 1: One-Time Setup
*Do these steps only the FIRST time you set up the Pi.*

### 1. Install System Tools
Open your terminal (SSH) and run:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0 nodejs npm
```

### 2. Set Up the Backend (Python)
We create a "virtual environment" to keep our Python libraries safe and avoid version conflicts.

```bash
# Go to the project folder
cd ~/Developer/solar-panel-image-proc/backend_edge

# Create the virtual environment folder named 'venv'
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install the Raspberry Pi specific libraries
# This is the most important step!
pip install -r requirements-pi.txt
```
*(Note: If the install is slow, be patient—it is building math libraries for ARM64!)*

### 3. Set Up the Frontend (Node.js)
```bash
# Open a NEW terminal window (or Ctrl+C to stop the backend if running)
cd ~/Developer/solar-panel-image-proc/frontend

# Install all website dependencies (this takes a few minutes)
npm install

# Tell the frontend where the backend lives
# We use 'solar-panel.local' so your Windows PC can access it.
echo "NEXT_PUBLIC_API_URL=http://solar-panel.local:8000" > .env.local
```

---

## 🚀 Step 2: Daily Startup Routine
*Do this every time you restart the Raspberry Pi.*

You will need **TWO** terminal windows open (SSH into the Pi twice).

### Terminal 1: Start the Backend (API)
```bash
cd ~/Developer/solar-panel-image-proc/backend_edge
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*   **Wait** until you see: `Application startup complete.`
*   *If you see "ModuleNotFoundError", ensure you ran the pip install step above!*

### Terminal 2: Start the Frontend (Dashboard)
```bash
cd ~/Developer/solar-panel-image-proc/frontend
npm run dev
```
*   **Wait** until you see: `Ready on http://localhost:3000`

---

## 🌐 Step 3: View the Dashboard
On your Windows Computer:
1.  Open Chrome or Edge.
2.  Type this address: **http://solar-panel.local:3000**

You should see the dashboard!

---

## ❌ Troubleshooting

*   **"Address already in use"**: You might have the server running in another window. Run `pkill -f uvicorn` or `pkill -f node` to stop everything forcefully.
*   **"command not found: npm"**: Try running `source ~/.bashrc` or reinstall nodejs.
*   **Frontend can't see Backend**: Make sure you created the `.env.local` file in Step 1.3 correctly.
*   **Permission Denied (Docker)**: If you accidentally used a docker command, remember to use `newgrp docker` or reboot if you haven't yet.
