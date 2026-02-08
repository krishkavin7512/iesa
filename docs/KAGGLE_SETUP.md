# Kaggle API Setup Guide

## Step 1: Create Kaggle Account
1. Go to https://www.kaggle.com/
2. Click "Register" and create an account
3. Verify your email

## Step 2: Generate API Token
1. Log in to Kaggle
2. Click on your profile picture (top right) → "Settings"
3. Scroll down to "API" section
4. Click "Create New Token"
5. This downloads `kaggle.json` file

## Step 3: Place kaggle.json
1. Create folder: `C:\Users\<YourUsername>\.kaggle\`
2. Copy the downloaded `kaggle.json` to this folder
3. The file should be at: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

## Step 4: Verify Setup
Run this command in terminal:
```bash
kaggle datasets list
```
If it shows datasets, you're good to go!

## Quick Commands for Windows PowerShell
```powershell
# Create .kaggle folder
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"

# After downloading kaggle.json, copy it:
# Move the downloaded kaggle.json to C:\Users\YourName\.kaggle\kaggle.json
```

## Datasets We Need
1. **MixedWM38**: https://www.kaggle.com/datasets/qingyi/mixedwm38
2. **Severstal Steel**: https://www.kaggle.com/c/severstal-steel-defect-detection
