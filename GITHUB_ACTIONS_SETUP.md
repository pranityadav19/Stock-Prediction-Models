# GitHub Actions Setup Guide

This guide will help you set up automated predictions via GitHub Actions.

## Step 1: Create a Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the **Google Sheets API** and **Google Drive API**:
   - Go to "APIs & Services" → "Enable APIs and Services"
   - Search for "Google Sheets API" and enable it
   - Search for "Google Drive API" and enable it
4. Create a Service Account:
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Name it something like `github-actions-sheets`
   - Click "Create and Continue" → "Done"
5. Create a key for the service account:
   - Click on your new service account
   - Go to "Keys" tab → "Add Key" → "Create new key"
   - Choose JSON format
   - Save the downloaded JSON file (you'll need it in Step 3)

## Step 2: Share Your Google Sheets

Your service account has an email like `github-actions-sheets@your-project.iam.gserviceaccount.com`.

Share your Google Sheets with this email (Editor access):
- NBA PrizePicks Edge Tracker
- Stock Prediction Tracker

## Step 3: Add GitHub Secrets

For **each repository** (NBA-Betting and Stock-Prediction-Models):

1. Go to your repo on GitHub
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret" and add:

| Secret Name | Value |
|-------------|-------|
| `GOOGLE_CREDENTIALS` | Entire contents of the JSON key file from Step 1 |
| `SENDGRID_API_KEY` | Your SendGrid API key: `SG.q5wa_5C-...` |

## Step 4: Push the Workflow Files

The workflow files are already created. Just push them to GitHub:

```bash
# For NBA project
cd /Users/pranityadav/Downloads/NBA-Betting
git add .github/
git commit -m "Add GitHub Actions for automated predictions"
git push

# For Stock project
cd /Users/pranityadav/Downloads/Stock-Prediction-Models
git add .github/
git commit -m "Add GitHub Actions for automated predictions"
git push
```

## Step 5: Verify It's Working

1. Go to your repo on GitHub → "Actions" tab
2. You should see the workflows listed
3. Click on a workflow → "Run workflow" to test manually
4. Check your Google Sheets for updates

## Schedule Summary

| Job | Time (CST) | Time (UTC) | Days |
|-----|------------|------------|------|
| NBA Predictions | 3:00 AM | 9:00 AM | Daily |
| NBA Results | 1:15 AM | 7:15 AM | Daily |
| Stock Predictions | 6:30 AM | 12:30 PM | Mon-Fri |
| Stock Results | 3:30 PM | 9:30 PM | Mon-Fri |

## Troubleshooting

**Workflow not running?**
- Check Actions tab for error logs
- Verify secrets are set correctly
- Ensure the JSON credentials are pasted in full (including braces)

**Google Sheets not updating?**
- Verify you shared the sheet with the service account email
- Check the workflow logs for authentication errors

**Emails not sending?**
- Verify SENDGRID_API_KEY is set correctly
- Check SendGrid dashboard for API errors
