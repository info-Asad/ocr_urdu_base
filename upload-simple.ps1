# Simple GitHub Upload for Urdu OCR Project

Write-Host "🚀 Uploading Urdu OCR to GitHub" -ForegroundColor Green

# Initialize git if needed
if (-not (Test-Path ".git")) {
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
}

# Add files
git add .
Write-Host "📁 Files added" -ForegroundColor Green

# Commit
$commitMsg = "Complete Urdu OCR Neural Network System - CNN+LSTM+CTC Architecture"
git commit -m "$commitMsg"
Write-Host "💾 Committed changes" -ForegroundColor Green

Write-Host ""
Write-Host "🔗 Next steps:" -ForegroundColor Yellow
Write-Host "1. Create a new repository on GitHub:"
Write-Host "   - Go to https://github.com/new"
Write-Host "   - Name: urdu-ocr-neural-network"
Write-Host "   - Description: Complete Urdu OCR system using deep learning"
Write-Host "   - Choose Public or Private"
Write-Host "   - Don't initialize with README"
Write-Host ""
Write-Host "2. Copy your repository URL and run these commands:"
Write-Host "   git remote add origin YOUR_REPO_URL"
Write-Host "   git branch -M main"
Write-Host "   git push -u origin main"
Write-Host ""
Write-Host "🎉 Your amazing Urdu OCR project will be on GitHub!"