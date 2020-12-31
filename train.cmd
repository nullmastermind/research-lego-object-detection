@echo off
set /p data=--data:
set /p epochs=--epochs:
python train.py --weights train_data/best.pt --data train_data/%data%/%data%.yaml --batch 16 --epochs %epochs%
pause