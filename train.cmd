@echo off
set /p directory=directory:
set /p epochs=--epochs:
python train.py --weights train_data/best.pt --data train_data/%directory%/%directory%.yaml --batch 16 --epochs %epochs%
pause