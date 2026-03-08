@echo off
set PY313=C:\Users\vbcma\AppData\Local\Programs\Python\Python31312\python.exe
if exist "%PY313%" (
  "%PY313%" "%~dp0desktop_keyer.py"
  goto :eof
)
python "%~dp0desktop_keyer.py"
