# EyeConnect MVP — запуск на ноутбуке (с экраном)

1. Проверка без камеры: `python -m eyeconnect.test_smoke` → SMOKE OK
2. Проверка камеры+трекинга: `python -m eyeconnect.main --mode preview` — сядь в 60см от камеры, Q — выход
3. Отладка клавиатуры мышью: `python -m eyeconnect.main --mode keyboard --lang KZ --mouse`
4. Калибровка взглядом: `python -m eyeconnect.main --mode calibrate` — смотри на 9 точек по 3с
5. Набор взглядом: `python -m eyeconnect.main --mode keyboard --lang KZ`, смена языка — зона LANG
6. Логи: `%USERPROFILE%\.eyeconnect\*.csv`, профиль: `%USERPROFILE%\.eyeconnect\gaze.npz`

На Pi: `pip install -r eyeconnect/requirements-pi.txt`, дальше те же команды.
