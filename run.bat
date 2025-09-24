@echo off
REM run.bat — 1-clique para processar PDFs via Docker Compose no Windows
setlocal ENABLEDELAYEDEXPANSION

REM Checagens básicas
where docker >nul 2>nul
IF ERRORLEVEL 1 (
  echo Erro: docker nao encontrado no PATH.
  exit /b 1
)

REM docker compose v2
docker compose version >nul 2>nul
IF ERRORLEVEL 1 (
  echo Erro: 'docker compose' (v2) nao encontrado. Atualize o Docker Desktop.
  exit /b 1
)

REM Pastas de dados
if not exist "data" mkdir "data"
if not exist "data\Input" mkdir "data\Input"
if not exist "data\Output" mkdir "data\Output"
if not exist "data\OutputTxt" mkdir "data\OutputTxt"

echo ➡️  Construindo imagem e executando (isso pode levar alguns minutos na 1a vez)...
docker compose up --build --remove-orphans
IF ERRORLEVEL 1 (
  echo.
  echo ❌ Falhou. Verifique as mensagens acima.
  exit /b 1
)

echo.
echo ✅ Concluido.
echo 📂 PDFs processados:   %cd%\data\Output
echo 📝 Texto (OCR):        %cd%\data\OutputTxt

endlocal
