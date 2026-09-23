FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements-demo.txt requirements-reports.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY . .
RUN python verify_results.py \
    && python main_simulation.py --demo \
    && chmod -R a+rX data outputs preview \
    && useradd --create-home --uid 1000 app \
    && mkdir -p outputs/reports \
    && chown app:app outputs/reports
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8080')+'/healthz',timeout=3)"
CMD ["gunicorn", "--config", "gunicorn.conf.py", "web_app:app"]
