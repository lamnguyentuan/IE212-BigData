
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install python deps
COPY requirements-crawl.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Browser is already installed in this image

# Copy code
COPY . /app

# Set env
ENV PYTHONUNBUFFERED=1
ENV HEADLESS=true

CMD ["tail", "-f", "/dev/null"]
