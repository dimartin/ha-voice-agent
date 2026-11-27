FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ha_voice_agent ./ha_voice_agent
# aiohttp pinned to the version the image was last tested with, so a rebuild changes nothing.
RUN pip install --no-cache-dir . aiohttp==3.14.5
COPY config ./config
ENV PORT=8011 VOICE_LOG=/data/voice-log.jsonl
EXPOSE 8011
CMD ["ha-voice-agent"]
