FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Streamlit
EXPOSE 8501

# bash : do build_index.py -> app
CMD ["bash", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
