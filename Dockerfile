# 1. Use a newer Python image that matches your libraries
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY src/ src/
COPY data/ data/
COPY model.pkl .

# 6. Expose the port
EXPOSE 8000

# 7. Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]