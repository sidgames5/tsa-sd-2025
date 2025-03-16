from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()


# Endpoint to serve the accuracy chart
@app.get("/accuracy-chart")
def get_accuracy_chart():
    file_path = "backend/static/accuracy_chart.png"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "Chart not found"}
