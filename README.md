# Sales Prediction API

A Flask-based REST API for predicting retail sales using a trained XGBoost model.

## 📁 Project Structure

```
sales-prediction-api/
│
├── app.py                      # Main Flask application
├── custom_transformers.py      # Custom sklearn transformers
├── train_model.py              # Model training script
├── predict_sales.py            # Standalone prediction script
├── requirements.txt            # Python dependencies
├── sales_model.joblib          # Trained model (generated)
│
├── templates/
│   └── index.html              # API documentation page
│
└── static/                     # (optional) CSS, JS, images
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
python train_model.py
```

This will create `sales_model.joblib` in your project directory.

### 3. Run the Flask API

```bash
python app.py
```

The API will start on `http://localhost:5000`

## 📡 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running",
  "model_loaded": true
}
```

### Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Store_id": "S001",
  "Store_Type": "A",
  "Location_Type": "Urban",
  "Region_Code": 1,
  "Date": "2024-01-15",
  "Holiday": 0,
  "Discount": 1
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "sales": 1234.56,
    "store_id": "S001",
    "date": "2024-01-15"
  },
  "input_data": { ... }
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "data": [
    {
      "Store_id": "S001",
      "Store_Type": "A",
      "Location_Type": "Urban",
      "Region_Code": 1,
      "Date": "2024-01-15",
      "Holiday": 0,
      "Discount": 1
    },
    {
      "Store_id": "S002",
      "Store_Type": "B",
      "Location_Type": "Rural",
      "Region_Code": 2,
      "Date": "2024-01-15",
      "Holiday": 0,
      "Discount": 0
    }
  ]
}
```

### CSV Upload
```http
POST /predict/csv
Content-Type: multipart/form-data
```

Upload a CSV file with required columns: `Store_id`, `Store_Type`, `Location_Type`, `Region_Code`, `Date`, `Holiday`, `Discount`

## 💻 Usage Examples

### Python
```python
import requests

# Single prediction
url = "http://localhost:5000/predict"
data = {
    "Store_id": "S001",
    "Store_Type": "A",
    "Location_Type": "Urban",
    "Region_Code": 1,
    "Date": "2024-01-15",
    "Holiday": 0,
    "Discount": 1
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Store_id": "S001",
    "Store_Type": "A",
    "Location_Type": "Urban",
    "Region_Code": 1,
    "Date": "2024-01-15",
    "Holiday": 0,
    "Discount": 1
  }'
```

### JavaScript (Fetch)
```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    Store_id: 'S001',
    Store_Type: 'A',
    Location_Type: 'Urban',
    Region_Code: 1,
    Date: '2024-01-15',
    Holiday: 0,
    Discount: 1
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📊 Required Input Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Store_id | string | Store identifier | "S001" |
| Store_Type | string | Type of store | "A", "B", "C" |
| Location_Type | string | Location category | "Urban", "Rural" |
| Region_Code | integer | Region code | 1, 2, 3 |
| Date | string | Date in YYYY-MM-DD | "2024-01-15" |
| Holiday | integer | Holiday indicator | 0, 1 |
| Discount | integer/string | Discount flag | 0, 1, "Yes", "No" |

## 🔧 Configuration

### Change Port
Edit `app.py`:
```python
app.run(host='0.0.0.0', port=8080, debug=True)
```

### Production Deployment
For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🐛 Troubleshooting

### Model Not Found
Ensure `sales_model.joblib` exists in the same directory as `app.py`. Run `train_model.py` to generate it.

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use
Change the port in `app.py` or kill the process using port 5000:
```bash
# Linux/Mac
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## 📝 Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "traceback": "Full traceback (in debug mode)"
}
```

## 🔒 Security Considerations

For production deployment:
- Disable debug mode: `app.run(debug=False)`
- Add authentication (JWT, API keys)
- Implement rate limiting
- Use HTTPS
- Validate and sanitize all inputs
- Set up CORS properly for web clients

## 📚 Additional Resources

- Flask Documentation: https://flask.palletsprojects.com/
- scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/

## 📄 License

MIT License