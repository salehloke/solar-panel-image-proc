# SolarAI Backend

A FastAPI-based backend service for solar panel dirt detection using deep learning.

## Features

- 🚀 **FastAPI** - Modern, fast web framework
- 🐘 **PostgreSQL** - Robust relational database
- 🤖 **PyTorch** - Deep learning model inference
- 🔐 **JWT Authentication** - Secure user authentication
- 📊 **Async Database Operations** - High-performance data access
- 🐳 **Docker Support** - Easy deployment and scaling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   PostgreSQL    │    │   Redis Cache   │
│                 │◄──►│                 │    │                 │
│ - ML Inference  │    │ - User Data     │    │ - Sessions      │
│ - API Endpoints │    │ - Analysis      │    │ - Caching       │
│ - Auth          │    │ - Solar Panels  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Database Schema

### Users Table

- `id` (UUID) - Primary key
- `email` (VARCHAR) - Unique email
- `username` (VARCHAR) - Unique username
- `password_hash` (VARCHAR) - Hashed password
- `role` (VARCHAR) - User role (user/admin/moderator)
- `created_at` (TIMESTAMP) - Creation time
- `updated_at` (TIMESTAMP) - Last update time

### Solar Panels Table

- `id` (UUID) - Primary key
- `user_id` (UUID) - Foreign key to users
- `name` (VARCHAR) - Panel name
- `latitude` (DECIMAL) - GPS latitude
- `longitude` (DECIMAL) - GPS longitude
- `address` (TEXT) - Physical address
- `capacity` (DECIMAL) - Panel capacity in kW
- `installation_date` (DATE) - Installation date
- `created_at` (TIMESTAMP) - Creation time
- `updated_at` (TIMESTAMP) - Last update time

### Analysis Results Table

- `id` (UUID) - Primary key
- `user_id` (UUID) - Foreign key to users
- `image_path` (TEXT) - Path to uploaded image
- `prediction` (VARCHAR) - Prediction result (clean/dirty)
- `confidence` (DECIMAL) - Confidence score (0-1)
- `model_version` (VARCHAR) - Model version used
- `processing_time` (DECIMAL) - Processing time in seconds
- `created_at` (TIMESTAMP) - Creation time

## API Endpoints

### Health & Info

- `GET /` - API information
- `GET /health` - Health check

### ML Prediction

- `POST /predict` - Predict solar panel dirt from image

### User Data

- `GET /users/{user_id}/analysis` - Get user's analysis results
- `GET /users/{user_id}/panels` - Get user's solar panels
- `GET /stats` - Get analysis statistics

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)

### Using Docker (Recommended)

1. **Clone and navigate to the project**

   ```bash
   cd solar-image-processing/backend
   ```

2. **Set up environment variables**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the services**

   ```bash
   docker-compose -f ../docker-compose.backend.yml up -d
   ```

4. **Check the services**

   ```bash
   # API Gateway
   curl http://localhost:8000/health

   # pgAdmin (optional)
   # Open http://localhost:8081
   # Login: admin@solarai.com / admin123
   ```

### Local Development

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL**

   ```bash
   # Install PostgreSQL locally or use Docker
   docker run -d --name postgres \
     -e POSTGRES_DB=solarai \
     -e POSTGRES_USER=solarai \
     -e POSTGRES_PASSWORD=solarai123 \
     -p 5432:5432 \
     postgres:15
   ```

3. **Run the application**
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Environment Variables

| Variable            | Description                  | Default                                                         |
| ------------------- | ---------------------------- | --------------------------------------------------------------- |
| `DATABASE_URL`      | PostgreSQL connection string | `postgresql+asyncpg://solarai:solarai123@postgres:5432/solarai` |
| `POSTGRES_PASSWORD` | PostgreSQL password          | `solarai123`                                                    |
| `JWT_SECRET`        | JWT secret key               | `your-secret-key-here`                                          |
| `JWT_ALGORITHM`     | JWT algorithm                | `HS256`                                                         |
| `JWT_EXPIRES_IN`    | JWT expiration (minutes)     | `30`                                                            |
| `MODEL_PATH`        | Path to ML models            | `/app/models`                                                   |
| `LOG_LEVEL`         | Logging level                | `INFO`                                                          |

## Database Management

### Using pgAdmin

1. Access pgAdmin at `http://localhost:8081`
2. Login with `admin@solarai.com` / `admin123`
3. Add server: `postgres:5432` / `solarai` / `solarai`

### Using psql

```bash
# Connect to database
docker exec -it solarai_postgres psql -U solarai -d solarai

# View tables
\dt

# View data
SELECT * FROM users;
SELECT * FROM analysis_results;
SELECT * FROM solar_panels;
```

## ML Model Integration

The backend uses a ResNet18 model for solar panel dirt detection:

```python
# Model loading
model_loader = ModelLoader(MODEL_PATH)
await model_loader.load_model()

# Prediction
prediction, confidence = await model_loader.predict_image(image_content)
```

## Development

### Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py      # SQLAlchemy models & Pydantic schemas
│   │   └── prediction.py    # ML model schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   └── prediction.py    # API routes
│   └── utils/
│       ├── __init__.py
│       └── model_loader.py  # ML model loading & inference
├── scripts/
│   └── init.sql            # Database initialization
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Adding New Endpoints

1. **Create route in main.py or separate router**

   ```python
   @app.post("/new-endpoint")
   async def new_endpoint(db_manager: DatabaseManager = Depends(get_db_manager)):
       # Your logic here
       pass
   ```

2. **Add database operations in DatabaseManager**

   ```python
   async def new_operation(self, data: dict):
       # Database operation
       pass
   ```

3. **Update Pydantic models if needed**
   ```python
   class NewModel(BaseModel):
       field: str
   ```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app
```

## Deployment

### Production Considerations

1. **Security**

   - Change default passwords
   - Use strong JWT secrets
   - Enable HTTPS
   - Configure CORS properly

2. **Performance**

   - Use connection pooling
   - Enable database indexing
   - Implement caching with Redis
   - Use CDN for static files

3. **Monitoring**
   - Add health checks
   - Implement logging
   - Set up metrics collection
   - Monitor database performance

### Docker Production

```bash
# Build production image
docker build -t solarai-backend:prod .

# Run with production settings
docker run -d \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e JWT_SECRET=your-secret \
  -p 8000:8000 \
  solarai-backend:prod
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**

   ```bash
   # Check if PostgreSQL is running
   docker ps | grep postgres

   # Check logs
   docker logs solarai_postgres
   ```

2. **Model Loading Failed**

   ```bash
   # Check if model file exists
   ls -la models/

   # Check model path in environment
   echo $MODEL_PATH
   ```

3. **Port Already in Use**

   ```bash
   # Find process using port
   lsof -i :8000

   # Kill process
   kill -9 <PID>
   ```

### Logs

```bash
# View API logs
docker logs solarai_api_gateway

# View database logs
docker logs solarai_postgres

# Follow logs
docker logs -f solarai_api_gateway
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
