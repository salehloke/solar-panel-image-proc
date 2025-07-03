# SolarAI Architecture Decisions

## Project Overview

Solar panel dirt detection system with micro-frontend and microservices architecture.

## Architecture Decisions

### 1. Frontend: Nx Monorepo

**Decision:** Nx Monorepo
**Rationale:** Excellent micro-frontend support, built-in tooling, smart caching

**Structure:**

```
frontend/ (Nx monorepo)
├── apps/
│   ├── dashboard/           # Main solar panel dashboard
│   ├── admin/               # Admin panel
│   └── mobile/              # Mobile app (future)
├── libs/
│   ├── shared-ui/           # Common components
│   ├── shared-api/          # API client
│   ├── shared-auth/         # Auth utilities
│   └── shared-types/        # TypeScript interfaces
```

### 2. Backend: Mixed Language Microservices

**Decision:** Mixed Language Microservices
**Rationale:** Python for ML, NestJS for business logic, technology flexibility

**Services:**

```
backend/
├── services/
│   ├── api-gateway/         # FastAPI - Main gateway
│   ├── ml-service/          # Python/FastAPI - PyTorch models
│   ├── auth-service/        # NestJS - Authentication
│   ├── storage-service/     # Python/FastAPI - File storage
│   └── notification-service/ # NestJS - Notifications
├── databases/
│   ├── postgres/            # Main application data
│   ├── redis/               # Caching & sessions
│   └── minio/               # Object storage for images
└── docker-compose.yml       # Service orchestration
```

### 3. Authentication: JWT Tokens

**Decision:** JWT Tokens
**Rationale:** Stateless, scalable, microservice-friendly

**Pros:**

- ✅ Stateless - no server-side storage
- ✅ Scalable - works across multiple servers
- ✅ Microservice-friendly - any service can validate
- ✅ Mobile-friendly - easy implementation

**Cons:**

- ❌ Can't revoke tokens until expiration
- ❌ Token size larger than session IDs
- ❌ Security concerns with client storage

### 4. Database Strategy: Separate Databases

**Decision:** Separate databases per service
**Rationale:** Service independence, technology flexibility

**Databases:**

- **PostgreSQL:** Main application data
- **Redis:** Caching, sessions, rate limiting
- **MinIO:** Object storage for images

### 5. File Storage: MinIO

**Decision:** MinIO
**Rationale:** S3-compatible, self-hosted, cost-effective

**Configuration:**

- File size limits: 10-50MB per image
- Supported formats: JPEG, PNG, WebP
- Image compression and optimization
- Virus scanning (optional)

### 6. Monitoring: Prometheus + Grafana

**Decision:** Prometheus + Grafana
**Rationale:** Industry standard, free, rich ecosystem

**Pros:**

- ✅ Industry standard
- ✅ Free and open source
- ✅ Rich ecosystem
- ✅ Customizable dashboards
- ✅ Built-in alerting

**Cons:**

- ❌ Learning curve
- ❌ Resource usage
- ❌ Configuration complexity

### 7. Reverse Proxy: Traefik

**Decision:** Traefik
**Rationale:** Docker-native, automatic service discovery

**Use Cases:**

- SSL/TLS termination
- Load balancing
- Security (rate limiting, DDoS protection)
- Caching
- API routing

## Implementation Phases

### Phase 1: Frontend Nx Setup

- [ ] Set up Nx monorepo
- [ ] Migrate current Next.js app
- [ ] Create shared libraries
- [ ] Set up Module Federation

### Phase 2: Backend Services

- [ ] API Gateway (FastAPI)
- [ ] ML Service (current FastAPI app)
- [ ] Auth Service (NestJS)
- [ ] Storage Service (Python)
- [ ] Notification Service (NestJS)

### Phase 3: Infrastructure

- [ ] Docker Compose setup
- [ ] Database migrations
- [ ] Service communication
- [ ] Authentication flow
- [ ] Monitoring setup

### Phase 4: Integration

- [ ] Frontend-backend integration
- [ ] Service-to-service communication
- [ ] Error handling
- [ ] Production deployment

## Technical Specifications

### Authentication Flow

1. User logs in → Auth Service validates credentials
2. Auth Service issues JWT access token + refresh token
3. Frontend stores tokens securely
4. API requests include JWT token
5. API Gateway validates token
6. Token refresh when expired

### Service Communication

- **Synchronous:** HTTP/REST for direct requests
- **Asynchronous:** Message queues for notifications
- **Service Discovery:** Docker Compose networking
- **Load Balancing:** Traefik reverse proxy

### Security Considerations

- JWT tokens with short expiration
- HTTPS everywhere (Traefik SSL termination)
- Rate limiting on API Gateway
- Input validation on all services
- File upload validation and scanning
- Database connection encryption

### Performance Considerations

- Redis caching for frequently accessed data
- Image compression and optimization
- Database indexing strategy
- CDN for static assets (production)
- Horizontal scaling capability

## Questions and Answers

### Q: JWT vs Session-Based Authentication?

**A:** JWT chosen for stateless, scalable microservices architecture.

### Q: File storage requirements?

**A:** MinIO with 10-50MB per image, compression, validation.

### Q: Monitoring solution?

**A:** Prometheus + Grafana for self-hosted monitoring.

### Q: Reverse proxy use cases?

**A:** SSL termination, load balancing, security, caching, routing.

## Next Steps

1. Confirm architecture decisions
2. Start with Nx frontend monorepo setup
3. Plan service boundaries and APIs
4. Set up development environment
5. Begin Phase 1 implementation
