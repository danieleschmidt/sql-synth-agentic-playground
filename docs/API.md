# API Documentation

## Overview

While the SQL Synthesis Agentic Playground primarily uses Streamlit for its user interface, it also provides REST API endpoints for programmatic access and integration with other systems.

## Base URL

- **Development**: `http://localhost:8501/api`
- **Production**: `https://sqlsynth.example.com/api`

## Authentication

Currently, the API uses basic authentication. Future versions will support API keys and OAuth2.

```bash
# Example authentication
curl -u username:password https://sqlsynth.example.com/api/health
```

## Endpoints

### Health Check

Check the application health status.

**Endpoint**: `GET /api/health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-29T12:00:00Z",
  "version": "0.1.0",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 0.1
    },
    "memory": {
      "status": "healthy",
      "usage_percent": 45.2,
      "available_mb": 2048
    },
    "disk": {
      "status": "healthy",
      "usage_percent": 32.1,
      "free_gb": 15.4
    }
  }
}
```

**Status Codes**:
- `200 OK`: System is healthy
- `503 Service Unavailable`: System is unhealthy

### SQL Generation

Generate SQL queries from natural language input.

**Endpoint**: `POST /api/generate`

**Request Body**:
```json
{
  "query": "Show me all users who registered last month",
  "database_type": "postgresql",
  "schema_context": {
    "tables": ["users", "registrations"],
    "description": "User management system"
  },
  "options": {
    "include_explanation": true,
    "validate_syntax": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "sql_query": "SELECT * FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' AND created_at < DATE_TRUNC('month', CURRENT_DATE);",
  "explanation": "This query selects all users who were created in the previous month by filtering the created_at timestamp.",
  "confidence": 0.92,
  "execution_time": 1.23,
  "metadata": {
    "tables_used": ["users"],
    "query_type": "SELECT",
    "complexity": "medium"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_QUERY",
    "message": "Could not parse natural language query",
    "details": "The query is too ambiguous. Please provide more specific information."
  }
}
```

**Status Codes**:
- `200 OK`: Query generated successfully
- `400 Bad Request`: Invalid request parameters
- `422 Unprocessable Entity`: Could not generate query
- `500 Internal Server Error`: Server error

### Query Validation

Validate SQL query syntax and security.

**Endpoint**: `POST /api/validate`

**Request Body**:
```json
{
  "sql_query": "SELECT * FROM users WHERE id = ?",
  "database_type": "postgresql",
  "parameters": [123]
}
```

**Response**:
```json
{
  "valid": true,
  "syntax_check": "passed",
  "security_check": "passed",
  "warnings": [],
  "suggestions": [
    "Consider limiting results with LIMIT clause for better performance"
  ]
}
```

### Benchmark Evaluation

Evaluate query accuracy against standard benchmarks.

**Endpoint**: `POST /api/evaluate`

**Request Body**:
```json
{
  "queries": [
    {
      "natural_language": "Show total sales by region",
      "generated_sql": "SELECT region, SUM(amount) FROM sales GROUP BY region",
      "expected_sql": "SELECT region, SUM(sales_amount) FROM sales_data GROUP BY region"
    }
  ],
  "benchmark": "spider"
}
```

**Response**:
```json
{
  "results": {
    "overall_accuracy": 0.85,
    "execution_accuracy": 0.92,
    "exact_match": 0.78,
    "details": [
      {
        "query_id": 0,
        "accuracy": 0.85,
        "match": false,
        "issues": ["Column name mismatch: 'amount' vs 'sales_amount'"]
      }
    ]
  }
}
```

### Database Schema

Retrieve database schema information.

**Endpoint**: `GET /api/schema/{database_type}`

**Parameters**:
- `database_type`: Type of database (postgresql, mysql, sqlite, snowflake)

**Query Parameters**:
- `include_samples`: Include sample data (default: false)
- `table_filter`: Filter tables by pattern

**Response**:
```json
{
  "database_type": "postgresql",
  "schema": {
    "tables": [
      {
        "name": "users",
        "columns": [
          {
            "name": "id",
            "type": "INTEGER",
            "nullable": false,
            "primary_key": true
          },
          {
            "name": "email",
            "type": "VARCHAR(255)",
            "nullable": false,
            "unique": true
          },
          {
            "name": "created_at",
            "type": "TIMESTAMP",
            "nullable": false,
            "default": "CURRENT_TIMESTAMP"
          }
        ],
        "sample_data": [
          {"id": 1, "email": "user@example.com", "created_at": "2025-01-01T10:00:00Z"}
        ]
      }
    ]
  }
}
```

## Webhook Support

### Query Generation Webhook

Receive notifications when queries are generated.

**Configuration**:
```json
{
  "webhook_url": "https://your-app.com/webhooks/sql-generation",
  "events": ["query.generated", "query.failed"],
  "secret": "your-webhook-secret"
}
```

**Payload**:
```json
{
  "event": "query.generated",
  "timestamp": "2025-01-29T12:00:00Z",
  "data": {
    "query_id": "uuid-123",
    "natural_language": "Show recent orders",
    "generated_sql": "SELECT * FROM orders WHERE created_at > NOW() - INTERVAL '7 days'",
    "confidence": 0.88,
    "user_id": "user-456"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Anonymous users**: 100 requests per hour
- **Authenticated users**: 1000 requests per hour
- **Premium users**: 10000 requests per hour

**Rate Limit Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

## Error Handling

### Standard Error Format

All API errors follow a consistent format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details",
    "timestamp": "2025-01-29T12:00:00Z",
    "request_id": "req-uuid-123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INVALID_QUERY` | 422 | Cannot process query |
| `SERVER_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## SDK Examples

### Python SDK

```python
import requests
from typing import Dict, Any, Optional

class SQLSynthesisClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password)
        self.session = requests.Session()
    
    def generate_sql(self, 
                    natural_query: str, 
                    database_type: str = "postgresql",
                    schema_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate SQL from natural language query."""
        payload = {
            "query": natural_query,
            "database_type": database_type,
            "schema_context": schema_context or {},
            "options": {
                "include_explanation": True,
                "validate_syntax": True
            }
        }
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def validate_sql(self, sql_query: str, database_type: str = "postgresql") -> Dict[str, Any]:
        """Validate SQL query syntax and security."""
        payload = {
            "sql_query": sql_query,
            "database_type": database_type
        }
        
        response = self.session.post(
            f"{self.base_url}/api/validate",
            json=payload,
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(
            f"{self.base_url}/api/health",
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = SQLSynthesisClient(
    base_url="https://sqlsynth.example.com",
    username="your-username",
    password="your-password"
)

# Generate SQL
result = client.generate_sql(
    "Show me all active users from last week",
    database_type="postgresql"
)
print(f"Generated SQL: {result['sql_query']}")
print(f"Confidence: {result['confidence']}")

# Validate SQL
validation = client.validate_sql(
    "SELECT * FROM users WHERE active = true"
)
print(f"Valid: {validation['valid']}")
```

### JavaScript SDK

```javascript
class SQLSynthesisClient {
    constructor(baseUrl, username, password) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.auth = btoa(`${username}:${password}`);
    }
    
    async generateSQL(naturalQuery, databaseType = 'postgresql', schemaContext = {}) {
        const response = await fetch(`${this.baseUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Basic ${this.auth}`
            },
            body: JSON.stringify({
                query: naturalQuery,
                database_type: databaseType,
                schema_context: schemaContext,
                options: {
                    include_explanation: true,
                    validate_syntax: true
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async validateSQL(sqlQuery, databaseType = 'postgresql') {
        const response = await fetch(`${this.baseUrl}/api/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Basic ${this.auth}`
            },
            body: JSON.stringify({
                sql_query: sqlQuery,
                database_type: databaseType
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/api/health`, {
            headers: {
                'Authorization': `Basic ${this.auth}`
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
}

// Usage
const client = new SQLSynthesisClient(
    'https://sqlsynth.example.com',
    'your-username',
    'your-password'
);

// Generate SQL
try {
    const result = await client.generateSQL(
        'Show me all orders from the last month'
    );
    console.log('Generated SQL:', result.sql_query);
} catch (error) {
    console.error('Error:', error.message);
}
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
- **JSON**: `/api/openapi.json`
- **YAML**: `/api/openapi.yaml`
- **Interactive Docs**: `/api/docs`

## Testing the API

### Using curl

```bash
# Health check
curl -X GET "https://sqlsynth.example.com/api/health" \
     -u "username:password"

# Generate SQL
curl -X POST "https://sqlsynth.example.com/api/generate" \
     -u "username:password" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Show all users who registered today",
       "database_type": "postgresql"
     }'

# Validate SQL
curl -X POST "https://sqlsynth.example.com/api/validate" \
     -u "username:password" \
     -H "Content-Type: application/json" \
     -d '{
       "sql_query": "SELECT * FROM users WHERE created_at::date = CURRENT_DATE",
       "database_type": "postgresql"
     }'
```

### Using Postman

Import the OpenAPI specification into Postman for easy testing:

1. Open Postman
2. Click "Import" 
3. Enter URL: `https://sqlsynth.example.com/api/openapi.json`
4. Configure authentication in the collection settings

## Versioning

The API uses semantic versioning with URL path versioning:

- **Current**: `/api/v1/`
- **Future**: `/api/v2/`

Version compatibility is maintained for at least one major version.

## Rate Limiting and Quotas

Different tiers have different limits:

| Tier | Requests/Hour | Requests/Day | Query Complexity |
|------|---------------|--------------|------------------|
| Free | 100 | 1,000 | Basic |
| Pro | 1,000 | 10,000 | Advanced |
| Enterprise | 10,000 | 100,000 | Unlimited |

## Support

For API support and questions:

- **Documentation**: [API Docs](https://sqlsynth.example.com/api/docs)
- **GitHub Issues**: [Report bugs](https://github.com/danieleschmidt/sql-synth-agentic-playground/issues)
- **Email**: support@sqlsynth.example.com

## References

- [REST API Best Practices](https://restfulapi.net/)
- [OpenAPI Specification](https://swagger.io/specification/)
- [HTTP Status Codes](https://httpstatuses.com/)