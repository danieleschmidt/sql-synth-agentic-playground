# Architecture Documentation

## System Overview

The SQL Synthesis Agentic Playground is a Python-based application that translates natural language queries into SQL using LangChain and provides an interactive evaluation framework against standard benchmarks.

## Problem Statement

Developers and analysts need an intuitive way to generate SQL queries from natural language descriptions while ensuring security, accuracy, and compatibility across different database dialects.

## Success Criteria

- **Accuracy**: >80% accuracy on Spider benchmark, >70% on WikiSQL
- **Security**: Zero SQL injection vulnerabilities
- **Performance**: <2s response time for query generation
- **Coverage**: Support for 4+ SQL dialects
- **Usability**: Intuitive Streamlit interface

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  SQL Synthesis  │───▶│   Database      │
│                 │    │     Agent       │    │   Connectors    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   LangChain     │              │
         │              │   SQL Toolkit   │              │
         │              └─────────────────┘              │
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│   Benchmark     │                              │   Security      │
│   Evaluation    │                              │   Layer         │
│   (Spider/Wiki) │                              │   (Parameterized)│
└─────────────────┘                              └─────────────────┘
```

## Component Architecture

### 1. Presentation Layer
- **Streamlit UI** (`app.py`): Interactive web interface
- **Input validation**: Natural language query sanitization
- **Result display**: SQL query output, execution results, metrics

### 2. Application Layer
- **SQL Synthesis Agent** (`src/sql_synth/`): Core NL-to-SQL translation
- **Database Manager** (`src/sql_synth/database.py`): Connection management
- **Evaluation Framework**: Benchmark testing against Spider/WikiSQL

### 3. Data Layer
- **Database Connectors**: PostgreSQL, MySQL, SQLite, Snowflake
- **Benchmark Datasets**: Spider and WikiSQL cached in Docker volumes
- **Configuration Management**: Environment-based settings

## Data Flow

1. **User Input**: Natural language query via Streamlit
2. **Preprocessing**: Input validation and sanitization
3. **Agent Processing**: LangChain SQL Agent generates SQL
4. **Security Check**: Parameterized query validation
5. **Execution**: Query execution against target database
6. **Response**: Results displayed with metrics

## Security Architecture

### SQL Injection Prevention
- **Parameterized Queries**: All queries use bound parameters
- **Input Validation**: Strict input sanitization
- **Query Analysis**: Static analysis for injection patterns
- **Execution Sandboxing**: Limited database permissions

### Authentication & Authorization
- **Environment Variables**: Secure credential management
- **Connection Pooling**: Secure database connections
- **Audit Logging**: Query execution tracking

## Technology Stack

### Core Technologies
- **Python 3.9+**: Primary language
- **Streamlit**: Web interface framework
- **LangChain**: NL-to-SQL agent framework
- **SQLAlchemy**: Database abstraction layer

### Database Support
- **PostgreSQL**: Primary production database
- **MySQL**: Secondary database support
- **SQLite**: Development and testing
- **Snowflake**: Enterprise data warehouse

### Development Tools
- **pytest**: Testing framework
- **black/ruff**: Code formatting and linting
- **mypy**: Type checking
- **Docker**: Containerization

## Deployment Architecture

### Development Environment
```
┌─────────────────┐
│   Developer     │
│   Machine       │
│                 │
│ ┌─────────────┐ │
│ │  Streamlit  │ │
│ │     App     │ │
│ └─────────────┘ │
│ ┌─────────────┐ │
│ │   Local     │ │
│ │  Database   │ │
│ └─────────────┘ │
└─────────────────┘
```

### Production Environment
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load          │    │   Application   │    │   Database      │
│   Balancer      │───▶│   Container     │───▶│   Cluster       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Monitoring    │              │
         │              │   & Logging     │              │
         │              └─────────────────┘              │
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│   CDN           │                              │   Backup        │
│   (Static       │                              │   Storage       │
│   Assets)       │                              │                 │
└─────────────────┘                              └─────────────────┘
```

## Performance Considerations

### Query Generation
- **Caching**: LLM response caching for common patterns
- **Optimization**: Query plan analysis and optimization
- **Batching**: Batch processing for evaluation runs

### Database Performance
- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Index recommendations
- **Monitoring**: Performance metrics collection

## Scalability

### Horizontal Scaling
- **Stateless Design**: No server-side session state
- **Container Ready**: Docker-based deployment
- **Load Balancing**: Multiple application instances

### Vertical Scaling
- **Memory Management**: Efficient LLM model loading
- **CPU Optimization**: Parallel query processing
- **Storage**: Efficient benchmark data caching

## Monitoring & Observability

### Application Metrics
- **Query Generation Time**: Response time tracking
- **Accuracy Metrics**: Benchmark performance
- **Error Rates**: Failure rate monitoring
- **Usage Patterns**: User interaction analytics

### Infrastructure Metrics
- **Resource Utilization**: CPU, memory, disk
- **Database Performance**: Query execution times
- **Network Latency**: Response time monitoring
- **Availability**: Uptime tracking

## Disaster Recovery

### Backup Strategy
- **Database Backups**: Regular automated backups
- **Configuration Backup**: Environment settings
- **Code Repository**: Git-based version control

### Recovery Procedures
- **RTO**: Recovery Time Objective < 1 hour
- **RPO**: Recovery Point Objective < 15 minutes
- **Failover**: Automated failover procedures
- **Testing**: Regular disaster recovery testing

## Future Architecture Considerations

### Planned Enhancements
- **Multi-tenant Architecture**: Support for multiple organizations
- **Advanced Analytics**: Query pattern analysis
- **ML Pipeline**: Continuous model improvement
- **API Gateway**: RESTful API for integration

### Technology Evolution
- **Cloud Migration**: Move to cloud-native architecture
- **Microservices**: Break down into smaller services
- **Event-Driven**: Implement event-driven patterns
- **GraphQL**: Modern API layer