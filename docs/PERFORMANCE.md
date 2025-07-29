# Performance Guidelines

## Performance Requirements

Based on the [Architecture Documentation](ARCHITECTURE.md), our performance targets are:

- **Query Generation**: <2s response time for SQL generation
- **Database Connection**: <500ms connection establishment
- **UI Responsiveness**: <100ms for user interactions
- **Memory Usage**: <512MB RAM under normal load
- **Throughput**: 100+ concurrent users

## Performance Testing

### Load Testing

Run performance tests with:

```bash
# Install performance testing dependencies
pip install -e ".[dev]"

# Run load tests
pytest tests/load/ -v

# Run benchmark tests
pytest tests/performance/ -v --benchmark-only
```

### Benchmarking

Our benchmark suite evaluates:

- **Spider Dataset**: Target >80% accuracy
- **WikiSQL Dataset**: Target >70% accuracy
- **Query Generation Speed**: Target <2s per query
- **Memory Efficiency**: Monitor memory leaks

### Monitoring

Key metrics to monitor in production:

- Response time percentiles (P50, P95, P99)
- Error rates by endpoint
- Database connection pool utilization
- Memory and CPU usage patterns
- Cache hit rates

## Optimization Strategies

### Database Performance

1. **Connection Pooling**
   - Use SQLAlchemy connection pooling
   - Configure appropriate pool sizes
   - Monitor connection leaks

2. **Query Optimization**
   - Use parameterized queries
   - Implement query result caching
   - Monitor slow queries

3. **Index Management**
   - Create indexes for frequent queries
   - Monitor index usage statistics
   - Regular index maintenance

### Application Performance

1. **Caching Strategy**
   - Cache LLM responses for common patterns
   - Implement Redis for distributed caching
   - Cache database schema information

2. **Memory Management**
   - Monitor memory usage patterns
   - Implement garbage collection tuning
   - Use memory profiling tools

3. **Async Processing**
   - Use async/await for I/O operations
   - Implement background job processing
   - Queue long-running operations

### UI Performance

1. **Streamlit Optimization**
   - Use st.cache_data for expensive operations
   - Implement proper state management
   - Minimize widget re-rendering

2. **Resource Loading**
   - Lazy load large datasets
   - Implement pagination for results
   - Optimize image and asset loading

## Performance Monitoring

### Application Metrics

Implement monitoring for:

```python
# Example metrics collection
import time
import psutil
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        # Log metrics
        logger.info(f"{func.__name__} - Time: {end_time - start_time:.2f}s, "
                   f"Memory Delta: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
        
        return result
    return wrapper
```

### Infrastructure Monitoring

Use Prometheus metrics for:

- HTTP request duration
- Database query performance
- System resource utilization
- Application health checks

### Alerting

Configure alerts for:

- Response time > 5s (P95)
- Error rate > 5%
- Memory usage > 80%
- Database connection failures

## Performance Testing in CI/CD

### Automated Performance Tests

```yaml
# Example GitHub Actions performance test
- name: Run Performance Tests
  run: |
    pytest tests/performance/ --benchmark-json=benchmark.json
    
- name: Performance Regression Check
  run: |
    python scripts/check_performance_regression.py benchmark.json
```

### Performance Budgets

Fail builds if performance degrades:

- Query generation time increases by >20%
- Memory usage increases by >15%
- Benchmark accuracy drops by >5%

## Profiling Tools

### Python Profiling

```bash
# CPU profiling
python -m cProfile -o profile.stats app.py

# Memory profiling
python -m memory_profiler app.py

# Line-by-line profiling
kernprof -l -v app.py
```

### Database Profiling

```sql
-- PostgreSQL query analysis
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;

-- Query performance monitoring
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC;
```

## Performance Best Practices

1. **Database Access**
   - Use connection pooling
   - Implement query result caching
   - Monitor and optimize slow queries

2. **Memory Management**
   - Avoid memory leaks in long-running processes
   - Use generators for large datasets
   - Implement proper cleanup

3. **Network Optimization**
   - Minimize database round trips
   - Use compression for large responses
   - Implement proper timeout handling

4. **Caching Strategy**
   - Cache at multiple levels (application, database, CDN)
   - Implement cache invalidation strategies
   - Monitor cache hit rates

## Troubleshooting Performance Issues

### Common Issues

1. **Slow Query Generation**
   - Check LLM model response times
   - Verify network connectivity
   - Review query complexity

2. **High Memory Usage**
   - Check for memory leaks
   - Review data structures
   - Monitor garbage collection

3. **Database Connection Issues**
   - Verify connection pool configuration
   - Check database server performance
   - Review connection timeout settings

### Debugging Tools

- Use application performance monitoring (APM)
- Implement distributed tracing
- Set up comprehensive logging
- Use profiling tools regularly

## References

- [SQLAlchemy Performance](https://docs.sqlalchemy.org/en/20/core/engines.html#pooling)
- [Streamlit Performance](https://docs.streamlit.io/library/advanced-features/caching)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)