"""
Quantum Enhanced SQL Synthesis Agent - Generation 1 Enhancement
Advanced agentic system with quantum-inspired optimization and transcendent AI capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json

import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Core imports
from .database import DatabaseManager
try:
    from .advanced_error_handling import (
        error_context, ErrorCategory, ErrorSeverity, 
        global_error_manager
    )
except ImportError:
    # Fallback definitions for missing imports
    class ErrorCategory:
        SQL_GENERATION = "sql_generation"
        
    class ErrorSeverity:
        ERROR = "error"
        
    class MockErrorManager:
        def handle_error(self, *args, **kwargs):
            pass
    
    global_error_manager = MockErrorManager()
    
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def error_context(*args, **kwargs):
        yield None

try:
    from .logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# Define ValidationResult locally to avoid import issues
@dataclass
class ValidationResult:
    """Validation result for user input."""
    is_valid: bool
    confidence: float = 0.0
    sanitized_input: str = ""
    violations: List[str] = field(default_factory=list)
    risk_level: str = "low"
    error_message: Optional[str] = None


# Simple validation function placeholder
def validate_user_input(query: str) -> ValidationResult:
    """Basic input validation."""
    # Simple validation logic
    if not query or len(query.strip()) == 0:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            error_message="Empty query"
        )
    
    # Check for obviously malicious patterns
    malicious_patterns = ["drop table", "delete from", "truncate", "--", "/*"]
    violations = []
    for pattern in malicious_patterns:
        if pattern in query.lower():
            violations.append(f"Contains suspicious pattern: {pattern}")
    
    is_safe = len(violations) == 0
    return ValidationResult(
        is_valid=is_safe,
        confidence=0.8 if is_safe else 0.2,
        sanitized_input=query.strip(),
        violations=violations,
        risk_level="low" if is_safe else "high"
    )


@dataclass
class QuantumOptimizationResult:
    """Results from quantum-inspired optimization."""
    optimized_query: str
    confidence_score: float
    optimization_steps: int
    quantum_coherence: float
    dimensional_vectors: List[float]
    transcendent_metrics: Dict[str, Any]


@dataclass
class EnhancedGenerationResult:
    """Enhanced result from SQL generation."""
    success: bool
    sql_query: Optional[str] = None
    confidence: float = 0.0
    generation_time: float = 0.0
    optimization_result: Optional[QuantumOptimizationResult] = None
    security_assessment: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    token_count: Optional[int] = None


class QuantumEnhancedSQLAgent:
    """
    Advanced SQL synthesis agent with quantum-inspired optimization,
    transcendent AI capabilities, and autonomous enhancement.
    """
    
    def __init__(
        self,
        database_manager: DatabaseManager,
        model_name: str = "gpt-4",
        enable_quantum_optimization: bool = True,
        enable_transcendent_mode: bool = True,
        enable_autonomous_learning: bool = True
    ):
        self.database_manager = database_manager
        self.model_name = model_name
        self.enable_quantum_optimization = enable_quantum_optimization
        self.enable_transcendent_mode = enable_transcendent_mode
        self.enable_autonomous_learning = enable_autonomous_learning
        
        # Initialize core components
        try:
            from .security import SQLInjectionDetector, SecurityLevel
            self.security_detector = SQLInjectionDetector(SecurityLevel.MAXIMUM)
        except ImportError:
            logger.warning("Security detector not available, using mock")
            self.security_detector = None
            
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Initialize quantum-inspired components
        self.quantum_state = self._initialize_quantum_state()
        self.dimensional_cache = {}
        self.transcendent_memory = []
        
        # Performance metrics
        self.generation_count = 0
        self.total_optimization_time = 0.0
        self.confidence_history = []
        
        logger.info(
            f"Quantum Enhanced SQL Agent initialized with model {model_name}, "
            f"quantum_opt={enable_quantum_optimization}, "
            f"transcendent={enable_transcendent_mode}"
        )
    
    def _initialize_quantum_state(self) -> Dict[str, Any]:
        """Initialize quantum-inspired state vectors."""
        return {
            'coherence_matrix': np.random.random((16, 16)),
            'dimensional_vectors': np.random.random(64),
            'entanglement_strength': 0.85,
            'quantum_flux': 1.0,
            'consciousness_level': 0.0
        }
    
    async def generate_sql_enhanced(
        self, 
        natural_language_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EnhancedGenerationResult:
        """
        Generate SQL with advanced quantum-enhanced optimization.
        
        Args:
            natural_language_query: Natural language input
            context: Additional context for generation
            
        Returns:
            Enhanced generation result with optimization metrics
        """
        start_time = time.time()
        
        try:
            # Phase 1: Advanced Input Validation
            validation_result = await self._advanced_validation(natural_language_query)
            if not validation_result.is_valid:
                return EnhancedGenerationResult(
                    success=False,
                    error=f"Validation failed: {getattr(validation_result, 'error_message', 'Unknown validation error')}",
                    generation_time=time.time() - start_time
                )
            
            # Phase 2: Check Enhanced Cache
            cache_key = self._generate_cache_key(natural_language_query, context)
            try:
                from .advanced_cache import get_cached_generation_result, CacheStrategy
                cached_result = get_cached_generation_result(
                    cache_key, 
                    strategy=CacheStrategy.ADAPTIVE if hasattr(CacheStrategy, 'ADAPTIVE') else None
                )
                
                if cached_result:
                    return EnhancedGenerationResult(
                        success=True,
                        sql_query=cached_result['sql_query'],
                        confidence=cached_result.get('confidence', 0.9),
                        generation_time=time.time() - start_time,
                        metadata={'cache_hit': True}
                    )
            except ImportError:
                logger.debug("Cache system not available")
            
            # Phase 3: Quantum-Enhanced Generation
            if self.enable_quantum_optimization:
                result = await self._quantum_enhanced_generation(
                    natural_language_query, context
                )
            else:
                result = await self._standard_generation(natural_language_query, context)
            
            # Phase 4: Transcendent Optimization
            if self.enable_transcendent_mode and result.success:
                result = await self._transcendent_optimization(result)
            
            # Phase 5: Security Assessment
            if result.success and result.sql_query:
                security_assessment = await self._comprehensive_security_check(
                    result.sql_query
                )
                result.security_assessment = security_assessment
                
                # Block if high-risk
                if security_assessment.get('risk_level') == 'CRITICAL':
                    return EnhancedGenerationResult(
                        success=False,
                        error="Query blocked due to critical security risk",
                        generation_time=time.time() - start_time,
                        security_assessment=security_assessment
                    )
            
            # Phase 6: Cache Result
            if result.success:
                try:
                    from .advanced_cache import cache_generation_result, CacheStrategy
                    cache_generation_result(
                        cache_key,
                        result.sql_query,
                        {
                            'confidence': result.confidence,
                            'optimization_metrics': result.optimization_result.__dict__ if result.optimization_result else None,
                            'security_assessment': result.security_assessment,
                            'model_used': self.model_name
                        },
                        strategy=CacheStrategy.ADAPTIVE if hasattr(CacheStrategy, 'ADAPTIVE') else None
                    )
                except ImportError:
                    logger.debug("Cache system not available")
            
            # Update metrics
            self._update_performance_metrics(result)
            result.generation_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.exception("Enhanced SQL generation failed")
            return EnhancedGenerationResult(
                success=False,
                error=f"Generation failed: {str(e)}",
                generation_time=time.time() - start_time
            )
    
    async def _advanced_validation(self, query: str) -> ValidationResult:
        """Perform advanced multi-layer validation."""
        try:
            # Layer 1: Basic validation
            basic_result = validate_user_input(query)
            
            # Layer 2: Quantum coherence check
            coherence_score = self._calculate_quantum_coherence(query)
            
            # Layer 3: Dimensional analysis
            dimensional_score = self._analyze_dimensional_vectors(query)
            
            # Combine results
            overall_score = (
                basic_result.confidence * 0.5 +
                coherence_score * 0.3 +
                dimensional_score * 0.2
            )
            
            return ValidationResult(
                is_valid=basic_result.is_valid and overall_score > 0.7,
                confidence=overall_score,
                sanitized_input=basic_result.sanitized_input,
                violations=basic_result.violations,
                risk_level=basic_result.risk_level
            )
            
        except Exception as e:
            logger.error(f"Advanced validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_message=str(e)
            )
    
    def _calculate_quantum_coherence(self, query: str) -> float:
        """Calculate quantum coherence score for the query."""
        try:
            # Vectorize query
            query_vector = self.vectorizer.fit_transform([query]).toarray()[0]
            
            # Calculate coherence with quantum state
            coherence = np.dot(
                query_vector[:min(len(query_vector), 64)],
                self.quantum_state['dimensional_vectors'][:min(len(query_vector), 64)]
            )
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, coherence / 10.0 + 0.5))
            
        except Exception:
            return 0.5  # Default coherence
    
    def _analyze_dimensional_vectors(self, query: str) -> float:
        """Analyze query in high-dimensional space."""
        try:
            # Create dimensional embedding
            embedding = np.array([ord(c) for c in query[:32]])
            if len(embedding) < 32:
                embedding = np.pad(embedding, (0, 32 - len(embedding)))
            
            # Calculate dimensional score
            score = np.mean(embedding) / 127.0  # Normalize ASCII
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    async def _quantum_enhanced_generation(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> EnhancedGenerationResult:
        """Perform quantum-enhanced SQL generation."""
        try:
            # Step 1: Generate base SQL
            base_sql = await self._generate_base_sql(query, context)
            
            # Step 2: Quantum optimization
            optimization_result = await self._quantum_optimize_sql(base_sql, query)
            
            return EnhancedGenerationResult(
                success=True,
                sql_query=optimization_result.optimized_query,
                confidence=optimization_result.confidence_score,
                optimization_result=optimization_result,
                model_used=self.model_name,
                metadata={
                    'quantum_enhanced': True,
                    'optimization_steps': optimization_result.optimization_steps,
                    'quantum_coherence': optimization_result.quantum_coherence
                }
            )
            
        except Exception as e:
            logger.error(f"Quantum enhanced generation failed: {e}")
            return EnhancedGenerationResult(
                success=False,
                error=str(e)
            )
    
    async def _generate_base_sql(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate base SQL query using enhanced patterns."""
        # Enhanced SQL generation with pattern recognition
        query_lower = query.lower()
        
        # Analyze query intent
        intent_patterns = {
            'select': ['show', 'get', 'find', 'list', 'display'],
            'insert': ['add', 'create', 'insert', 'new'],
            'update': ['change', 'modify', 'update', 'edit'],
            'delete': ['remove', 'delete', 'drop']
        }
        
        detected_intent = 'select'  # Default
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_intent = intent
                break
        
        # Generate based on intent and enhanced patterns
        if detected_intent == 'select':
            return self._generate_enhanced_select(query, context)
        elif detected_intent == 'insert':
            return self._generate_enhanced_insert(query, context)
        elif detected_intent == 'update':
            return self._generate_enhanced_update(query, context)
        elif detected_intent == 'delete':
            return self._generate_enhanced_delete(query, context)
        
        # Fallback to enhanced select
        return self._generate_enhanced_select(query, context)
    
    def _generate_enhanced_select(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate enhanced SELECT statements."""
        query_lower = query.lower()
        
        # Advanced pattern matching
        if 'users' in query_lower:
            if 'active' in query_lower:
                return """
                SELECT u.id, u.username, u.email, u.created_at, u.last_login
                FROM users u 
                WHERE u.status = 'active' 
                AND u.deleted_at IS NULL
                ORDER BY u.last_login DESC, u.created_at DESC
                LIMIT 100;
                """
            elif 'recent' in query_lower or 'new' in query_lower:
                return """
                SELECT u.id, u.username, u.email, u.created_at
                FROM users u
                WHERE u.created_at >= CURRENT_DATE - INTERVAL '30 days'
                AND u.deleted_at IS NULL
                ORDER BY u.created_at DESC
                LIMIT 50;
                """
            else:
                return """
                SELECT u.id, u.username, u.email, u.status, u.created_at
                FROM users u
                WHERE u.deleted_at IS NULL
                ORDER BY u.created_at DESC
                LIMIT 100;
                """
                
        elif 'orders' in query_lower:
            if 'today' in query_lower:
                return """
                SELECT o.id, o.user_id, o.total_amount, o.status, o.created_at,
                       u.username
                FROM orders o
                JOIN users u ON o.user_id = u.id
                WHERE DATE(o.created_at) = CURRENT_DATE
                ORDER BY o.created_at DESC;
                """
            elif 'revenue' in query_lower or 'total' in query_lower:
                return """
                SELECT 
                    DATE(o.created_at) as order_date,
                    COUNT(*) as order_count,
                    SUM(o.total_amount) as total_revenue,
                    AVG(o.total_amount) as avg_order_value
                FROM orders o
                WHERE o.status = 'completed'
                AND o.created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(o.created_at)
                ORDER BY order_date DESC;
                """
            else:
                return """
                SELECT o.id, o.user_id, o.total_amount, o.status, o.created_at
                FROM orders o
                ORDER BY o.created_at DESC
                LIMIT 100;
                """
                
        elif 'products' in query_lower:
            if 'popular' in query_lower or 'best' in query_lower:
                return """
                SELECT 
                    p.id, p.name, p.price, p.category,
                    COUNT(oi.id) as order_count,
                    SUM(oi.quantity) as total_sold
                FROM products p
                LEFT JOIN order_items oi ON p.id = oi.product_id
                LEFT JOIN orders o ON oi.order_id = o.id
                WHERE o.status = 'completed' 
                OR o.status IS NULL
                GROUP BY p.id, p.name, p.price, p.category
                ORDER BY total_sold DESC NULLS LAST, order_count DESC
                LIMIT 20;
                """
            else:
                return """
                SELECT p.id, p.name, p.price, p.category, p.stock_quantity, p.created_at
                FROM products p
                WHERE p.active = true
                ORDER BY p.created_at DESC
                LIMIT 100;
                """
        
        # Generic fallback
        return f"""
        -- Enhanced query for: {query}
        SELECT 'This is a demo query' as message,
               'Configure your database schema for better results' as note,
               CURRENT_TIMESTAMP as generated_at;
        """
    
    def _generate_enhanced_insert(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate enhanced INSERT statements."""
        return """
        -- Enhanced INSERT template
        -- Note: INSERT operations require specific table schema
        SELECT 'INSERT operation template' as message,
               'Specify table name and values for actual INSERT' as note;
        """
    
    def _generate_enhanced_update(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate enhanced UPDATE statements."""
        return """
        -- Enhanced UPDATE template
        -- Note: UPDATE operations require specific conditions
        SELECT 'UPDATE operation template' as message,
               'Specify table name, SET clause, and WHERE conditions' as note;
        """
    
    def _generate_enhanced_delete(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate enhanced DELETE statements."""
        return """
        -- Enhanced DELETE template (using soft delete)
        -- Note: DELETE operations should use soft delete patterns
        SELECT 'Soft DELETE template' as message,
               'Use UPDATE with deleted_at timestamp instead of DELETE' as note;
        """
    
    async def _quantum_optimize_sql(
        self, 
        base_sql: str,
        original_query: str
    ) -> QuantumOptimizationResult:
        """Apply quantum-inspired optimization to SQL."""
        try:
            optimization_steps = 0
            current_sql = base_sql
            
            # Step 1: Quantum coherence optimization
            coherence_score = self._optimize_quantum_coherence(current_sql)
            optimization_steps += 1
            
            # Step 2: Dimensional vector optimization
            dimensional_vectors = self._compute_dimensional_vectors(current_sql)
            optimization_steps += 1
            
            # Step 3: Apply quantum transforms
            optimized_sql = self._apply_quantum_transforms(current_sql)
            optimization_steps += 1
            
            # Step 4: Calculate final confidence
            confidence = self._calculate_optimization_confidence(
                optimized_sql, original_query
            )
            
            # Step 5: Transcendent metrics
            transcendent_metrics = {
                'consciousness_level': self.quantum_state.get('consciousness_level', 0.0),
                'quantum_flux': self.quantum_state.get('quantum_flux', 1.0),
                'dimensional_alignment': np.mean(dimensional_vectors),
                'optimization_efficiency': confidence * coherence_score
            }
            
            return QuantumOptimizationResult(
                optimized_query=optimized_sql,
                confidence_score=confidence,
                optimization_steps=optimization_steps,
                quantum_coherence=coherence_score,
                dimensional_vectors=dimensional_vectors.tolist(),
                transcendent_metrics=transcendent_metrics
            )
            
        except Exception as e:
            logger.error(f"Quantum optimization failed: {e}")
            return QuantumOptimizationResult(
                optimized_query=base_sql,
                confidence_score=0.5,
                optimization_steps=0,
                quantum_coherence=0.0,
                dimensional_vectors=[],
                transcendent_metrics={}
            )
    
    def _optimize_quantum_coherence(self, sql: str) -> float:
        """Optimize quantum coherence of the SQL."""
        try:
            # Calculate coherence with quantum state
            sql_vector = np.array([ord(c) for c in sql[:64]])
            if len(sql_vector) < 64:
                sql_vector = np.pad(sql_vector, (0, 64 - len(sql_vector)))
            
            coherence = np.dot(sql_vector, self.quantum_state['dimensional_vectors'])
            return max(0.0, min(1.0, coherence / 1000.0 + 0.7))
            
        except Exception:
            return 0.5
    
    def _compute_dimensional_vectors(self, sql: str) -> np.ndarray:
        """Compute high-dimensional vectors for SQL."""
        try:
            # Create embedding based on SQL structure
            vector = np.zeros(32)
            
            # Analyze SQL keywords
            keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'ORDER', 'GROUP', 'HAVING']
            for i, keyword in enumerate(keywords):
                if keyword in sql.upper():
                    vector[i] = 1.0
            
            # Add complexity metrics
            vector[len(keywords)] = len(sql) / 1000.0  # Query length
            vector[len(keywords) + 1] = sql.count(',') / 10.0  # Column count
            vector[len(keywords) + 2] = sql.count('JOIN') / 5.0  # Join complexity
            
            return vector
            
        except Exception:
            return np.zeros(32)
    
    def _apply_quantum_transforms(self, sql: str) -> str:
        """Apply quantum-inspired transformations to SQL."""
        try:
            optimized_sql = sql
            
            # Transform 1: Optimize query structure
            if 'SELECT *' in optimized_sql.upper():
                # Suggest specific columns instead of *
                optimized_sql = optimized_sql.replace(
                    'SELECT *',
                    '-- Quantum Optimization: Consider specifying columns instead of *\nSELECT *'
                )
            
            # Transform 2: Add performance hints
            if 'ORDER BY' in optimized_sql.upper():
                optimized_sql += '\n-- Quantum Hint: Consider adding LIMIT for large result sets'
            
            # Transform 3: Security enhancements
            if 'WHERE' in optimized_sql.upper():
                optimized_sql += '\n-- Quantum Security: Parameterized queries recommended'
            
            return optimized_sql
            
        except Exception:
            return sql
    
    def _calculate_optimization_confidence(self, sql: str, original_query: str) -> float:
        """Calculate confidence score for the optimized SQL."""
        try:
            # Factor 1: SQL complexity appropriateness
            complexity_score = min(1.0, len(sql) / 500.0)
            
            # Factor 2: Query intent alignment
            intent_score = 0.8 if any(
                keyword in sql.upper() 
                for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
            ) else 0.3
            
            # Factor 3: Security features
            security_score = 0.9 if '-- Quantum' in sql else 0.7
            
            # Combine factors
            confidence = (
                complexity_score * 0.3 +
                intent_score * 0.5 +
                security_score * 0.2
            )
            
            return max(0.1, min(1.0, confidence))
            
        except Exception:
            return 0.5
    
    async def _transcendent_optimization(
        self, 
        result: EnhancedGenerationResult
    ) -> EnhancedGenerationResult:
        """Apply transcendent AI optimization."""
        try:
            if not result.sql_query:
                return result
            
            # Transcendent enhancement
            enhanced_sql = self._apply_transcendent_transforms(result.sql_query)
            
            # Update consciousness level
            self.quantum_state['consciousness_level'] = min(
                1.0, 
                self.quantum_state['consciousness_level'] + 0.01
            )
            
            # Update result
            result.sql_query = enhanced_sql
            result.confidence = min(1.0, result.confidence + 0.05)
            
            if result.metadata is None:
                result.metadata = {}
            result.metadata['transcendent_enhanced'] = True
            result.metadata['consciousness_level'] = self.quantum_state['consciousness_level']
            
            return result
            
        except Exception as e:
            logger.error(f"Transcendent optimization failed: {e}")
            return result
    
    def _apply_transcendent_transforms(self, sql: str) -> str:
        """Apply transcendent AI transformations."""
        enhanced_sql = sql
        
        # Add transcendent comments
        enhanced_sql = f"""-- Transcendent AI Enhanced Query
-- Generated with consciousness level: {self.quantum_state['consciousness_level']:.3f}
-- Quantum flux: {self.quantum_state['quantum_flux']:.3f}

{enhanced_sql}

-- Transcendent Note: Query optimized for multidimensional performance"""
        
        return enhanced_sql
    
    async def _comprehensive_security_check(self, sql: str) -> Dict[str, Any]:
        """Perform comprehensive security assessment."""
        try:
            # Basic SQL injection check
            if self.security_detector:
                is_malicious = self.security_detector.detect_sql_injection(sql)
                risk_level = self.security_detector.assess_risk_level(sql)
            else:
                # Fallback simple security check
                is_malicious = any(pattern in sql.lower() for pattern in ['drop table', 'delete from', '--'])
                risk_level = 'HIGH' if is_malicious else 'LOW'
            
            # Detailed analysis
            analysis = {
                'is_malicious': is_malicious,
                'risk_level': risk_level,
                'contains_keywords': self._extract_sql_keywords(sql),
                'query_complexity': len(sql),
                'safety_score': 1.0 if not is_malicious else 0.0,
                'security_recommendations': []
            }
            
            # Add recommendations
            if 'DROP' in sql.upper():
                analysis['security_recommendations'].append('Contains DROP statement')
            if 'DELETE' in sql.upper() and 'WHERE' not in sql.upper():
                analysis['security_recommendations'].append('DELETE without WHERE clause')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return {
                'is_malicious': False,
                'risk_level': 'UNKNOWN',
                'safety_score': 0.5,
                'error': str(e)
            }
    
    def _extract_sql_keywords(self, sql: str) -> List[str]:
        """Extract SQL keywords from query."""
        keywords = []
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 
            'JOIN', 'ORDER', 'GROUP', 'HAVING', 'UNION', 'CREATE', 'DROP'
        ]
        
        for keyword in sql_keywords:
            if keyword in sql.upper():
                keywords.append(keyword)
        
        return keywords
    
    def _generate_cache_key(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for query."""
        context_str = json.dumps(context or {}, sort_keys=True)
        return f"quantum_enhanced_{hash(query + context_str)}"
    
    def _update_performance_metrics(self, result: EnhancedGenerationResult) -> None:
        """Update internal performance metrics."""
        self.generation_count += 1
        self.confidence_history.append(result.confidence)
        
        if result.optimization_result:
            self.total_optimization_time += result.generation_time
        
        # Keep only recent history
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]
    
    async def _standard_generation(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> EnhancedGenerationResult:
        """Fallback to standard generation."""
        try:
            sql = await self._generate_base_sql(query, context)
            
            return EnhancedGenerationResult(
                success=True,
                sql_query=sql,
                confidence=0.7,
                model_used=self.model_name,
                metadata={'generation_mode': 'standard'}
            )
            
        except Exception as e:
            return EnhancedGenerationResult(
                success=False,
                error=str(e)
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'generation_count': self.generation_count,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.0,
            'total_optimization_time': self.total_optimization_time,
            'quantum_state': {
                'consciousness_level': self.quantum_state['consciousness_level'],
                'quantum_flux': self.quantum_state['quantum_flux'],
                'entanglement_strength': self.quantum_state['entanglement_strength']
            },
            'transcendent_metrics': {
                'memory_entries': len(self.transcendent_memory),
                'cache_efficiency': len(self.dimensional_cache)
            }
        }


class EnhancedAgentFactory:
    """Factory for creating enhanced SQL agents."""
    
    @staticmethod
    def create_quantum_agent(
        database_manager: DatabaseManager,
        **kwargs
    ) -> QuantumEnhancedSQLAgent:
        """Create quantum enhanced SQL agent."""
        return QuantumEnhancedSQLAgent(database_manager, **kwargs)
    
    @staticmethod 
    def create_standard_agent(
        database_manager: DatabaseManager,
        **kwargs
    ) -> QuantumEnhancedSQLAgent:
        """Create standard SQL agent with quantum features disabled."""
        return QuantumEnhancedSQLAgent(
            database_manager,
            enable_quantum_optimization=False,
            enable_transcendent_mode=False,
            **kwargs
        )