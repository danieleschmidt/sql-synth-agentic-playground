"""Multi-modal AI intelligence system for advanced SQL synthesis.

This module implements sophisticated multi-modal AI capabilities including:
- Visual query understanding from charts/diagrams
- Natural language + visual context integration
- Multi-model ensemble reasoning
- Cross-modal semantic alignment
- Intelligent context fusion
"""

import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    SCHEMA = "schema"
    CONTEXT = "context"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class ModalInput:
    """Multi-modal input representation."""
    modality: ModalityType
    content: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SemanticAlignment:
    """Semantic alignment between modalities."""
    source_modality: ModalityType
    target_modality: ModalityType
    alignment_score: float
    alignment_features: Dict[str, Any]
    confidence: float


class VisualQueryParser:
    """Advanced visual query understanding system."""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        self.chart_types = {
            'bar_chart': self._parse_bar_chart,
            'line_chart': self._parse_line_chart,
            'pie_chart': self._parse_pie_chart,
            'scatter_plot': self._parse_scatter_plot,
            'dashboard': self._parse_dashboard,
            'schema_diagram': self._parse_schema_diagram
        }
        
    def parse_visual_input(self, image_data: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """Parse visual input and extract query-relevant information.
        
        Args:
            image_data: Image data in various formats
            
        Returns:
            Parsed visual information with SQL generation hints
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image_data, str):
                # Assume base64 encoded
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError(f"Unsupported image format: {type(image_data)}")
                
            # Analyze image characteristics
            image_features = self._extract_image_features(image)
            
            # Detect chart/diagram type
            chart_type = self._detect_chart_type(image, image_features)
            
            # Parse specific chart type
            if chart_type in self.chart_types:
                parsing_result = self.chart_types[chart_type](image, image_features)
            else:
                parsing_result = self._parse_generic_visual(image, image_features)
                
            return {
                'visual_type': chart_type,
                'parsing_result': parsing_result,
                'image_features': image_features,
                'sql_hints': self._generate_sql_hints(chart_type, parsing_result),
                'confidence': parsing_result.get('confidence', 0.7),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Visual parsing failed: {e}")
            return {
                'error': str(e),
                'visual_type': 'unknown',
                'confidence': 0.0,
                'timestamp': time.time()
            }
    
    def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic image features."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Basic features
        width, height = image.size
        aspect_ratio = width / height
        
        # Color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5] if colors else []
        
        # Simple complexity estimation
        pixel_array = np.array(image)
        complexity_score = np.std(pixel_array) / 255.0
        
        return {
            'dimensions': (width, height),
            'aspect_ratio': aspect_ratio,
            'dominant_colors': dominant_colors,
            'complexity_score': complexity_score,
            'total_pixels': width * height
        }
    
    def _detect_chart_type(self, image: Image.Image, features: Dict[str, Any]) -> str:
        """Detect the type of chart or diagram."""
        # Simplified chart type detection
        aspect_ratio = features['aspect_ratio']
        complexity = features['complexity_score']
        
        # Heuristic-based detection
        if 0.8 <= aspect_ratio <= 1.2 and complexity > 0.3:
            return 'pie_chart'
        elif aspect_ratio > 1.5 and complexity > 0.2:
            return 'bar_chart'
        elif aspect_ratio > 1.2 and complexity > 0.15:
            return 'line_chart'
        elif complexity > 0.4:
            return 'dashboard'
        elif complexity < 0.1:
            return 'schema_diagram'
        else:
            return 'scatter_plot'
    
    def _parse_bar_chart(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse bar chart and extract data structure."""
        return {
            'chart_type': 'bar_chart',
            'suggested_queries': [
                'GROUP BY with aggregation functions',
                'Categorical data analysis',
                'Ranking and TOP N queries'
            ],
            'data_structure': {
                'categories': 'categorical_column',
                'values': 'numeric_column',
                'aggregation': ['SUM', 'COUNT', 'AVG']
            },
            'sql_patterns': [
                'SELECT category, SUM(value) FROM table GROUP BY category ORDER BY SUM(value) DESC',
                'SELECT category, COUNT(*) as count FROM table GROUP BY category'
            ],
            'confidence': 0.8
        }
    
    def _parse_line_chart(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse line chart and extract time series patterns."""
        return {
            'chart_type': 'line_chart',
            'suggested_queries': [
                'Time series analysis',
                'Trend analysis with date functions',
                'Moving averages and window functions'
            ],
            'data_structure': {
                'time_column': 'date/timestamp',
                'value_column': 'numeric',
                'trend_analysis': True
            },
            'sql_patterns': [
                'SELECT DATE(timestamp), AVG(value) FROM table GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)',
                'SELECT timestamp, value, LAG(value) OVER (ORDER BY timestamp) as prev_value FROM table'
            ],
            'confidence': 0.8
        }
    
    def _parse_pie_chart(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pie chart and extract proportion data."""
        return {
            'chart_type': 'pie_chart',
            'suggested_queries': [
                'Percentage calculations',
                'Proportion analysis',
                'Categorical distribution'
            ],
            'data_structure': {
                'categories': 'categorical_column',
                'proportions': 'calculated_percentage',
                'total': 'SUM(value)'
            },
            'sql_patterns': [
                'SELECT category, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM table) as percentage FROM table GROUP BY category',
                'SELECT category, SUM(value) / (SELECT SUM(value) FROM table) * 100 as percentage FROM table GROUP BY category'
            ],
            'confidence': 0.75
        }
    
    def _parse_scatter_plot(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse scatter plot and extract correlation patterns."""
        return {
            'chart_type': 'scatter_plot',
            'suggested_queries': [
                'Correlation analysis',
                'Two-variable relationships',
                'Statistical functions'
            ],
            'data_structure': {
                'x_variable': 'numeric_column_1',
                'y_variable': 'numeric_column_2',
                'correlation': True
            },
            'sql_patterns': [
                'SELECT x_col, y_col, CORR(x_col, y_col) OVER () as correlation FROM table',
                'SELECT x_col, y_col FROM table WHERE x_col BETWEEN ? AND ? AND y_col BETWEEN ? AND ?'
            ],
            'confidence': 0.7
        }
    
    def _parse_dashboard(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse dashboard with multiple visualizations."""
        return {
            'chart_type': 'dashboard',
            'suggested_queries': [
                'Multiple table joins',
                'Complex aggregations',
                'KPI calculations',
                'Multi-dimensional analysis'
            ],
            'data_structure': {
                'multiple_tables': True,
                'kpis': ['COUNT', 'SUM', 'AVG', 'RATIO'],
                'time_dimension': True,
                'categorical_dimensions': True
            },
            'sql_patterns': [
                'SELECT * FROM (subquery1) JOIN (subquery2) ON common_key',
                'WITH kpi_calc AS (...) SELECT * FROM kpi_calc',
                'SELECT dimension, SUM(metric) as total FROM fact_table GROUP BY ROLLUP(dimension)'
            ],
            'confidence': 0.85
        }
    
    def _parse_schema_diagram(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse database schema diagram."""
        return {
            'chart_type': 'schema_diagram',
            'suggested_queries': [
                'Table relationship queries',
                'Foreign key joins',
                'Entity relationship analysis'
            ],
            'data_structure': {
                'tables': 'multiple_entities',
                'relationships': 'foreign_keys',
                'cardinality': 'one_to_many_many_to_many'
            },
            'sql_patterns': [
                'SELECT t1.*, t2.* FROM table1 t1 JOIN table2 t2 ON t1.id = t2.foreign_id',
                'SELECT COUNT(*) FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.foreign_id WHERE t2.id IS NULL'
            ],
            'confidence': 0.9
        }
    
    def _parse_generic_visual(self, image: Image.Image, features: Dict[str, Any]) -> Dict[str, Any]:
        """Parse generic visual content."""
        return {
            'chart_type': 'generic',
            'suggested_queries': [
                'General data exploration',
                'Basic SELECT statements',
                'Data discovery queries'
            ],
            'data_structure': {
                'unknown_structure': True,
                'exploration_needed': True
            },
            'sql_patterns': [
                'SELECT * FROM table LIMIT 10',
                'DESCRIBE table',
                'SHOW TABLES'
            ],
            'confidence': 0.5
        }
    
    def _generate_sql_hints(self, chart_type: str, parsing_result: Dict[str, Any]) -> List[str]:
        """Generate SQL generation hints based on visual analysis."""
        hints = []
        
        data_structure = parsing_result.get('data_structure', {})
        sql_patterns = parsing_result.get('sql_patterns', [])
        
        # Add structure-based hints
        if 'categories' in data_structure:
            hints.append('Use GROUP BY for categorical analysis')
        if 'time_column' in data_structure:
            hints.append('Consider date/time functions and window functions')
        if 'proportions' in data_structure:
            hints.append('Calculate percentages and ratios')
        if 'correlation' in data_structure:
            hints.append('Use statistical functions like CORR()')
        if 'multiple_tables' in data_structure:
            hints.append('Consider multi-table joins and CTEs')
        
        # Add pattern-based hints
        hints.extend([f"Consider pattern: {pattern}" for pattern in sql_patterns[:2]])
        
        return hints


class ContextualFusionEngine:
    """Advanced context fusion for multi-modal inputs."""
    
    def __init__(self):
        self.fusion_strategies = {
            'weighted_average': self._weighted_fusion,
            'attention_mechanism': self._attention_fusion,
            'ensemble_voting': self._ensemble_fusion,
            'semantic_alignment': self._semantic_fusion
        }
        
    def fuse_multi_modal_inputs(
        self, 
        inputs: List[ModalInput], 
        strategy: str = 'semantic_alignment'
    ) -> Dict[str, Any]:
        """Fuse multiple modal inputs into unified representation.
        
        Args:
            inputs: List of modal inputs
            strategy: Fusion strategy to use
            
        Returns:
            Fused representation with enhanced context
        """
        if not inputs:
            return {'error': 'No inputs provided for fusion'}
            
        if strategy not in self.fusion_strategies:
            strategy = 'semantic_alignment'
            
        try:
            fusion_result = self.fusion_strategies[strategy](inputs)
            
            # Enhance with cross-modal insights
            cross_modal_insights = self._extract_cross_modal_insights(inputs, fusion_result)
            
            return {
                'fused_representation': fusion_result,
                'cross_modal_insights': cross_modal_insights,
                'fusion_strategy': strategy,
                'input_modalities': [inp.modality.value for inp in inputs],
                'confidence': self._calculate_fusion_confidence(inputs, fusion_result),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Context fusion failed: {e}")
            return {
                'error': str(e),
                'fusion_strategy': strategy,
                'timestamp': time.time()
            }
    
    def _weighted_fusion(self, inputs: List[ModalInput]) -> Dict[str, Any]:
        """Weighted average fusion based on confidence scores."""
        total_weight = sum(inp.confidence for inp in inputs)
        if total_weight == 0:
            return {'error': 'Zero total weight for fusion'}
            
        fused_content = {}
        weighted_metadata = {}
        
        for inp in inputs:
            weight = inp.confidence / total_weight
            
            # Fuse content
            if isinstance(inp.content, dict):
                for key, value in inp.content.items():
                    if key not in fused_content:
                        fused_content[key] = []
                    fused_content[key].append({'value': value, 'weight': weight, 'modality': inp.modality.value})
            
            # Fuse metadata
            for key, value in inp.metadata.items():
                if key not in weighted_metadata:
                    weighted_metadata[key] = []
                weighted_metadata[key].append({'value': value, 'weight': weight})
        
        return {
            'fusion_type': 'weighted_average',
            'fused_content': fused_content,
            'fused_metadata': weighted_metadata,
            'total_weight': total_weight
        }
    
    def _attention_fusion(self, inputs: List[ModalInput]) -> Dict[str, Any]:
        """Attention-based fusion focusing on most relevant inputs."""
        # Calculate attention weights based on relevance
        attention_weights = self._calculate_attention_weights(inputs)
        
        # Apply attention to fuse inputs
        attended_content = {}
        attention_summary = {}
        
        for i, inp in enumerate(inputs):
            weight = attention_weights[i]
            attention_summary[inp.modality.value] = weight
            
            if isinstance(inp.content, dict):
                for key, value in inp.content.items():
                    if key not in attended_content:
                        attended_content[key] = []
                    attended_content[key].append({
                        'value': value, 
                        'attention_weight': weight,
                        'modality': inp.modality.value
                    })
        
        return {
            'fusion_type': 'attention_mechanism',
            'attended_content': attended_content,
            'attention_weights': attention_summary,
            'primary_modality': max(attention_summary.items(), key=lambda x: x[1])[0]
        }
    
    def _ensemble_fusion(self, inputs: List[ModalInput]) -> Dict[str, Any]:
        """Ensemble voting fusion for robust decisions."""
        ensemble_results = {}
        voting_summary = {}
        
        # Extract features from each modality
        for inp in inputs:
            modality = inp.modality.value
            voting_summary[modality] = {
                'confidence': inp.confidence,
                'vote_weight': self._calculate_vote_weight(inp)
            }
            
            if isinstance(inp.content, dict):
                for key, value in inp.content.items():
                    if key not in ensemble_results:
                        ensemble_results[key] = {}
                    
                    if modality not in ensemble_results[key]:
                        ensemble_results[key][modality] = []
                    ensemble_results[key][modality].append(value)
        
        # Calculate ensemble consensus
        consensus_results = {}
        for key, modality_values in ensemble_results.items():
            consensus_results[key] = self._calculate_ensemble_consensus(modality_values, voting_summary)
        
        return {
            'fusion_type': 'ensemble_voting',
            'ensemble_results': consensus_results,
            'voting_summary': voting_summary,
            'consensus_strength': self._calculate_consensus_strength(consensus_results)
        }
    
    def _semantic_fusion(self, inputs: List[ModalInput]) -> Dict[str, Any]:
        """Semantic alignment-based fusion."""
        # Calculate semantic alignments between modalities
        alignments = []
        for i in range(len(inputs)):
            for j in range(i + 1, len(inputs)):
                alignment = self._calculate_semantic_alignment(inputs[i], inputs[j])
                alignments.append(alignment)
        
        # Create semantic graph
        semantic_graph = self._build_semantic_graph(inputs, alignments)
        
        # Fuse based on semantic coherence
        semantic_fusion = self._apply_semantic_fusion(inputs, semantic_graph)
        
        return {
            'fusion_type': 'semantic_alignment',
            'semantic_fusion': semantic_fusion,
            'semantic_alignments': [
                {
                    'source': align.source_modality.value,
                    'target': align.target_modality.value,
                    'score': align.alignment_score,
                    'confidence': align.confidence
                } for align in alignments
            ],
            'semantic_coherence': self._calculate_semantic_coherence(alignments)
        }
    
    def _calculate_attention_weights(self, inputs: List[ModalInput]) -> List[float]:
        """Calculate attention weights for inputs."""
        # Simplified attention calculation
        base_weights = [inp.confidence for inp in inputs]
        
        # Boost weights based on modality importance
        modality_boosts = {
            ModalityType.TEXT: 1.2,
            ModalityType.IMAGE: 1.1,
            ModalityType.SCHEMA: 1.3,
            ModalityType.CONTEXT: 1.0,
            ModalityType.TEMPORAL: 0.9,
            ModalityType.SPATIAL: 0.8
        }
        
        boosted_weights = [
            base_weights[i] * modality_boosts.get(inputs[i].modality, 1.0)
            for i in range(len(inputs))
        ]
        
        # Normalize to sum to 1
        total_weight = sum(boosted_weights)
        if total_weight == 0:
            return [1.0 / len(inputs)] * len(inputs)
        
        return [w / total_weight for w in boosted_weights]
    
    def _calculate_vote_weight(self, inp: ModalInput) -> float:
        """Calculate voting weight for input."""
        base_weight = inp.confidence
        
        # Adjust based on content complexity
        if isinstance(inp.content, dict):
            complexity_bonus = min(len(inp.content) * 0.1, 0.3)
            base_weight += complexity_bonus
        
        return min(base_weight, 1.0)
    
    def _calculate_ensemble_consensus(
        self, 
        modality_values: Dict[str, List[Any]], 
        voting_summary: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate ensemble consensus for values."""
        consensus = {
            'values': modality_values,
            'consensus_score': 0.0,
            'agreement_level': 'low'
        }
        
        # Calculate weighted consensus
        total_weight = sum(voting_summary[mod]['vote_weight'] for mod in modality_values.keys())
        if total_weight > 0:
            consensus_score = sum(
                voting_summary[mod]['vote_weight'] * voting_summary[mod]['confidence']
                for mod in modality_values.keys()
            ) / total_weight
            
            consensus['consensus_score'] = consensus_score
            
            if consensus_score >= 0.8:
                consensus['agreement_level'] = 'high'
            elif consensus_score >= 0.6:
                consensus['agreement_level'] = 'medium'
        
        return consensus
    
    def _calculate_consensus_strength(self, consensus_results: Dict[str, Any]) -> float:
        """Calculate overall consensus strength."""
        if not consensus_results:
            return 0.0
        
        consensus_scores = [
            result.get('consensus_score', 0.0) 
            for result in consensus_results.values() 
            if isinstance(result, dict)
        ]
        
        if not consensus_scores:
            return 0.0
        
        return sum(consensus_scores) / len(consensus_scores)
    
    def _calculate_semantic_alignment(self, input1: ModalInput, input2: ModalInput) -> SemanticAlignment:
        """Calculate semantic alignment between two inputs."""
        # Simplified semantic alignment calculation
        base_score = 0.5
        
        # Boost score if both contain similar concepts
        if isinstance(input1.content, dict) and isinstance(input2.content, dict):
            common_keys = set(input1.content.keys()) & set(input2.content.keys())
            key_similarity = len(common_keys) / max(len(input1.content), len(input2.content), 1)
            base_score += key_similarity * 0.3
        
        # Modality compatibility boost
        modality_compatibility = {
            (ModalityType.TEXT, ModalityType.CONTEXT): 0.9,
            (ModalityType.IMAGE, ModalityType.SCHEMA): 0.8,
            (ModalityType.TEXT, ModalityType.IMAGE): 0.7,
            (ModalityType.SCHEMA, ModalityType.CONTEXT): 0.8
        }
        
        modality_pair = (input1.modality, input2.modality)
        if modality_pair in modality_compatibility:
            compatibility_boost = modality_compatibility[modality_pair] * 0.2
            base_score += compatibility_boost
        elif (modality_pair[1], modality_pair[0]) in modality_compatibility:
            compatibility_boost = modality_compatibility[(modality_pair[1], modality_pair[0])] * 0.2
            base_score += compatibility_boost
        
        alignment_score = min(base_score, 1.0)
        confidence = (input1.confidence + input2.confidence) / 2
        
        return SemanticAlignment(
            source_modality=input1.modality,
            target_modality=input2.modality,
            alignment_score=alignment_score,
            alignment_features={
                'key_similarity': key_similarity if 'key_similarity' in locals() else 0.0,
                'modality_compatibility': compatibility_boost if 'compatibility_boost' in locals() else 0.0
            },
            confidence=confidence
        )
    
    def _build_semantic_graph(self, inputs: List[ModalInput], alignments: List[SemanticAlignment]) -> Dict[str, Any]:
        """Build semantic graph from inputs and alignments."""
        nodes = {inp.modality.value: {'input': inp, 'connections': []} for inp in inputs}
        
        for alignment in alignments:
            source = alignment.source_modality.value
            target = alignment.target_modality.value
            
            if source in nodes:
                nodes[source]['connections'].append({
                    'target': target,
                    'weight': alignment.alignment_score,
                    'confidence': alignment.confidence
                })
            
            if target in nodes:
                nodes[target]['connections'].append({
                    'target': source,
                    'weight': alignment.alignment_score,
                    'confidence': alignment.confidence
                })
        
        return {
            'nodes': nodes,
            'edge_count': len(alignments),
            'connectivity_score': len(alignments) / max(len(inputs) * (len(inputs) - 1) / 2, 1)
        }
    
    def _apply_semantic_fusion(self, inputs: List[ModalInput], semantic_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Apply semantic fusion based on semantic graph."""
        fused_content = {}
        semantic_weights = {}
        
        # Calculate semantic centrality for each modality
        nodes = semantic_graph['nodes']
        for modality, node_data in nodes.items():
            connections = node_data['connections']
            centrality = sum(conn['weight'] * conn['confidence'] for conn in connections)
            semantic_weights[modality] = centrality
        
        # Normalize weights
        total_weight = sum(semantic_weights.values())
        if total_weight > 0:
            semantic_weights = {k: v / total_weight for k, v in semantic_weights.items()}
        
        # Fuse content based on semantic weights
        for inp in inputs:
            modality = inp.modality.value
            weight = semantic_weights.get(modality, 1.0 / len(inputs))
            
            if isinstance(inp.content, dict):
                for key, value in inp.content.items():
                    if key not in fused_content:
                        fused_content[key] = []
                    fused_content[key].append({
                        'value': value,
                        'semantic_weight': weight,
                        'modality': modality
                    })
        
        return {
            'semantic_weights': semantic_weights,
            'fused_content': fused_content,
            'dominant_modality': max(semantic_weights.items(), key=lambda x: x[1])[0] if semantic_weights else None
        }
    
    def _calculate_semantic_coherence(self, alignments: List[SemanticAlignment]) -> float:
        """Calculate overall semantic coherence score."""
        if not alignments:
            return 0.0
        
        alignment_scores = [align.alignment_score for align in alignments]
        confidence_scores = [align.confidence for align in alignments]
        
        # Weighted average of alignment scores by confidence
        weighted_score = sum(
            score * conf for score, conf in zip(alignment_scores, confidence_scores)
        ) / sum(confidence_scores)
        
        return weighted_score
    
    def _extract_cross_modal_insights(self, inputs: List[ModalInput], fusion_result: Dict[str, Any]) -> List[str]:
        """Extract insights from cross-modal analysis."""
        insights = []
        
        modalities = [inp.modality.value for inp in inputs]
        
        if 'text' in modalities and 'image' in modalities:
            insights.append("Text-visual alignment detected - consider context-aware query generation")
        
        if 'schema' in modalities:
            insights.append("Database schema context available - enable relationship-aware queries")
        
        if 'temporal' in modalities:
            insights.append("Temporal context detected - suggest time-based analysis functions")
        
        if len(modalities) >= 3:
            insights.append("Multi-modal context rich environment - enable advanced fusion strategies")
        
        # Add fusion-specific insights
        fusion_type = fusion_result.get('fusion_type', 'unknown')
        if fusion_type == 'semantic_alignment':
            coherence = fusion_result.get('semantic_coherence', 0.0)
            if coherence > 0.8:
                insights.append("High semantic coherence - confident multi-modal understanding")
            elif coherence < 0.4:
                insights.append("Low semantic coherence - consider additional context")
        
        return insights
    
    def _calculate_fusion_confidence(self, inputs: List[ModalInput], fusion_result: Dict[str, Any]) -> float:
        """Calculate overall confidence in fusion result."""
        # Base confidence from input confidences
        input_confidences = [inp.confidence for inp in inputs]
        base_confidence = sum(input_confidences) / len(input_confidences)
        
        # Adjust based on fusion quality
        fusion_type = fusion_result.get('fusion_type', 'unknown')
        
        if fusion_type == 'semantic_alignment':
            coherence = fusion_result.get('semantic_coherence', 0.5)
            confidence_adjustment = coherence * 0.2  # Up to 20% boost
        elif fusion_type == 'ensemble_voting':
            consensus_strength = fusion_result.get('consensus_strength', 0.5)
            confidence_adjustment = consensus_strength * 0.15  # Up to 15% boost
        elif fusion_type == 'attention_mechanism':
            max_attention = max(fusion_result.get('attention_weights', {}).values(), default=0.5)
            confidence_adjustment = max_attention * 0.1  # Up to 10% boost
        else:
            confidence_adjustment = 0.0
        
        # Consider modality diversity bonus
        unique_modalities = len(set(inp.modality for inp in inputs))
        diversity_bonus = min(unique_modalities * 0.05, 0.15)  # Up to 15% for diverse inputs
        
        final_confidence = min(base_confidence + confidence_adjustment + diversity_bonus, 1.0)
        return final_confidence


class MultiModalSQLSynthesizer:
    """Advanced multi-modal SQL synthesis system."""
    
    def __init__(self, base_agent, visual_parser: VisualQueryParser, fusion_engine: ContextualFusionEngine):
        self.base_agent = base_agent
        self.visual_parser = visual_parser
        self.fusion_engine = fusion_engine
        self.synthesis_history = []
        
    def synthesize_from_multimodal_input(
        self,
        natural_query: str,
        visual_inputs: Optional[List[Any]] = None,
        context_data: Optional[Dict[str, Any]] = None,
        schema_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize SQL from multi-modal inputs.
        
        Args:
            natural_query: Natural language query
            visual_inputs: Optional visual inputs (images, charts, etc.)
            context_data: Optional contextual information
            schema_info: Optional database schema information
            
        Returns:
            Enhanced SQL synthesis result with multi-modal insights
        """
        start_time = time.time()
        
        try:
            # Prepare modal inputs
            modal_inputs = [ModalInput(
                modality=ModalityType.TEXT,
                content={'query': natural_query},
                confidence=0.9,
                metadata={'length': len(natural_query)}
            )]
            
            # Process visual inputs
            if visual_inputs:
                for visual_input in visual_inputs:
                    visual_result = self.visual_parser.parse_visual_input(visual_input)
                    if 'error' not in visual_result:
                        modal_inputs.append(ModalInput(
                            modality=ModalityType.IMAGE,
                            content=visual_result,
                            confidence=visual_result.get('confidence', 0.7),
                            metadata={'visual_type': visual_result.get('visual_type')}
                        ))
            
            # Add context data
            if context_data:
                modal_inputs.append(ModalInput(
                    modality=ModalityType.CONTEXT,
                    content=context_data,
                    confidence=0.8,
                    metadata={'context_keys': list(context_data.keys())}
                ))
            
            # Add schema information
            if schema_info:
                modal_inputs.append(ModalInput(
                    modality=ModalityType.SCHEMA,
                    content=schema_info,
                    confidence=0.95,
                    metadata={'table_count': len(schema_info.get('tables', []))}
                ))
            
            # Fuse multi-modal inputs
            fusion_result = self.fusion_engine.fuse_multi_modal_inputs(modal_inputs)
            
            if 'error' in fusion_result:
                raise ValueError(f"Fusion failed: {fusion_result['error']}")
            
            # Generate enhanced context for SQL synthesis
            enhanced_context = self._generate_enhanced_context(fusion_result, natural_query)
            
            # Generate SQL using base agent with enhanced context
            base_result = self.base_agent.generate_sql(enhanced_context['enhanced_query'])
            
            if not base_result.get('success', False):
                raise ValueError(f"Base SQL generation failed: {base_result.get('error', 'Unknown error')}")
            
            # Apply multi-modal enhancements
            enhanced_result = self._apply_multimodal_enhancements(
                base_result, fusion_result, enhanced_context
            )
            
            synthesis_time = time.time() - start_time
            
            # Record synthesis
            synthesis_record = {
                'natural_query': natural_query,
                'modal_inputs_count': len(modal_inputs),
                'fusion_confidence': fusion_result.get('confidence', 0.0),
                'synthesis_time': synthesis_time,
                'success': True,
                'timestamp': time.time()
            }
            self.synthesis_history.append(synthesis_record)
            
            return {
                **enhanced_result,
                'multimodal_metadata': {
                    'modal_inputs': [inp.modality.value for inp in modal_inputs],
                    'fusion_strategy': fusion_result.get('fusion_strategy'),
                    'fusion_confidence': fusion_result.get('confidence'),
                    'cross_modal_insights': fusion_result.get('cross_modal_insights', []),
                    'synthesis_time': synthesis_time
                },
                'enhanced_context': enhanced_context,
                'fusion_details': fusion_result
            }
            
        except Exception as e:
            synthesis_time = time.time() - start_time
            logger.error(f"Multi-modal SQL synthesis failed: {e}")
            
            # Record failed synthesis
            synthesis_record = {
                'natural_query': natural_query,
                'modal_inputs_count': len(modal_inputs) if 'modal_inputs' in locals() else 0,
                'synthesis_time': synthesis_time,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            self.synthesis_history.append(synthesis_record)
            
            return {
                'success': False,
                'error': str(e),
                'synthesis_time': synthesis_time,
                'multimodal_metadata': {
                    'modal_inputs': [],
                    'fusion_confidence': 0.0
                }
            }
    
    def _generate_enhanced_context(self, fusion_result: Dict[str, Any], natural_query: str) -> Dict[str, Any]:
        """Generate enhanced context for SQL synthesis."""
        enhanced_query = natural_query
        context_enhancements = []
        
        # Extract insights from fusion result
        cross_modal_insights = fusion_result.get('cross_modal_insights', [])
        fused_representation = fusion_result.get('fused_representation', {})
        
        # Apply visual insights
        if 'image' in fusion_result.get('input_modalities', []):
            visual_hints = self._extract_visual_hints(fused_representation)
            context_enhancements.extend(visual_hints)
            enhanced_query += f" [Visual Context: {', '.join(visual_hints)}]"
        
        # Apply schema insights
        if 'schema' in fusion_result.get('input_modalities', []):
            schema_hints = self._extract_schema_hints(fused_representation)
            context_enhancements.extend(schema_hints)
            enhanced_query += f" [Schema Context: {', '.join(schema_hints)}]"
        
        # Apply contextual insights
        if 'context' in fusion_result.get('input_modalities', []):
            context_hints = self._extract_context_hints(fused_representation)
            context_enhancements.extend(context_hints)
            enhanced_query += f" [Additional Context: {', '.join(context_hints)}]"
        
        return {
            'original_query': natural_query,
            'enhanced_query': enhanced_query,
            'context_enhancements': context_enhancements,
            'cross_modal_insights': cross_modal_insights,
            'fusion_confidence': fusion_result.get('confidence', 0.0)
        }
    
    def _extract_visual_hints(self, fused_representation: Dict[str, Any]) -> List[str]:
        """Extract SQL hints from visual information."""
        hints = []
        
        # Extract from fusion content
        fused_content = fused_representation.get('fused_content', {})
        
        for key, values in fused_content.items():
            if isinstance(values, list):
                for value_info in values:
                    if value_info.get('modality') == 'image':
                        if 'sql_hints' in value_info.get('value', {}):
                            hints.extend(value_info['value']['sql_hints'])
                        if 'chart_type' in value_info.get('value', {}):
                            chart_type = value_info['value']['chart_type']
                            hints.append(f"Optimize for {chart_type} visualization")
        
        return list(set(hints))  # Remove duplicates
    
    def _extract_schema_hints(self, fused_representation: Dict[str, Any]) -> List[str]:
        """Extract SQL hints from schema information."""
        hints = []
        
        fused_content = fused_representation.get('fused_content', {})
        
        for key, values in fused_content.items():
            if isinstance(values, list):
                for value_info in values:
                    if value_info.get('modality') == 'schema':
                        schema_data = value_info.get('value', {})
                        if 'tables' in schema_data:
                            table_count = len(schema_data['tables'])
                            if table_count > 1:
                                hints.append("Consider multi-table joins")
                            hints.append(f"Available tables: {', '.join(schema_data['tables'][:3])}")
        
        return hints
    
    def _extract_context_hints(self, fused_representation: Dict[str, Any]) -> List[str]:
        """Extract SQL hints from contextual information."""
        hints = []
        
        fused_content = fused_representation.get('fused_content', {})
        
        for key, values in fused_content.items():
            if isinstance(values, list):
                for value_info in values:
                    if value_info.get('modality') == 'context':
                        context_data = value_info.get('value', {})
                        if 'time_range' in context_data:
                            hints.append("Apply time-based filtering")
                        if 'user_preferences' in context_data:
                            hints.append("Personalize results based on preferences")
                        if 'business_rules' in context_data:
                            hints.append("Apply business rule constraints")
        
        return hints
    
    def _apply_multimodal_enhancements(
        self, 
        base_result: Dict[str, Any], 
        fusion_result: Dict[str, Any],
        enhanced_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply multi-modal enhancements to base SQL result."""
        enhanced_result = base_result.copy()
        
        # Enhance metadata with multi-modal insights
        if 'metadata' not in enhanced_result:
            enhanced_result['metadata'] = {}
        
        enhanced_result['metadata'].update({
            'multimodal_enhancement': True,
            'modal_inputs_used': fusion_result.get('input_modalities', []),
            'fusion_confidence': fusion_result.get('confidence', 0.0),
            'context_enhancements': enhanced_context.get('context_enhancements', []),
            'cross_modal_insights': enhanced_context.get('cross_modal_insights', [])
        })
        
        # Enhance SQL query with multi-modal optimizations
        if 'sql_query' in enhanced_result:
            optimized_sql = self._optimize_sql_with_multimodal_context(
                enhanced_result['sql_query'], fusion_result
            )
            enhanced_result['sql_query'] = optimized_sql
            enhanced_result['sql_optimized'] = True
        
        # Add confidence boost for successful fusion
        fusion_confidence = fusion_result.get('confidence', 0.0)
        if fusion_confidence > 0.7:
            enhanced_result['confidence_boost'] = 0.1
            if 'metadata' in enhanced_result:
                enhanced_result['metadata']['confidence_enhanced'] = True
        
        return enhanced_result
    
    def _optimize_sql_with_multimodal_context(self, sql_query: str, fusion_result: Dict[str, Any]) -> str:
        """Optimize SQL query based on multi-modal context."""
        optimized_sql = sql_query
        
        # Apply visual-based optimizations
        if 'image' in fusion_result.get('input_modalities', []):
            # Add appropriate ORDER BY for visualizations
            if 'ORDER BY' not in sql_query.upper() and 'GROUP BY' in sql_query.upper():
                optimized_sql = sql_query.rstrip(';') + ' ORDER BY 2 DESC;'
        
        # Apply schema-based optimizations
        if 'schema' in fusion_result.get('input_modalities', []):
            # Ensure proper table aliases for multi-table queries
            if sql_query.upper().count('JOIN') > 0 and 'AS ' not in sql_query.upper():
                # This is a simplified optimization - in practice, would need more sophisticated parsing
                pass
        
        return optimized_sql
    
    def get_synthesis_analytics(self) -> Dict[str, Any]:
        """Get analytics on multi-modal synthesis performance."""
        if not self.synthesis_history:
            return {'message': 'No synthesis history available'}
        
        total_syntheses = len(self.synthesis_history)
        successful_syntheses = sum(1 for s in self.synthesis_history if s.get('success', False))
        
        # Modal input statistics
        modal_input_stats = {}
        for synthesis in self.synthesis_history:
            count = synthesis.get('modal_inputs_count', 0)
            if count not in modal_input_stats:
                modal_input_stats[count] = 0
            modal_input_stats[count] += 1
        
        # Performance statistics
        synthesis_times = [s.get('synthesis_time', 0) for s in self.synthesis_history if s.get('success', False)]
        fusion_confidences = [s.get('fusion_confidence', 0) for s in self.synthesis_history if s.get('success', False)]
        
        return {
            'total_syntheses': total_syntheses,
            'successful_syntheses': successful_syntheses,
            'success_rate': successful_syntheses / total_syntheses if total_syntheses > 0 else 0.0,
            'modal_input_distribution': modal_input_stats,
            'performance_metrics': {
                'avg_synthesis_time': sum(synthesis_times) / len(synthesis_times) if synthesis_times else 0.0,
                'avg_fusion_confidence': sum(fusion_confidences) / len(fusion_confidences) if fusion_confidences else 0.0,
                'min_synthesis_time': min(synthesis_times) if synthesis_times else 0.0,
                'max_synthesis_time': max(synthesis_times) if synthesis_times else 0.0
            },
            'recent_activity': self.synthesis_history[-10:] if len(self.synthesis_history) > 10 else self.synthesis_history
        }


# Global instances
global_visual_parser = VisualQueryParser()
global_fusion_engine = ContextualFusionEngine()

# Example usage function
def create_multimodal_synthesizer(base_agent) -> MultiModalSQLSynthesizer:
    """Create a multi-modal SQL synthesizer with the given base agent."""
    return MultiModalSQLSynthesizer(
        base_agent=base_agent,
        visual_parser=global_visual_parser,
        fusion_engine=global_fusion_engine
    )