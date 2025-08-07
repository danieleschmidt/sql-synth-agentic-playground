"""Internationalization (i18n) support for SQL Synthesis Agent.

This module provides multi-language support, localization, and
cross-platform compatibility features.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import locale
import re


logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the application."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    KOREAN = "ko"


@dataclass
class LocalizationContext:
    """Context for localization operations."""
    language: SupportedLanguage
    region: Optional[str] = None
    timezone: Optional[str] = None
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    currency: str = "USD"


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        """Initialize i18n manager.
        
        Args:
            default_language: Default language for the application
        """
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.localization_context = LocalizationContext(language=default_language)
        
        # Load default translations
        self._load_default_translations()
    
    def _load_default_translations(self) -> None:
        """Load default translations for all supported languages."""
        # English (base language)
        self.translations["en"] = {
            # Application messages
            "app.title": "Sentiment-Aware SQL Synthesis Agent",
            "app.description": "Intelligent Natural Language to SQL Converter",
            "app.demo_mode": "Demo Mode: No Database Connection",
            
            # Input/Output
            "input.query_placeholder": "e.g., Show me the best performing products this month",
            "input.query_help": "Type your query in natural language. The system will analyze emotional context.",
            "input.generate_button": "Generate SQL",
            "output.sentiment_analysis": "Sentiment Analysis",
            "output.generated_sql": "Generated SQL Query",
            "output.query_results": "Query Results",
            
            # Sentiment analysis
            "sentiment.polarity": "Sentiment",
            "sentiment.intent": "Query Intent",
            "sentiment.confidence": "Confidence",
            "sentiment.biases": "Detected Biases",
            "sentiment.keywords": "Emotional Keywords",
            "sentiment.very_positive": "Very Positive",
            "sentiment.positive": "Positive",
            "sentiment.neutral": "Neutral",
            "sentiment.negative": "Negative",
            "sentiment.very_negative": "Very Negative",
            
            # Query intents
            "intent.analytical": "Analytical",
            "intent.exploratory": "Exploratory",
            "intent.investigative": "Investigative",
            "intent.comparative": "Comparative",
            "intent.trending": "Trending",
            "intent.problem_solving": "Problem Solving",
            
            # Error messages
            "error.empty_query": "Query cannot be empty",
            "error.query_too_long": "Query is too long",
            "error.security_violation": "Security validation failed",
            "error.database_connection": "Database connection error",
            "error.sql_execution": "SQL execution failed",
            "error.sentiment_analysis": "Sentiment analysis failed",
            
            # Success messages
            "success.sql_generated": "SQL generated successfully",
            "success.query_executed": "Query executed successfully",
            "success.results_loaded": "Results loaded successfully",
            
            # Features
            "feature.sentiment_analysis": "Sentiment analysis for query understanding",
            "feature.intent_detection": "Intent detection (analytical, exploratory, investigative)",
            "feature.temporal_bias": "Temporal bias detection (recent, historical, trending)",
            "feature.magnitude_bias": "Magnitude bias detection (top, bottom, extreme)",
            "feature.security_validation": "Security validation and SQL injection prevention",
        }
        
        # Spanish translations
        self.translations["es"] = {
            "app.title": "Agente de Síntesis SQL con Análisis de Sentimientos",
            "app.description": "Conversor Inteligente de Lenguaje Natural a SQL",
            "app.demo_mode": "Modo Demo: Sin Conexión a Base de Datos",
            
            "input.query_placeholder": "ej., Muéstrame los productos con mejor rendimiento este mes",
            "input.query_help": "Escriba su consulta en lenguaje natural. El sistema analizará el contexto emocional.",
            "input.generate_button": "Generar SQL",
            "output.sentiment_analysis": "Análisis de Sentimientos",
            "output.generated_sql": "Consulta SQL Generada",
            "output.query_results": "Resultados de la Consulta",
            
            "sentiment.polarity": "Sentimiento",
            "sentiment.intent": "Intención de Consulta",
            "sentiment.confidence": "Confianza",
            "sentiment.biases": "Sesgos Detectados",
            "sentiment.keywords": "Palabras Clave Emocionales",
            "sentiment.very_positive": "Muy Positivo",
            "sentiment.positive": "Positivo",
            "sentiment.neutral": "Neutral",
            "sentiment.negative": "Negativo",
            "sentiment.very_negative": "Muy Negativo",
            
            "intent.analytical": "Analítico",
            "intent.exploratory": "Exploratorio",
            "intent.investigative": "Investigativo",
            "intent.comparative": "Comparativo",
            "intent.trending": "Tendencias",
            "intent.problem_solving": "Resolución de Problemas",
            
            "error.empty_query": "La consulta no puede estar vacía",
            "error.query_too_long": "La consulta es demasiado larga",
            "error.security_violation": "Error de validación de seguridad",
            "error.database_connection": "Error de conexión a base de datos",
            "error.sql_execution": "Error en ejecución de SQL",
            "error.sentiment_analysis": "Error en análisis de sentimientos",
            
            "success.sql_generated": "SQL generado exitosamente",
            "success.query_executed": "Consulta ejecutada exitosamente",
            "success.results_loaded": "Resultados cargados exitosamente",
        }
        
        # French translations
        self.translations["fr"] = {
            "app.title": "Agent de Synthèse SQL avec Analyse de Sentiment",
            "app.description": "Convertisseur Intelligent de Langage Naturel vers SQL",
            "app.demo_mode": "Mode Démo: Aucune Connexion Base de Données",
            
            "input.query_placeholder": "ex., Montrez-moi les produits les plus performants ce mois-ci",
            "input.query_help": "Tapez votre requête en langage naturel. Le système analysera le contexte émotionnel.",
            "input.generate_button": "Générer SQL",
            "output.sentiment_analysis": "Analyse de Sentiment",
            "output.generated_sql": "Requête SQL Générée",
            "output.query_results": "Résultats de la Requête",
            
            "sentiment.polarity": "Sentiment",
            "sentiment.intent": "Intention de Requête",
            "sentiment.confidence": "Confiance",
            "sentiment.biases": "Biais Détectés",
            "sentiment.keywords": "Mots-clés Émotionnels",
            "sentiment.very_positive": "Très Positif",
            "sentiment.positive": "Positif",
            "sentiment.neutral": "Neutre",
            "sentiment.negative": "Négatif",
            "sentiment.very_negative": "Très Négatif",
            
            "intent.analytical": "Analytique",
            "intent.exploratory": "Exploratoire",
            "intent.investigative": "Investigatif",
            "intent.comparative": "Comparatif",
            "intent.trending": "Tendances",
            "intent.problem_solving": "Résolution de Problèmes",
            
            "error.empty_query": "La requête ne peut pas être vide",
            "error.query_too_long": "La requête est trop longue",
            "error.security_violation": "Échec de validation de sécurité",
            "error.database_connection": "Erreur de connexion base de données",
            "error.sql_execution": "Échec d'exécution SQL",
            "error.sentiment_analysis": "Échec d'analyse de sentiment",
            
            "success.sql_generated": "SQL généré avec succès",
            "success.query_executed": "Requête exécutée avec succès",
            "success.results_loaded": "Résultats chargés avec succès",
        }
        
        # German translations
        self.translations["de"] = {
            "app.title": "Sentiment-bewusster SQL-Synthese-Agent",
            "app.description": "Intelligenter Natürlichsprache-zu-SQL-Konverter",
            "app.demo_mode": "Demo-Modus: Keine Datenbankverbindung",
            
            "input.query_placeholder": "z.B., Zeigen Sie mir die leistungsstärksten Produkte diesen Monat",
            "input.query_help": "Geben Sie Ihre Anfrage in natürlicher Sprache ein. Das System analysiert emotionalen Kontext.",
            "input.generate_button": "SQL Generieren",
            "output.sentiment_analysis": "Sentiment-Analyse",
            "output.generated_sql": "Generierte SQL-Abfrage",
            "output.query_results": "Abfrageergebnisse",
            
            "sentiment.polarity": "Sentiment",
            "sentiment.intent": "Abfrageabsicht",
            "sentiment.confidence": "Vertrauen",
            "sentiment.biases": "Erkannte Verzerrungen",
            "sentiment.keywords": "Emotionale Schlüsselwörter",
            "sentiment.very_positive": "Sehr Positiv",
            "sentiment.positive": "Positiv",
            "sentiment.neutral": "Neutral",
            "sentiment.negative": "Negativ",
            "sentiment.very_negative": "Sehr Negativ",
            
            "intent.analytical": "Analytisch",
            "intent.exploratory": "Erkundend",
            "intent.investigative": "Untersuchend",
            "intent.comparative": "Vergleichend",
            "intent.trending": "Trending",
            "intent.problem_solving": "Problemlösung",
            
            "error.empty_query": "Abfrage kann nicht leer sein",
            "error.query_too_long": "Abfrage ist zu lang",
            "error.security_violation": "Sicherheitsvalidierung fehlgeschlagen",
            "error.database_connection": "Datenbankverbindungsfehler",
            "error.sql_execution": "SQL-Ausführung fehlgeschlagen",
            "error.sentiment_analysis": "Sentiment-Analyse fehlgeschlagen",
            
            "success.sql_generated": "SQL erfolgreich generiert",
            "success.query_executed": "Abfrage erfolgreich ausgeführt",
            "success.results_loaded": "Ergebnisse erfolgreich geladen",
        }
        
        # Japanese translations
        self.translations["ja"] = {
            "app.title": "感情認識SQL合成エージェント",
            "app.description": "インテリジェント自然言語SQL変換器",
            "app.demo_mode": "デモモード: データベース接続なし",
            
            "input.query_placeholder": "例：今月の最高パフォーマンス製品を表示",
            "input.query_help": "自然言語でクエリを入力してください。システムが感情的コンテキストを分析します。",
            "input.generate_button": "SQL生成",
            "output.sentiment_analysis": "感情分析",
            "output.generated_sql": "生成されたSQLクエリ",
            "output.query_results": "クエリ結果",
            
            "sentiment.polarity": "感情",
            "sentiment.intent": "クエリ意図",
            "sentiment.confidence": "信頼度",
            "sentiment.biases": "検出されたバイアス",
            "sentiment.keywords": "感情キーワード",
            "sentiment.very_positive": "非常にポジティブ",
            "sentiment.positive": "ポジティブ",
            "sentiment.neutral": "中立",
            "sentiment.negative": "ネガティブ",
            "sentiment.very_negative": "非常にネガティブ",
            
            "intent.analytical": "分析的",
            "intent.exploratory": "探索的",
            "intent.investigative": "調査的",
            "intent.comparative": "比較的",
            "intent.trending": "トレンド",
            "intent.problem_solving": "問題解決",
            
            "error.empty_query": "クエリを空にできません",
            "error.query_too_long": "クエリが長すぎます",
            "error.security_violation": "セキュリティ検証に失敗",
            "error.database_connection": "データベース接続エラー",
            "error.sql_execution": "SQL実行に失敗",
            "error.sentiment_analysis": "感情分析に失敗",
            
            "success.sql_generated": "SQLが正常に生成されました",
            "success.query_executed": "クエリが正常に実行されました",
            "success.results_loaded": "結果が正常に読み込まれました",
        }
        
        # Chinese (Simplified) translations
        self.translations["zh-CN"] = {
            "app.title": "情感感知SQL合成代理",
            "app.description": "智能自然语言到SQL转换器",
            "app.demo_mode": "演示模式：无数据库连接",
            
            "input.query_placeholder": "例如：显示本月表现最佳的产品",
            "input.query_help": "用自然语言输入您的查询。系统将分析情感背景。",
            "input.generate_button": "生成SQL",
            "output.sentiment_analysis": "情感分析",
            "output.generated_sql": "生成的SQL查询",
            "output.query_results": "查询结果",
            
            "sentiment.polarity": "情感",
            "sentiment.intent": "查询意图",
            "sentiment.confidence": "置信度",
            "sentiment.biases": "检测到的偏差",
            "sentiment.keywords": "情感关键词",
            "sentiment.very_positive": "非常积极",
            "sentiment.positive": "积极",
            "sentiment.neutral": "中性",
            "sentiment.negative": "消极",
            "sentiment.very_negative": "非常消极",
            
            "intent.analytical": "分析性",
            "intent.exploratory": "探索性",
            "intent.investigative": "调查性",
            "intent.comparative": "比较性",
            "intent.trending": "趋势性",
            "intent.problem_solving": "问题解决",
            
            "error.empty_query": "查询不能为空",
            "error.query_too_long": "查询太长",
            "error.security_violation": "安全验证失败",
            "error.database_connection": "数据库连接错误",
            "error.sql_execution": "SQL执行失败",
            "error.sentiment_analysis": "情感分析失败",
            
            "success.sql_generated": "SQL生成成功",
            "success.query_executed": "查询执行成功",
            "success.results_loaded": "结果加载成功",
        }
    
    def set_language(self, language: Union[str, SupportedLanguage]) -> None:
        """Set the current language.
        
        Args:
            language: Language to set (string code or enum)
        """
        if isinstance(language, str):
            try:
                language = SupportedLanguage(language)
            except ValueError:
                logger.warning("Unsupported language: %s, using default", language)
                language = self.default_language
        
        self.current_language = language
        self.localization_context.language = language
        logger.info("Language changed to: %s", language.value)
    
    def get_text(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """Get localized text for a given key.
        
        Args:
            key: Translation key
            language: Override language (uses current if None)
            **kwargs: Format parameters for the text
            
        Returns:
            Localized text string
        """
        if language is None:
            language = self.current_language
        
        lang_code = language.value
        
        # Try to get translation
        if lang_code in self.translations and key in self.translations[lang_code]:
            text = self.translations[lang_code][key]
        elif key in self.translations["en"]:  # Fallback to English
            text = self.translations["en"][key]
            logger.debug("Translation not found for %s in %s, using English", key, lang_code)
        else:
            # Fallback to key itself
            text = key
            logger.warning("Translation key not found: %s", key)
        
        # Format with provided kwargs
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError) as e:
            logger.warning("Failed to format translation %s: %s", key, e)
            return text
    
    def detect_user_language(self, accept_language_header: Optional[str] = None) -> SupportedLanguage:
        """Detect user's preferred language.
        
        Args:
            accept_language_header: HTTP Accept-Language header value
            
        Returns:
            Detected or default language
        """
        detected_languages = []
        
        # Try to parse Accept-Language header
        if accept_language_header:
            detected_languages.extend(self._parse_accept_language(accept_language_header))
        
        # Try system locale
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                lang_code = system_locale.split('_')[0]
                detected_languages.append(lang_code)
        except Exception as e:
            logger.debug("Failed to get system locale: %s", e)
        
        # Find first supported language
        for lang_code in detected_languages:
            try:
                return SupportedLanguage(lang_code)
            except ValueError:
                continue
        
        # Return default if nothing detected
        return self.default_language
    
    def _parse_accept_language(self, accept_lang: str) -> List[str]:
        """Parse Accept-Language header.
        
        Args:
            accept_lang: Accept-Language header value
            
        Returns:
            List of language codes in preference order
        """
        languages = []
        
        # Parse "en-US,en;q=0.9,es;q=0.8"
        for item in accept_lang.split(','):
            item = item.strip()
            
            # Extract language code
            if ';q=' in item:
                lang, quality = item.split(';q=', 1)
                quality_val = float(quality.strip())
            else:
                lang = item
                quality_val = 1.0
            
            # Normalize language code
            lang = lang.strip().split('-')[0].lower()
            languages.append((lang, quality_val))
        
        # Sort by quality and return language codes
        languages.sort(key=lambda x: x[1], reverse=True)
        return [lang for lang, _ in languages]
    
    def format_number(self, number: Union[int, float], language: Optional[SupportedLanguage] = None) -> str:
        """Format number according to locale.
        
        Args:
            number: Number to format
            language: Language for formatting
            
        Returns:
            Formatted number string
        """
        if language is None:
            language = self.current_language
        
        # Simple formatting based on language
        if language in [SupportedLanguage.ENGLISH]:
            return f"{number:,}"  # 1,234.56
        elif language in [SupportedLanguage.FRENCH, SupportedLanguage.GERMAN]:
            return f"{number:,}".replace(',', ' ')  # 1 234,56
        elif language in [SupportedLanguage.SPANISH]:
            return f"{number:,}".replace(',', '.')  # 1.234,56
        else:
            return str(number)
    
    def format_date(self, date_obj, language: Optional[SupportedLanguage] = None) -> str:
        """Format date according to locale.
        
        Args:
            date_obj: Date object to format
            language: Language for formatting
            
        Returns:
            Formatted date string
        """
        if language is None:
            language = self.current_language
        
        # Date format patterns by language
        formats = {
            SupportedLanguage.ENGLISH: "%m/%d/%Y",
            SupportedLanguage.SPANISH: "%d/%m/%Y",
            SupportedLanguage.FRENCH: "%d/%m/%Y",
            SupportedLanguage.GERMAN: "%d.%m.%Y",
            SupportedLanguage.JAPANESE: "%Y年%m月%d日",
            SupportedLanguage.CHINESE_SIMPLIFIED: "%Y年%m月%d日",
        }
        
        format_string = formats.get(language, "%Y-%m-%d")
        
        try:
            return date_obj.strftime(format_string)
        except Exception as e:
            logger.warning("Failed to format date: %s", e)
            return str(date_obj)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages.
        
        Returns:
            List of language info dictionaries
        """
        language_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français", 
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE_SIMPLIFIED: "简体中文",
            SupportedLanguage.PORTUGUESE: "Português",
            SupportedLanguage.ITALIAN: "Italiano",
            SupportedLanguage.RUSSIAN: "Русский",
            SupportedLanguage.KOREAN: "한국어",
        }
        
        return [
            {
                "code": lang.value,
                "name": language_names.get(lang, lang.value),
                "available": lang.value in self.translations
            }
            for lang in SupportedLanguage
        ]


# Global i18n manager instance
i18n_manager = I18nManager()


def _(key: str, **kwargs) -> str:
    """Shorthand function for getting localized text.
    
    Args:
        key: Translation key
        **kwargs: Format parameters
        
    Returns:
        Localized text
    """
    return i18n_manager.get_text(key, **kwargs)


def set_language(language: Union[str, SupportedLanguage]) -> None:
    """Set the application language.
    
    Args:
        language: Language to set
    """
    i18n_manager.set_language(language)


def detect_language(accept_language: Optional[str] = None) -> SupportedLanguage:
    """Detect user's preferred language.
    
    Args:
        accept_language: HTTP Accept-Language header
        
    Returns:
        Detected language
    """
    return i18n_manager.detect_user_language(accept_language)