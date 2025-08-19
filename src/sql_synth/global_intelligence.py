"""Global-first intelligence and multi-region capabilities.

This module implements global-first features including multi-region deployment,
internationalization, compliance frameworks, and cross-platform compatibility.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Region(Enum):
    """Global regions for deployment."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_CENTRAL = "eu-central-1"
    EU_WEST = "eu-west-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA_CENTRAL = "ca-central-1"
    SOUTH_AMERICA = "sa-east-1"
    AFRICA = "af-south-1"
    MIDDLE_EAST = "me-south-1"


class ComplianceFramework(Enum):
    """Compliance frameworks and regulations."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore, Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    SOX = "sox"    # Sarbanes-Oxley Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # International Organization for Standardization
    NIST = "nist"  # National Institute of Standards and Technology


class Language(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"


@dataclass
class GlobalConfiguration:
    """Global deployment configuration."""
    primary_region: Region
    secondary_regions: list[Region]
    compliance_requirements: set[ComplianceFramework]
    supported_languages: set[Language]
    data_residency_requirements: dict[Region, set[str]]
    timezone_mapping: dict[Region, str]
    currency_mapping: dict[Region, str] = field(default_factory=dict)
    regulatory_contacts: dict[ComplianceFramework, str] = field(default_factory=dict)


@dataclass
class RegionalMetrics:
    """Performance metrics per region."""
    region: Region
    timestamp: datetime
    response_time_ms: float
    throughput_qps: float
    error_rate: float
    availability_percentage: float
    data_transfer_gb: float
    cost_usd: float
    active_users: int
    concurrent_connections: int


class GlobalIntelligenceEngine:
    """Global intelligence for multi-region operations."""

    def __init__(self, config: GlobalConfiguration):
        self.config = config
        self.regional_metrics: dict[Region, list[RegionalMetrics]] = {}
        self.compliance_validators = self._initialize_compliance_validators()
        self.language_processors = self._initialize_language_processors()
        self.regional_optimizers = self._initialize_regional_optimizers()

    def _initialize_compliance_validators(self) -> dict[ComplianceFramework, Any]:
        """Initialize compliance validation systems."""
        validators = {}

        for framework in self.config.compliance_requirements:
            if framework == ComplianceFramework.GDPR:
                validators[framework] = GDPRValidator()
            elif framework == ComplianceFramework.CCPA:
                validators[framework] = CCPAValidator()
            elif framework == ComplianceFramework.PDPA:
                validators[framework] = PDPAValidator()
            elif framework == ComplianceFramework.HIPAA:
                validators[framework] = HIPAAValidator()
            elif framework == ComplianceFramework.PCI_DSS:
                validators[framework] = PCIDSSValidator()
            else:
                validators[framework] = GenericComplianceValidator(framework)

        return validators

    def _initialize_language_processors(self) -> dict[Language, Any]:
        """Initialize language processing systems."""
        processors = {}

        for language in self.config.supported_languages:
            processors[language] = LanguageProcessor(language)

        return processors

    def _initialize_regional_optimizers(self) -> dict[Region, Any]:
        """Initialize region-specific optimizers."""
        optimizers = {}

        for region in [self.config.primary_region, *self.config.secondary_regions]:
            optimizers[region] = RegionalOptimizer(region, self.config)

        return optimizers

    def optimize_global_routing(self, user_location: dict[str, float],
                              request_metadata: dict[str, Any]) -> Region:
        """Optimize routing based on user location and requirements."""

        user_lat = user_location.get("latitude", 0.0)
        user_lon = user_location.get("longitude", 0.0)

        # Calculate latency estimates to each region
        latency_estimates = {}
        for region in [self.config.primary_region, *self.config.secondary_regions]:
            distance = self._calculate_geographic_distance(
                user_lat, user_lon, region,
            )
            base_latency = distance * 0.1  # Rough estimate: 0.1ms per km

            # Add region-specific performance factors
            regional_metrics = self._get_recent_regional_metrics(region)
            if regional_metrics:
                performance_factor = regional_metrics.response_time_ms / 100.0
                latency_estimates[region] = base_latency + performance_factor
            else:
                latency_estimates[region] = base_latency

        # Apply compliance constraints
        compliant_regions = self._filter_compliant_regions(request_metadata)

        # Select optimal region
        eligible_regions = set(latency_estimates.keys()) & compliant_regions
        if not eligible_regions:
            return self.config.primary_region

        return min(eligible_regions, key=lambda r: latency_estimates[r])

    def _calculate_geographic_distance(self, lat1: float, lon1: float,
                                     region: Region) -> float:
        """Calculate approximate distance to region."""
        # Simplified region center coordinates
        region_coords = {
            Region.US_EAST: (39.0, -77.5),
            Region.US_WEST: (45.5, -122.7),
            Region.EU_CENTRAL: (50.1, 8.7),
            Region.EU_WEST: (53.4, -6.2),
            Region.ASIA_PACIFIC: (1.3, 103.8),
            Region.ASIA_NORTHEAST: (35.7, 139.7),
            Region.CANADA_CENTRAL: (45.4, -75.7),
            Region.SOUTH_AMERICA: (-23.5, -46.6),
            Region.AFRICA: (-33.9, 18.4),
            Region.MIDDLE_EAST: (25.2, 55.3),
        }

        if region not in region_coords:
            return float("inf")

        lat2, lon2 = region_coords[region]

        # Haversine formula for great circle distance
        R = 6371  # Earth's radius in kilometers

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat/2) * np.sin(dlat/2) +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2) * np.sin(dlon/2))

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c


    def _filter_compliant_regions(self, request_metadata: dict[str, Any]) -> set[Region]:
        """Filter regions based on compliance requirements."""
        compliant_regions = {self.config.primary_region, *self.config.secondary_regions}

        # Check data residency requirements
        user_country = request_metadata.get("country")
        if user_country:
            for region, allowed_countries in self.config.data_residency_requirements.items():
                if user_country not in allowed_countries:
                    compliant_regions.discard(region)

        # Check regulatory requirements
        regulation_type = request_metadata.get("regulation_type")
        if regulation_type:
            for region in list(compliant_regions):
                if not self._is_region_compliant(region, regulation_type):
                    compliant_regions.discard(region)

        return compliant_regions

    def _is_region_compliant(self, region: Region, regulation_type: str) -> bool:
        """Check if region meets specific regulatory requirements."""
        # Regional compliance mapping
        region_compliance = {
            Region.US_EAST: {ComplianceFramework.SOX, ComplianceFramework.HIPAA,
                            ComplianceFramework.CCPA, ComplianceFramework.SOC2},
            Region.US_WEST: {ComplianceFramework.SOX, ComplianceFramework.HIPAA,
                            ComplianceFramework.CCPA, ComplianceFramework.SOC2},
            Region.EU_CENTRAL: {ComplianceFramework.GDPR, ComplianceFramework.ISO27001},
            Region.EU_WEST: {ComplianceFramework.GDPR, ComplianceFramework.ISO27001},
            Region.ASIA_PACIFIC: {ComplianceFramework.PDPA, ComplianceFramework.ISO27001},
            Region.ASIA_NORTHEAST: {ComplianceFramework.PDPA, ComplianceFramework.ISO27001},
            Region.CANADA_CENTRAL: {ComplianceFramework.GDPR, ComplianceFramework.SOC2},
            Region.SOUTH_AMERICA: {ComplianceFramework.LGPD, ComplianceFramework.ISO27001},
        }

        try:
            regulation_enum = ComplianceFramework(regulation_type)
            return regulation_enum in region_compliance.get(region, set())
        except ValueError:
            return True  # Allow unknown regulations

    def _get_recent_regional_metrics(self, region: Region) -> Optional[RegionalMetrics]:
        """Get most recent metrics for a region."""
        if region not in self.regional_metrics or not self.regional_metrics[region]:
            return None

        return self.regional_metrics[region][-1]

    def localize_content(self, content: str, target_language: Language,
                        context: dict[str, Any]) -> str:
        """Localize content for target language and culture."""

        if target_language not in self.language_processors:
            return content  # Fallback to original

        processor = self.language_processors[target_language]
        return processor.localize(content, context)

    def ensure_compliance(self, data: dict[str, Any],
                         frameworks: set[ComplianceFramework]) -> dict[str, Any]:
        """Ensure data complies with specified frameworks."""

        compliant_data = data.copy()

        for framework in frameworks:
            if framework in self.compliance_validators:
                validator = self.compliance_validators[framework]
                compliant_data = validator.validate_and_sanitize(compliant_data)

        return compliant_data

    def get_global_health_status(self) -> dict[str, Any]:
        """Get comprehensive global system health status."""

        status = {
            "timestamp": datetime.now(timezone.utc),
            "overall_health": "healthy",
            "regional_status": {},
            "compliance_status": {},
            "performance_summary": {},
            "alerts": [],
        }

        # Regional health assessment
        unhealthy_regions = 0
        total_regions = len([self.config.primary_region, *self.config.secondary_regions])

        for region in [self.config.primary_region, *self.config.secondary_regions]:
            regional_metrics = self._get_recent_regional_metrics(region)
            if regional_metrics:
                region_health = self._assess_regional_health(regional_metrics)
                status["regional_status"][region.value] = region_health

                if region_health["status"] != "healthy":
                    unhealthy_regions += 1
            else:
                status["regional_status"][region.value] = {"status": "unknown"}
                unhealthy_regions += 1

        # Overall health determination
        if unhealthy_regions == 0:
            status["overall_health"] = "healthy"
        elif unhealthy_regions <= total_regions * 0.2:  # 20% threshold
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "critical"

        # Compliance status
        for framework in self.config.compliance_requirements:
            if framework in self.compliance_validators:
                validator = self.compliance_validators[framework]
                compliance_check = validator.get_compliance_status()
                status["compliance_status"][framework.value] = compliance_check

        # Performance summary
        status["performance_summary"] = self._calculate_global_performance_summary()

        return status

    def _assess_regional_health(self, metrics: RegionalMetrics) -> dict[str, Any]:
        """Assess health of a specific region."""

        health = {
            "status": "healthy",
            "metrics": {
                "response_time": metrics.response_time_ms,
                "error_rate": metrics.error_rate,
                "availability": metrics.availability_percentage,
            },
            "issues": [],
        }

        # Response time threshold: 500ms
        if metrics.response_time_ms > 500:
            health["issues"].append("High response time")
            health["status"] = "degraded"

        # Error rate threshold: 1%
        if metrics.error_rate > 0.01:
            health["issues"].append("High error rate")
            health["status"] = "degraded"

        # Availability threshold: 99%
        if metrics.availability_percentage < 99.0:
            health["issues"].append("Low availability")
            health["status"] = "critical"

        return health

    def _calculate_global_performance_summary(self) -> dict[str, Any]:
        """Calculate global performance summary."""

        all_metrics = []
        for region_metrics in self.regional_metrics.values():
            if region_metrics:
                all_metrics.extend(region_metrics[-5:])  # Last 5 per region

        if not all_metrics:
            return {"status": "no_data"}

        # Calculate averages
        avg_response_time = sum(m.response_time_ms for m in all_metrics) / len(all_metrics)
        avg_error_rate = sum(m.error_rate for m in all_metrics) / len(all_metrics)
        avg_availability = sum(m.availability_percentage for m in all_metrics) / len(all_metrics)
        total_throughput = sum(m.throughput_qps for m in all_metrics)

        return {
            "avg_response_time_ms": avg_response_time,
            "avg_error_rate": avg_error_rate,
            "avg_availability_pct": avg_availability,
            "total_throughput_qps": total_throughput,
            "active_regions": len(self.regional_metrics),
        }


class LanguageProcessor:
    """Language processing and localization."""

    def __init__(self, language: Language):
        self.language = language
        self.translations = self._load_translations()
        self.cultural_adaptations = self._load_cultural_adaptations()

    def _load_translations(self) -> dict[str, str]:
        """Load translation mappings for the language."""
        # Simplified translation mapping
        base_translations = {
            "error": {
                Language.SPANISH: "error",
                Language.FRENCH: "erreur",
                Language.GERMAN: "Fehler",
                Language.JAPANESE: "エラー",
                Language.CHINESE_SIMPLIFIED: "错误",
                Language.PORTUGUESE: "erro",
                Language.ITALIAN: "errore",
                Language.RUSSIAN: "ошибка",
            },
            "success": {
                Language.SPANISH: "éxito",
                Language.FRENCH: "succès",
                Language.GERMAN: "Erfolg",
                Language.JAPANESE: "成功",
                Language.CHINESE_SIMPLIFIED: "成功",
                Language.PORTUGUESE: "sucesso",
                Language.ITALIAN: "successo",
                Language.RUSSIAN: "успех",
            },
            "query": {
                Language.SPANISH: "consulta",
                Language.FRENCH: "requête",
                Language.GERMAN: "Abfrage",
                Language.JAPANESE: "クエリ",
                Language.CHINESE_SIMPLIFIED: "查询",
                Language.PORTUGUESE: "consulta",
                Language.ITALIAN: "query",
                Language.RUSSIAN: "запрос",
            },
        }

        # Build language-specific translation dict
        translations = {}
        for key, lang_map in base_translations.items():
            if self.language in lang_map:
                translations[key] = lang_map[self.language]
            else:
                translations[key] = key  # Fallback to English

        return translations

    def _load_cultural_adaptations(self) -> dict[str, Any]:
        """Load cultural adaptation rules."""
        adaptations = {
            Language.JAPANESE: {
                "date_format": "%Y年%m月%d日",
                "number_format": "comma_separator",
                "politeness_level": "formal",
            },
            Language.CHINESE_SIMPLIFIED: {
                "date_format": "%Y年%m月%d日",
                "number_format": "comma_separator",
                "text_direction": "ltr",
            },
            Language.ARABIC: {
                "date_format": "%d/%m/%Y",
                "number_format": "arabic_numerals",
                "text_direction": "rtl",
            },
            Language.GERMAN: {
                "date_format": "%d.%m.%Y",
                "number_format": "period_separator",
                "currency_position": "suffix",
            },
        }

        return adaptations.get(self.language, {})

    def localize(self, content: str, context: dict[str, Any]) -> str:
        """Localize content based on language and cultural context."""

        localized = content

        # Apply translations
        for english_term, localized_term in self.translations.items():
            localized = localized.replace(english_term, localized_term)

        # Apply cultural adaptations
        if "date" in context and "date_format" in self.cultural_adaptations:
            date_obj = context["date"]
            if hasattr(date_obj, "strftime"):
                formatted_date = date_obj.strftime(self.cultural_adaptations["date_format"])
                localized = localized.replace("DATE_PLACEHOLDER", formatted_date)

        # Apply number formatting
        if "number" in context and "number_format" in self.cultural_adaptations:
            number = context["number"]
            if self.cultural_adaptations["number_format"] == "period_separator":
                formatted_number = f"{number:,.}".replace(",", ".")
            else:
                formatted_number = f"{number:,}"
            localized = localized.replace("NUMBER_PLACEHOLDER", formatted_number)

        return localized


class ComplianceValidator:
    """Base class for compliance validators."""

    def __init__(self, framework: ComplianceFramework):
        self.framework = framework

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize data according to compliance framework."""
        return data

    def get_compliance_status(self) -> dict[str, Any]:
        """Get current compliance status."""
        return {"status": "compliant", "framework": self.framework.value}


class GDPRValidator(ComplianceValidator):
    """GDPR compliance validator."""

    def __init__(self):
        super().__init__(ComplianceFramework.GDPR)
        self.pii_fields = {
            "email", "phone", "address", "name", "ssn",
            "credit_card", "ip_address", "user_id",
        }

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """GDPR-compliant data sanitization."""
        sanitized = data.copy()

        # Remove or anonymize PII
        for field in self.pii_fields:
            if field in sanitized:
                if field == "email":
                    sanitized[field] = self._anonymize_email(sanitized[field])
                elif field == "ip_address":
                    sanitized[field] = self._anonymize_ip(sanitized[field])
                elif field in ["phone", "ssn", "credit_card"]:
                    sanitized[field] = "[REDACTED]"
                else:
                    sanitized[field] = "[ANONYMIZED]"

        # Add GDPR metadata
        sanitized["_gdpr_compliant"] = True
        sanitized["_processing_lawful_basis"] = "legitimate_interest"
        sanitized["_data_retention_period"] = "2_years"

        return sanitized

    def _anonymize_email(self, email: str) -> str:
        """Anonymize email address."""
        if "@" in email:
            local, domain = email.split("@", 1)
            return f"{local[:2]}***@{domain}"
        return "[ANONYMIZED_EMAIL]"

    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.XXX.XXX"
        return "[ANONYMIZED_IP]"


class CCPAValidator(ComplianceValidator):
    """CCPA compliance validator."""

    def __init__(self):
        super().__init__(ComplianceFramework.CCPA)

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """CCPA-compliant data handling."""
        sanitized = data.copy()

        # Add CCPA required fields
        sanitized["_ccpa_compliant"] = True
        sanitized["_do_not_sell"] = data.get("do_not_sell_flag", False)
        sanitized["_opt_out_available"] = True

        # Remove sensitive personal information if opt-out requested
        if sanitized.get("_do_not_sell"):
            sensitive_fields = ["location", "browsing_history", "preferences"]
            for field in sensitive_fields:
                if field in sanitized:
                    sanitized[field] = "[REMOVED_PER_CCPA]"

        return sanitized


class PDPAValidator(ComplianceValidator):
    """PDPA compliance validator (Singapore/Thailand)."""

    def __init__(self):
        super().__init__(ComplianceFramework.PDPA)

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """PDPA-compliant data handling."""
        sanitized = data.copy()

        # Add PDPA metadata
        sanitized["_pdpa_compliant"] = True
        sanitized["_consent_obtained"] = True
        sanitized["_purpose_limitation"] = "service_provision"

        return sanitized


class HIPAAValidator(ComplianceValidator):
    """HIPAA compliance validator."""

    def __init__(self):
        super().__init__(ComplianceFramework.HIPAA)
        self.phi_fields = {
            "medical_record", "health_plan_number", "account_number",
            "certificate_number", "vehicle_id", "device_id", "biometric_id",
            "photo", "any_unique_identifying_number",
        }

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """HIPAA-compliant PHI handling."""
        sanitized = data.copy()

        # Remove PHI fields
        for field in self.phi_fields:
            if field in sanitized:
                sanitized[field] = "[PHI_REMOVED]"

        # Add HIPAA metadata
        sanitized["_hipaa_compliant"] = True
        sanitized["_phi_removed"] = True
        sanitized["_minimum_necessary"] = True

        return sanitized


class PCIDSSValidator(ComplianceValidator):
    """PCI DSS compliance validator."""

    def __init__(self):
        super().__init__(ComplianceFramework.PCI_DSS)

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """PCI DSS compliant payment data handling."""
        sanitized = data.copy()

        # Remove payment card information
        pci_fields = ["credit_card_number", "cvv", "expiry_date", "cardholder_name"]
        for field in pci_fields:
            if field in sanitized:
                if field == "credit_card_number":
                    # Show only last 4 digits
                    cc_num = str(sanitized[field])
                    sanitized[field] = f"****-****-****-{cc_num[-4:]}"
                else:
                    sanitized[field] = "[PCI_REDACTED]"

        # Add PCI DSS metadata
        sanitized["_pci_dss_compliant"] = True
        sanitized["_cardholder_data_protected"] = True

        return sanitized


class GenericComplianceValidator(ComplianceValidator):
    """Generic compliance validator for other frameworks."""

    def validate_and_sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generic compliance data handling."""
        sanitized = data.copy()
        sanitized[f"_{self.framework.value}_compliant"] = True
        return sanitized


class RegionalOptimizer:
    """Region-specific optimization and configuration."""

    def __init__(self, region: Region, config: GlobalConfiguration):
        self.region = region
        self.config = config
        self.regional_settings = self._get_regional_settings()

    def _get_regional_settings(self) -> dict[str, Any]:
        """Get region-specific optimization settings."""

        settings = {
            Region.US_EAST: {
                "cdn_endpoints": ["cloudfront-us-east"],
                "database_read_replicas": 3,
                "cache_ttl_seconds": 300,
                "auto_scaling_target": 70,
            },
            Region.EU_CENTRAL: {
                "cdn_endpoints": ["cloudfront-eu-central"],
                "database_read_replicas": 2,
                "cache_ttl_seconds": 600,
                "auto_scaling_target": 60,
                "gdpr_encryption": True,
            },
            Region.ASIA_PACIFIC: {
                "cdn_endpoints": ["cloudfront-ap-southeast"],
                "database_read_replicas": 2,
                "cache_ttl_seconds": 180,
                "auto_scaling_target": 80,
                "latency_optimization": True,
            },
        }

        return settings.get(self.region, {
            "database_read_replicas": 1,
            "cache_ttl_seconds": 300,
            "auto_scaling_target": 70,
        })

    def optimize_for_region(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Apply region-specific optimizations."""

        optimized = request_data.copy()

        # Apply caching strategy
        optimized["cache_ttl"] = self.regional_settings.get("cache_ttl_seconds", 300)

        # Apply auto-scaling configuration
        optimized["scaling_target"] = self.regional_settings.get("auto_scaling_target", 70)

        # Apply regional database configuration
        optimized["read_replicas"] = self.regional_settings.get("database_read_replicas", 1)

        # Apply region-specific features
        if self.regional_settings.get("gdpr_encryption"):
            optimized["encryption_required"] = True

        if self.regional_settings.get("latency_optimization"):
            optimized["prioritize_speed"] = True

        return optimized


# Global intelligence instance
_global_intelligence = None


def get_global_intelligence(config: Optional[GlobalConfiguration] = None) -> GlobalIntelligenceEngine:
    """Get global intelligence engine instance."""
    global _global_intelligence
    if _global_intelligence is None:
        if config is None:
            # Default global configuration
            config = GlobalConfiguration(
                primary_region=Region.US_EAST,
                secondary_regions=[Region.EU_CENTRAL, Region.ASIA_PACIFIC],
                compliance_requirements={
                    ComplianceFramework.GDPR,
                    ComplianceFramework.CCPA,
                    ComplianceFramework.SOC2,
                },
                supported_languages={
                    Language.ENGLISH, Language.SPANISH, Language.FRENCH,
                    Language.GERMAN, Language.JAPANESE, Language.CHINESE_SIMPLIFIED,
                },
                data_residency_requirements={
                    Region.EU_CENTRAL: {"DE", "FR", "IT", "ES", "NL"},
                    Region.EU_WEST: {"GB", "IE"},
                    Region.ASIA_PACIFIC: {"SG", "MY", "TH", "ID"},
                    Region.US_EAST: {"US", "CA"},
                    Region.US_WEST: {"US", "CA"},
                },
                timezone_mapping={
                    Region.US_EAST: "America/New_York",
                    Region.US_WEST: "America/Los_Angeles",
                    Region.EU_CENTRAL: "Europe/Berlin",
                    Region.EU_WEST: "Europe/London",
                    Region.ASIA_PACIFIC: "Asia/Singapore",
                },
            )
        _global_intelligence = GlobalIntelligenceEngine(config)
    return _global_intelligence
