"""Compliance and regulatory support for SQL Synthesis Agent.

This module provides GDPR, CCPA, PDPA compliance features including
data privacy, consent management, and audit trails.
"""

import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum


logger = logging.getLogger(__name__)


class DataProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    SQL_GENERATION = "sql_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    QUERY_OPTIMIZATION = "query_optimization"
    SECURITY_MONITORING = "security_monitoring"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    ERROR_DEBUGGING = "error_debugging"


class ConsentStatus(Enum):
    """User consent status for data processing."""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"


class DataClassification(Enum):
    """Data classification levels for privacy compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    user_id: str
    purpose: DataProcessingPurpose
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    consent_version: str = "1.0"
    legal_basis: str = "consent"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProcessingRecord:
    """Record of data processing activities."""
    processing_id: str
    user_id: Optional[str]
    purpose: DataProcessingPurpose
    data_categories: List[DataClassification]
    processing_start: datetime
    processing_end: Optional[datetime] = None
    data_source: str = "user_input"
    recipients: List[str] = field(default_factory=list)
    retention_period: Optional[int] = None  # days
    legal_basis: str = "legitimate_interest"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceManager:
    """Manager for regulatory compliance and data privacy."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.consent_records: Dict[str, Dict[str, ConsentRecord]] = {}  # user_id -> purpose -> record
        self.processing_records: List[DataProcessingRecord] = []
        self.data_retention_policies: Dict[DataProcessingPurpose, int] = {
            DataProcessingPurpose.SQL_GENERATION: 30,  # days
            DataProcessingPurpose.SENTIMENT_ANALYSIS: 7,
            DataProcessingPurpose.QUERY_OPTIMIZATION: 90,
            DataProcessingPurpose.SECURITY_MONITORING: 365,
            DataProcessingPurpose.PERFORMANCE_ANALYTICS: 180,
            DataProcessingPurpose.ERROR_DEBUGGING: 14,
        }
        
        # Privacy settings
        self.anonymization_enabled = True
        self.pseudonymization_enabled = True
        self.encryption_enabled = True
        
    def record_consent(
        self,
        user_id: str,
        purpose: DataProcessingPurpose,
        status: ConsentStatus,
        consent_version: str = "1.0",
        legal_basis: str = "consent",
        **metadata
    ) -> ConsentRecord:
        """Record user consent for data processing.
        
        Args:
            user_id: User identifier
            purpose: Purpose of data processing
            status: Consent status
            consent_version: Version of consent agreement
            legal_basis: Legal basis for processing
            **metadata: Additional metadata
            
        Returns:
            Created consent record
        """
        now = datetime.utcnow()
        
        consent = ConsentRecord(
            user_id=user_id,
            purpose=purpose,
            status=status,
            consent_version=consent_version,
            legal_basis=legal_basis,
            metadata=metadata
        )
        
        if status == ConsentStatus.GRANTED:
            consent.granted_at = now
            # Set expiration (1 year from grant)
            consent.expires_at = now + timedelta(days=365)
        elif status == ConsentStatus.WITHDRAWN:
            consent.withdrawn_at = now
        
        # Store consent record
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][purpose.value] = consent
        
        logger.info(
            "Consent recorded",
            extra={
                "user_id": self._hash_user_id(user_id),
                "purpose": purpose.value,
                "status": status.value,
                "legal_basis": legal_basis
            }
        )
        
        return consent
    
    def check_consent(self, user_id: str, purpose: DataProcessingPurpose) -> bool:
        """Check if user has valid consent for purpose.
        
        Args:
            user_id: User identifier
            purpose: Processing purpose to check
            
        Returns:
            True if consent is valid, False otherwise
        """
        if user_id not in self.consent_records:
            return False
        
        purpose_key = purpose.value
        if purpose_key not in self.consent_records[user_id]:
            return False
        
        consent = self.consent_records[user_id][purpose_key]
        
        # Check status
        if consent.status != ConsentStatus.GRANTED:
            return False
        
        # Check expiration
        if consent.expires_at and datetime.utcnow() > consent.expires_at:
            # Mark as expired
            consent.status = ConsentStatus.EXPIRED
            return False
        
        return True
    
    def withdraw_consent(self, user_id: str, purpose: DataProcessingPurpose) -> bool:
        """Withdraw user consent for a purpose.
        
        Args:
            user_id: User identifier
            purpose: Processing purpose
            
        Returns:
            True if consent was withdrawn, False if not found
        """
        if user_id not in self.consent_records:
            return False
        
        purpose_key = purpose.value
        if purpose_key not in self.consent_records[user_id]:
            return False
        
        consent = self.consent_records[user_id][purpose_key]
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.utcnow()
        
        logger.info(
            "Consent withdrawn",
            extra={
                "user_id": self._hash_user_id(user_id),
                "purpose": purpose.value
            }
        )
        
        return True
    
    def start_processing(
        self,
        purpose: DataProcessingPurpose,
        data_categories: List[DataClassification],
        user_id: Optional[str] = None,
        data_source: str = "user_input",
        legal_basis: str = "legitimate_interest",
        **metadata
    ) -> str:
        """Start a data processing activity.
        
        Args:
            purpose: Processing purpose
            data_categories: Categories of data being processed
            user_id: User identifier (if applicable)
            data_source: Source of the data
            legal_basis: Legal basis for processing
            **metadata: Additional metadata
            
        Returns:
            Processing ID for tracking
        """
        processing_id = f"proc_{int(time.time() * 1000)}_{hash(str(metadata))}"
        
        # Check consent if user_id provided and required
        if user_id and legal_basis == "consent":
            if not self.check_consent(user_id, purpose):
                raise PermissionError(f"No valid consent for {purpose.value}")
        
        record = DataProcessingRecord(
            processing_id=processing_id,
            user_id=user_id,
            purpose=purpose,
            data_categories=data_categories,
            processing_start=datetime.utcnow(),
            data_source=data_source,
            legal_basis=legal_basis,
            retention_period=self.data_retention_policies.get(purpose),
            metadata=metadata
        )
        
        self.processing_records.append(record)
        
        logger.info(
            "Processing started",
            extra={
                "processing_id": processing_id,
                "purpose": purpose.value,
                "user_id": self._hash_user_id(user_id) if user_id else None,
                "legal_basis": legal_basis
            }
        )
        
        return processing_id
    
    def end_processing(self, processing_id: str) -> bool:
        """End a data processing activity.
        
        Args:
            processing_id: Processing ID to end
            
        Returns:
            True if processing was ended, False if not found
        """
        for record in self.processing_records:
            if record.processing_id == processing_id:
                record.processing_end = datetime.utcnow()
                
                logger.info(
                    "Processing ended",
                    extra={
                        "processing_id": processing_id,
                        "duration_seconds": (
                            record.processing_end - record.processing_start
                        ).total_seconds()
                    }
                )
                
                return True
        
        return False
    
    def get_user_data_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's data processing (for GDPR Article 15).
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary containing user's data processing summary
        """
        # Get consent records
        consents = {}
        if user_id in self.consent_records:
            for purpose, consent in self.consent_records[user_id].items():
                consents[purpose] = {
                    "status": consent.status.value,
                    "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
                    "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
                    "legal_basis": consent.legal_basis
                }
        
        # Get processing records
        processing = []
        for record in self.processing_records:
            if record.user_id == user_id:
                processing.append({
                    "processing_id": record.processing_id,
                    "purpose": record.purpose.value,
                    "legal_basis": record.legal_basis,
                    "processing_start": record.processing_start.isoformat(),
                    "processing_end": record.processing_end.isoformat() if record.processing_end else None,
                    "data_categories": [cat.value for cat in record.data_categories],
                    "retention_period_days": record.retention_period
                })
        
        return {
            "user_id": self._hash_user_id(user_id),
            "data_subject_rights": {
                "right_to_access": "Available via this summary",
                "right_to_rectification": "Contact data protection officer",
                "right_to_erasure": "Use delete_user_data() method",
                "right_to_restrict_processing": "Use withdraw_consent() method",
                "right_to_data_portability": "Data export available on request",
                "right_to_object": "Use withdraw_consent() method"
            },
            "consents": consents,
            "processing_activities": processing,
            "data_retention_info": {
                purpose.value: f"{days} days" 
                for purpose, days in self.data_retention_policies.items()
            },
            "contact_dpo": "dpo@example.com",  # Should be configured
            "supervisory_authority": "Relevant data protection authority"
        }
    
    def delete_user_data(self, user_id: str, verify_request: bool = True) -> Dict[str, Any]:
        """Delete user data (Right to Erasure - GDPR Article 17).
        
        Args:
            user_id: User identifier
            verify_request: Whether to verify the deletion request
            
        Returns:
            Summary of deleted data
        """
        if verify_request:
            logger.warning(
                "User data deletion requested",
                extra={"user_id": self._hash_user_id(user_id)}
            )
        
        deleted_items = {
            "consent_records": 0,
            "processing_records": 0,
            "cached_data": 0
        }
        
        # Delete consent records
        if user_id in self.consent_records:
            deleted_items["consent_records"] = len(self.consent_records[user_id])
            del self.consent_records[user_id]
        
        # Mark processing records for deletion (retain for legal compliance)
        for record in self.processing_records:
            if record.user_id == user_id:
                record.user_id = f"DELETED_{self._hash_user_id(user_id)}"
                record.metadata["deleted_at"] = datetime.utcnow().isoformat()
                deleted_items["processing_records"] += 1
        
        logger.info(
            "User data deleted",
            extra={
                "user_id": self._hash_user_id(user_id),
                "deleted_items": deleted_items
            }
        )
        
        return deleted_items
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data based on retention policies.
        
        Returns:
            Summary of cleaned up data
        """
        now = datetime.utcnow()
        cleanup_summary = {
            "expired_consents": 0,
            "expired_processing_records": 0
        }
        
        # Clean up expired consents
        for user_id, consents in self.consent_records.items():
            expired_purposes = []
            for purpose, consent in consents.items():
                if consent.expires_at and now > consent.expires_at:
                    consent.status = ConsentStatus.EXPIRED
                    expired_purposes.append(purpose)
                    cleanup_summary["expired_consents"] += 1
        
        # Clean up old processing records
        retention_cutoff = {}
        for purpose, days in self.data_retention_policies.items():
            retention_cutoff[purpose] = now - timedelta(days=days)
        
        expired_records = []
        for i, record in enumerate(self.processing_records):
            cutoff = retention_cutoff.get(record.purpose)
            if cutoff and record.processing_start < cutoff:
                expired_records.append(i)
                cleanup_summary["expired_processing_records"] += 1
        
        # Remove expired records (in reverse order to maintain indices)
        for i in reversed(expired_records):
            del self.processing_records[i]
        
        logger.info("Data cleanup completed", extra=cleanup_summary)
        return cleanup_summary
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for auditing.
        
        Returns:
            Comprehensive compliance report
        """
        now = datetime.utcnow()
        
        # Consent statistics
        consent_stats = {
            "total_users": len(self.consent_records),
            "consents_by_purpose": {},
            "consents_by_status": {}
        }
        
        for user_consents in self.consent_records.values():
            for consent in user_consents.values():
                purpose = consent.purpose.value
                status = consent.status.value
                
                consent_stats["consents_by_purpose"][purpose] = \
                    consent_stats["consents_by_purpose"].get(purpose, 0) + 1
                consent_stats["consents_by_status"][status] = \
                    consent_stats["consents_by_status"].get(status, 0) + 1
        
        # Processing statistics
        processing_stats = {
            "total_processing_activities": len(self.processing_records),
            "active_processing": len([r for r in self.processing_records if r.processing_end is None]),
            "processing_by_purpose": {},
            "processing_by_legal_basis": {}
        }
        
        for record in self.processing_records:
            purpose = record.purpose.value
            legal_basis = record.legal_basis
            
            processing_stats["processing_by_purpose"][purpose] = \
                processing_stats["processing_by_purpose"].get(purpose, 0) + 1
            processing_stats["processing_by_legal_basis"][legal_basis] = \
                processing_stats["processing_by_legal_basis"].get(legal_basis, 0) + 1
        
        return {
            "report_generated_at": now.isoformat(),
            "compliance_framework": "GDPR/CCPA/PDPA",
            "consent_statistics": consent_stats,
            "processing_statistics": processing_stats,
            "data_retention_policies": {
                purpose.value: f"{days} days"
                for purpose, days in self.data_retention_policies.items()
            },
            "privacy_controls": {
                "anonymization_enabled": self.anonymization_enabled,
                "pseudonymization_enabled": self.pseudonymization_enabled,
                "encryption_enabled": self.encryption_enabled
            }
        }
    
    def _hash_user_id(self, user_id: Optional[str]) -> str:
        """Create hash of user ID for logging (privacy protection).
        
        Args:
            user_id: User identifier to hash
            
        Returns:
            Hashed user identifier
        """
        if not user_id:
            return "anonymous"
        
        return hashlib.sha256(user_id.encode()).hexdigest()[:12]


# Global compliance manager
compliance_manager = ComplianceManager()


def with_consent_check(purpose: DataProcessingPurpose):
    """Decorator to check user consent before processing.
    
    Args:
        purpose: Data processing purpose
    """
    def decorator(func):
        def wrapper(*args, user_id: Optional[str] = None, **kwargs):
            if user_id and not compliance_manager.check_consent(user_id, purpose):
                raise PermissionError(f"No valid consent for {purpose.value}")
            return func(*args, user_id=user_id, **kwargs)
        return wrapper
    return decorator


def with_processing_record(
    purpose: DataProcessingPurpose,
    data_categories: List[DataClassification],
    legal_basis: str = "legitimate_interest"
):
    """Decorator to automatically record data processing.
    
    Args:
        purpose: Processing purpose
        data_categories: Categories of data processed
        legal_basis: Legal basis for processing
    """
    def decorator(func):
        def wrapper(*args, user_id: Optional[str] = None, **kwargs):
            processing_id = compliance_manager.start_processing(
                purpose=purpose,
                data_categories=data_categories,
                user_id=user_id,
                legal_basis=legal_basis,
                function=func.__name__
            )
            
            try:
                result = func(*args, user_id=user_id, **kwargs)
                return result
            finally:
                compliance_manager.end_processing(processing_id)
                
        return wrapper
    return decorator