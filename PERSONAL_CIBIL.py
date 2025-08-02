# =============================================================================
# Enhanced Production-Ready SME Credit Risk Assessment System (CUDA Enabled)
# =============================================================================
# Key Improvements:
# 1. GPU Acceleration: Added CUDA support for PyTorch, XGBoost, and LightGBM.
#    The system automatically detects CUDA and uses the GPU if available.
# 2. Comprehensive error handling and input validation.
# 3. Enhanced feature engineering with statistical validation.
# 4. Model interpretability (SHAP values).
# 5. Improved cross-validation with temporal awareness.
# 6. Comprehensive interactive dashboards.
# 7. Enhanced monitoring and drift detection.
# 8. Improved logging and debugging capabilities.
# =============================================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix, average_precision_score)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import warnings
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from dataclasses import dataclass, field
from pathlib import Path
import gc
import json
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import shap
from functools import wraps
import time
import traceback

# --- Enhanced Configuration ---
warnings.filterwarnings('ignore')

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_credit_risk_system_cuda.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def exception_handler(func):
    """Decorator for comprehensive exception handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@dataclass
class EnhancedModelConfig:
    """Enhanced configuration with validation and monitoring parameters."""
    # Data parameters
    n_samples: int = 10000
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1

    # Model parameters
    cv_folds: int = 5
    max_iter: int = 1000
    early_stopping_rounds: int = 50

    # Business parameters
    default_threshold: float = 0.5
    cost_fp: float = 1.0  # Cost of false positive (approving bad loan)
    cost_fn: float = 5.0  # Cost of false negative (rejecting good loan)
    profit_per_good_loan: float = 0.15  # Expected profit from good loans

    # Technical parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_jobs: int = -1

    # Regulatory and fairness parameters
    min_approval_rate: float = 0.3
    max_approval_rate: float = 0.8
    fairness_threshold: float = 0.05  # Maximum allowed difference in approval rates

    # Monitoring parameters
    drift_threshold: float = 0.05
    performance_threshold: float = 0.02

    # Feature engineering parameters
    outlier_threshold: float = 3.0
    correlation_threshold: float = 0.95

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.validation_size < 1 - self.test_size:
            raise ValueError("validation_size must be valid given test_size")
        if self.cost_fp <= 0 or self.cost_fn <= 0:
            raise ValueError("Costs must be positive")
        logger.info(f"Configuration validated successfully. Running on device: '{self.device.upper()}'")

class DataValidator:
    """Comprehensive data validation and quality checks."""
    @staticmethod
    @exception_handler
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive data validation."""
        validation_report = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_stats': {},
            'categorical_stats': {},
            'quality_score': 0.0
        }
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            validation_report['numeric_stats'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'outliers': len(df[col][np.abs(stats.zscore(df[col].fillna(0))) > 3]),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum()
            }
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            validation_report['categorical_stats'][col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_dist': df[col].value_counts().head().to_dict()
            }
        # Calculate quality score
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        duplicate_ratio = validation_report['duplicate_rows'] / df.shape[0]
        validation_report['quality_score'] = max(0, 1 - missing_ratio - duplicate_ratio)
        logger.info(f"Data validation completed. Quality score: {validation_report['quality_score']:.3f}")
        return validation_report

class EnhancedSMEDataGenerator:
    """Enhanced data generator with improved realism and validation."""
    def __init__(self, config: EnhancedModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        self.validator = DataValidator()
        logger.info(f"Initialized enhanced data generator for {config.n_samples} samples")

    @timing_decorator
    @exception_handler
    def _generate_temporal_features(self) -> pd.DataFrame:
        """Generate enhanced time-based features with economic cycles."""
        try:
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2024, 12, 31)
            date_range = (end_date - start_date).days
            dates = [start_date + timedelta(days=int(self.rng.integers(0, date_range))) for _ in range(self.config.n_samples)]
            df = pd.DataFrame({'application_date': dates, 'days_since_founding': np.maximum(30, self.rng.exponential(scale=800, size=self.config.n_samples))})
            # Enhanced temporal features
            df['quarter'] = pd.to_datetime(df['application_date']).dt.quarter
            df['month'] = pd.to_datetime(df['application_date']).dt.month
            df['day_of_week'] = pd.to_datetime(df['application_date']).dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            # Economic cycle with COVID impact
            def get_economic_cycle(date):
                if date.year == 2020: return 'covid_recession'
                elif date.year == 2021: return 'recovery'
                elif date.year in [2022, 2023]: return 'growth'
                else: return 'stable'
            df['economic_cycle'] = pd.to_datetime(df['application_date']).apply(get_economic_cycle)
            # Seasonal business patterns
            df['seasonal_multiplier'] = 1 + 0.2 * np.sin(2 * np.pi * df['month'] / 12)
            return df
        except Exception as e:
            logger.error(f"Error generating temporal features: {e}")
            raise

    @exception_handler
    def _generate_business_features(self) -> pd.DataFrame:
        """Generate enhanced business characteristics with validation."""
        industries = {
            'retail': {'risk': 0.15, 'seasonality': 0.3, 'covid_impact': 0.4},
            'manufacturing': {'risk': 0.12, 'seasonality': 0.1, 'covid_impact': 0.2},
            'services': {'risk': 0.10, 'seasonality': 0.2, 'covid_impact': 0.3},
            'technology': {'risk': 0.08, 'seasonality': 0.05, 'covid_impact': -0.1},
            'healthcare': {'risk': 0.09, 'seasonality': 0.1, 'covid_impact': -0.2},
            'construction': {'risk': 0.18, 'seasonality': 0.4, 'covid_impact': 0.3},
            'hospitality': {'risk': 0.22, 'seasonality': 0.5, 'covid_impact': 0.8},
            'agriculture': {'risk': 0.16, 'seasonality': 0.6, 'covid_impact': 0.1},
            'transportation': {'risk': 0.14, 'seasonality': 0.2, 'covid_impact': 0.5},
            'finance': {'risk': 0.11, 'seasonality': 0.1, 'covid_impact': 0.2}
        }
        locations = {
            'urban': {'risk': 0.10, 'competition': 0.8, 'accessibility': 0.9},
            'suburban': {'risk': 0.12, 'competition': 0.6, 'accessibility': 0.7},
            'rural': {'risk': 0.15, 'competition': 0.3, 'accessibility': 0.4}
        }
        business_sizes = ['micro', 'small', 'medium']
        selected_industries = self.rng.choice(list(industries.keys()), self.config.n_samples)
        selected_locations = self.rng.choice(list(locations.keys()), self.config.n_samples)
        df = pd.DataFrame({
            'industry': selected_industries,
            'location': selected_locations,
            'business_size': self.rng.choice(business_sizes, self.config.n_samples, p=[0.6, 0.3, 0.1]),
            'num_employees': np.maximum(1, self.rng.poisson(lam=8, size=self.config.n_samples)),
            'years_in_business': np.maximum(0.5, self.rng.exponential(scale=3, size=self.config.n_samples)),
            'legal_structure': self.rng.choice(['llc', 'corporation', 'partnership', 'sole_proprietorship'], self.config.n_samples, p=[0.4, 0.3, 0.2, 0.1])
        })
        # Add industry-specific metrics
        df['industry_risk_score'] = [industries[ind]['risk'] for ind in df['industry']]
        df['seasonality_factor'] = [industries[ind]['seasonality'] for ind in df['industry']]
        df['covid_impact_factor'] = [industries[ind]['covid_impact'] for ind in df['industry']]
        # Add location-specific metrics
        df['location_risk_score'] = [locations[loc]['risk'] for loc in df['location']]
        df['competition_level'] = [locations[loc]['competition'] for loc in df['location']]
        df['market_accessibility'] = [locations[loc]['accessibility'] for loc in df['location']]
        return df

    @exception_handler
    def _generate_financial_features(self, business_df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced financial metrics with cross-correlations."""
        size_multiplier = {'micro': 1, 'small': 3, 'medium': 8}
        industry_multiplier = {'retail': 1.2, 'manufacturing': 1.5, 'services': 0.8, 'technology': 2.0, 'healthcare': 1.3, 'construction': 1.1, 'hospitality': 0.9, 'agriculture': 0.7, 'transportation': 1.0, 'finance': 1.8}
        base_revenue = np.array([50000 * size_multiplier[row['business_size']] * industry_multiplier.get(row['industry'], 1.0) * (1 + row['years_in_business'] * 0.1) * row['market_accessibility'] for _, row in business_df.iterrows()])
        annual_revenue = self.rng.lognormal(mean=np.log(np.maximum(base_revenue, 10000)), sigma=0.5)
        profit_margin_base = self.rng.normal(0.08, 0.05, len(business_df))
        profit_margin = np.clip(profit_margin_base - business_df['industry_risk_score'] * 0.1 + business_df['years_in_business'] * 0.01, -0.1, 0.3)
        debt_to_equity_base = self.rng.exponential(scale=0.6, size=len(business_df))
        debt_to_equity_ratio = debt_to_equity_base * (1 + business_df['industry_risk_score'])
        return pd.DataFrame({
            'annual_revenue': annual_revenue, 'monthly_revenue': annual_revenue / 12, 'profit_margin': profit_margin, 'debt_to_equity_ratio': debt_to_equity_ratio,
            'current_ratio': np.clip(self.rng.normal(1.5, 0.5, len(business_df)), 0.5, 5.0), 'cash_flow_volatility': self.rng.exponential(scale=0.3, size=len(business_df)),
            'working_capital': annual_revenue * self.rng.normal(0.15, 0.08, len(business_df)), 'inventory_turnover': np.clip(self.rng.normal(6, 2, len(business_df)), 1, 20),
            'receivables_turnover': np.clip(self.rng.normal(8, 3, len(business_df)), 2, 30), 'gross_margin': np.clip(profit_margin + self.rng.normal(0.1, 0.05, len(business_df)), 0, 0.8),
            'operating_margin': np.clip(profit_margin - self.rng.normal(0.02, 0.01, len(business_df)), -0.1, 0.5)
        })

    @exception_handler
    def _generate_credit_history(self) -> pd.DataFrame:
        """Generate comprehensive credit and payment history."""
        credit_scores = np.clip(self.rng.normal(650, 80, self.config.n_samples), 300, 850).astype(int)
        late_payments = self.rng.poisson(lam=1.2 * (1 - (credit_scores - 300) / 550), size=self.config.n_samples)
        credit_utilization = np.clip(self.rng.beta(2, 3, self.config.n_samples) * (1 + (850 - credit_scores) / 550 * 0.5), 0, 1)
        return pd.DataFrame({
            'credit_score': credit_scores, 'num_credit_accounts': self.rng.poisson(lam=3, size=self.config.n_samples),
            'late_payments_12m': late_payments, 'late_payments_24m': late_payments + self.rng.poisson(lam=0.8, size=self.config.n_samples),
            'bankruptcy_history': self.rng.binomial(1, 0.05, self.config.n_samples), 'credit_utilization': credit_utilization,
            'payment_delay_days': self.rng.exponential(scale=15, size=self.config.n_samples), 'credit_inquiries_6m': self.rng.poisson(lam=2, size=self.config.n_samples),
            'longest_credit_history_months': np.maximum(12, self.rng.exponential(scale=60, size=self.config.n_samples))
        })

    @exception_handler
    def _generate_loan_features(self, financial_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive loan-specific features."""
        loan_amount_ratio = np.clip(self.rng.normal(0.3, 0.15, len(financial_df)), 0.05, 1.0)
        loan_amount = financial_df['annual_revenue'] * loan_amount_ratio
        loan_purposes = ['expansion', 'equipment', 'working_capital', 'refinancing', 'inventory', 'marketing', 'hiring']
        loan_terms = [12, 24, 36, 48, 60, 72, 84]
        return pd.DataFrame({
            'loan_amount': loan_amount, 'loan_purpose': self.rng.choice(loan_purposes, len(financial_df)),
            'loan_term_months': self.rng.choice(loan_terms, len(financial_df), p=[0.1, 0.15, 0.25, 0.2, 0.15, 0.1, 0.05]),
            'collateral_value': loan_amount * self.rng.uniform(0.5, 2.0, len(financial_df)), 'guarantor_present': self.rng.binomial(1, 0.3, len(financial_df)),
            'down_payment_ratio': self.rng.uniform(0, 0.3, len(financial_df)), 'interest_rate_quoted': self.rng.normal(8.5, 2.0, len(financial_df)),
            'requested_vs_offered': self.rng.uniform(0.8, 1.2, len(financial_df))
        })

    @exception_handler
    def _calculate_default_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced default probability calculation with more sophisticated logic."""
        credit_risk = 1 - (df['credit_score'] - 300) / 550
        financial_risk = (np.clip(df['debt_to_equity_ratio'] / 3, 0, 1) * 0.25 + np.clip((2 - df['current_ratio']) / 2, 0, 1) * 0.15 + np.clip(df['cash_flow_volatility'], 0, 1) * 0.2 + np.clip(-df['profit_margin'] * 5, 0, 1) * 0.25 + np.clip(-df['operating_margin'] * 5, 0, 1) * 0.15)
        business_risk = (df['industry_risk_score'] * 0.3 + df['location_risk_score'] * 0.2 + np.clip(1 / np.maximum(df['years_in_business'], 0.5) * 2, 0, 1) * 0.3 + (1 - df['market_accessibility']) * 0.2)
        economic_risk = df['economic_cycle'].map({'covid_recession': 0.4, 'recovery': 0.2, 'growth': 0.05, 'stable': 0.1}) + df['covid_impact_factor'] * 0.1
        payment_risk = (np.clip(df['late_payments_12m'] / 10, 0, 1) * 0.3 + np.clip(df['late_payments_24m'] / 20, 0, 1) * 0.2 + df['bankruptcy_history'] * 0.3 + df['credit_utilization'] * 0.2)
        loan_risk = (np.clip(df['loan_amount'] / df['annual_revenue'] - 0.2, 0, 0.8) / 0.8 * 0.4 + np.clip((df['loan_term_months'] - 36) / 48, 0, 1) * 0.2 + (1 - df['guarantor_present']) * 0.2 + np.clip((1 - df['down_payment_ratio']) * 1.5, 0, 1) * 0.2)
        total_risk = (credit_risk * 0.20 + financial_risk * 0.25 + business_risk * 0.20 + economic_risk * 0.10 + payment_risk * 0.15 + loan_risk * 0.10)
        base_prob = 1 / (1 + np.exp(-5 * (total_risk - 0.5)))
        final_prob = np.clip(base_prob + self.rng.normal(0, 0.03, len(base_prob)), 0.01, 0.99)
        return final_prob

    @timing_decorator
    @exception_handler
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete enhanced dataset with validation."""
        logger.info("Generating enhanced SME credit dataset...")
        temporal_df = self._generate_temporal_features()
        business_df = self._generate_business_features()
        financial_df = self._generate_financial_features(business_df)
        credit_df = self._generate_credit_history()
        loan_df = self._generate_loan_features(financial_df)
        combined_df = pd.concat([pd.DataFrame({'sme_id': [f"SME_{1000+i:05d}" for i in range(self.config.n_samples)]}), temporal_df, business_df, financial_df, credit_df, loan_df], axis=1)
        default_prob = self._calculate_default_probability(combined_df)
        combined_df['default_probability'] = default_prob
        combined_df['credit_default'] = self.rng.binomial(1, default_prob)
        validation_report = self.validator.validate_dataframe(combined_df)
        logger.info(f"Generated dataset: {len(combined_df)} samples, default rate: {combined_df['credit_default'].mean():.3f}, quality score: {validation_report['quality_score']:.3f}")
        return combined_df

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Enhanced feature engineering with statistical validation and monitoring."""
    def __init__(self, config: EnhancedModelConfig):
        self.config = config; self.scalers = {}; self.encoders = {}; self.feature_names_out_ = []; self.feature_importance_ = {}; self.outlier_detector = None; self.feature_stats_ = {}; self.correlation_matrix_ = None
    @exception_handler
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using multiple methods."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns; outlier_indices = set()
        for col in numeric_cols: z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median()))); outlier_indices.update(df.index[z_scores > self.config.outlier_threshold])
        if len(numeric_cols) > 1:
            iso_forest = IsolationForest(contamination=0.1, random_state=self.config.random_state)
            outlier_pred = iso_forest.fit_predict(df[numeric_cols].fillna(df[numeric_cols].median())); multivariate_outliers = df.index[outlier_pred == -1]; outlier_indices.update(multivariate_outliers)
        logger.info(f"Detected {len(outlier_indices)} outliers ({len(outlier_indices)/len(df)*100:.2f}%)"); return df.drop(index=list(outlier_indices))
    @exception_handler
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive engineered features."""
        df = df.copy()
        df['revenue_per_employee'] = df['annual_revenue'] / np.maximum(df['num_employees'], 1); df['loan_to_revenue_ratio'] = df['loan_amount'] / np.maximum(df['annual_revenue'], 1e3); df['debt_service_coverage'] = (df['annual_revenue'] * df['profit_margin']) / np.maximum(df['loan_amount'] * 0.12, 1e2); df['working_capital_ratio'] = df['working_capital'] / np.maximum(df['annual_revenue'], 1e3); df['asset_turnover'] = df['annual_revenue'] / np.maximum(df['collateral_value'], 1e3); df['volatility_adjusted_revenue'] = df['monthly_revenue'] / np.maximum(df['cash_flow_volatility'], 1e-2)
        df['inventory_efficiency'] = df['inventory_turnover'] / df['industry_risk_score']; df['receivables_efficiency'] = df['receivables_turnover'] * df['current_ratio']; df['margin_stability'] = df['gross_margin'] - df['operating_margin']
        df['payment_behavior_score'] = ((850 - df['credit_score']) / 550 * 0.3 + np.clip(df['late_payments_12m'] / 12, 0, 1) * 0.25 + np.clip(df['late_payments_24m'] / 24, 0, 1) * 0.15 + df['credit_utilization'] * 0.2 + df['bankruptcy_history'] * 0.1)
        df['credit_maturity_score'] = (df['longest_credit_history_months'] / 120 * 0.4 + np.clip(df['num_credit_accounts'] / 10, 0, 1) * 0.3 + (1 - np.clip(df['credit_inquiries_6m'] / 10, 0, 1)) * 0.3)
        df['business_stability_score'] = (np.clip(df['years_in_business'] / 10, 0, 1) * 0.3 + (1 - np.clip(df['cash_flow_volatility'], 0, 1)) * 0.25 + np.clip(df['current_ratio'] / 3, 0, 1) * 0.25 + df['market_accessibility'] * 0.2)
        df['growth_potential_score'] = ((1 - df['industry_risk_score']) * 0.3 + df['market_accessibility'] * 0.3 + np.clip(df['profit_margin'] * 5, 0, 1) * 0.4)
        if 'application_date' in df.columns:
            df['application_date'] = pd.to_datetime(df['application_date']); df['days_since_founding_cat'] = pd.cut(df['days_since_founding'], bins=[0, 365, 1095, 3650, np.inf], labels=['new', 'young', 'established', 'mature']); df['month_sin'] = np.sin(2 * np.pi * df['application_date'].dt.month / 12); df['month_cos'] = np.cos(2 * np.pi * df['application_date'].dt.month / 12)
        df = df.replace([np.inf, -np.inf], np.nan); logger.info(f"Created {len(df.columns)} features after engineering."); return df
    def fit(self, X: pd.DataFrame, y=None):
        """Fit scalers, encoders and feature selection logic."""
        X_eng = self._create_advanced_features(X.copy()); self.numerical_cols = X_eng.select_dtypes(include=np.number).columns.tolist(); self.categorical_cols = X_eng.select_dtypes(include=['object', 'category']).columns.tolist()
        self.scalers['robust'] = RobustScaler().fit(X_eng[self.numerical_cols].fillna(X_eng[self.numerical_cols].median()))
        for col in self.categorical_cols: self.encoders[col] = LabelEncoder().fit(X_eng[col].astype(str).unique().tolist() + ['__unseen__'])
        X_processed = self._apply_transformations(X_eng); corr_matrix = X_processed.corr().abs(); upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)); self.to_drop = [column for column in upper.columns if any(upper[column] > self.config.correlation_threshold)]
        logger.info(f"Dropping {len(self.to_drop)} features due to high correlation: {self.to_drop}"); self.feature_names_out_ = [col for col in X_processed.columns if col not in self.to_drop]; return self
    def _apply_transformations(self, X: pd.DataFrame) -> pd.DataFrame:
        """Helper to apply scaling and encoding."""
        X_copy = X.copy(); X_copy[self.numerical_cols] = self.scalers['robust'].transform(X_copy[self.numerical_cols].fillna(X_copy[self.numerical_cols].median()))
        for col in self.categorical_cols: unseen_mask = ~X_copy[col].astype(str).isin(self.encoders[col].classes_); X_copy.loc[unseen_mask, col] = '__unseen__'; X_copy[col] = self.encoders[col].transform(X_copy[col].astype(str))
        return X_copy
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering and selection steps."""
        X_eng = self._create_advanced_features(X.copy()); X_processed = self._apply_transformations(X_eng)
        for col in self.feature_names_out_:
            if col not in X_processed.columns: X_processed[col] = 0
        return X_processed[self.feature_names_out_]
    def get_feature_names_out(self, input_features=None): return self.feature_names_out_

class CreditRiskDNN(nn.Module):
    """Advanced Deep Neural Network for credit risk assessment with BatchNorm."""
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], dropout_rate: float = 0.4):
        super().__init__(); layers = []
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]); input_size = hidden_size
        layers.extend([nn.Linear(input_size, 1), nn.Sigmoid()]); self.network = nn.Sequential(*layers)
        logger.info(f"Initialized CreditRiskDNN with layers: {hidden_sizes} and dropout: {dropout_rate}")
    def forward(self, x): return self.network(x)

class ProductionModelTrainer:
    """Production-ready model training with temporal validation and business logic."""
    def __init__(self, config: EnhancedModelConfig):
        self.config = config; self.models = {}; self.feature_engineer = AdvancedFeatureEngineer(config); self.metrics = {}; self.best_model_name = None; self.best_model = None; self.raw_feature_names = []
    @timing_decorator
    @exception_handler
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data with proper temporal splitting and feature engineering."""
        logger.info("Preparing data with temporal splitting..."); df_sorted = df.sort_values('application_date').reset_index(drop=True)
        self.raw_feature_names = [col for col in df.columns if col not in ['sme_id', 'credit_default', 'default_probability', 'application_date']]; X = df_sorted[self.raw_feature_names]; y = df_sorted['credit_default']
        split_idx = int(len(df_sorted) * (1 - self.config.test_size)); X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]; y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        self.feature_engineer.fit(X_train, y_train); X_train_processed = self.feature_engineer.transform(X_train); X_test_processed = self.feature_engineer.transform(X_test)
        logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples"); logger.info(f"Processed feature count: {X_train_processed.shape[1]}"); return X_train_processed, X_test_processed, y_train, y_test
    @timing_decorator
    @exception_handler
    def train_baseline_models(self, X_train, y_train):
        """Train multiple baseline models with hyperparameter tuning using temporal CV and GPU acceleration."""
        logger.info("Training baseline models with TimeSeriesSplit CV..."); tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        lgbm_gpu_params = {}; xgb_gpu_params = {}
        if self.config.device == 'cuda':
            logger.info("CUDA detected. Configuring LightGBM and XGBoost for GPU training."); lgbm_gpu_params['device'] = 'gpu'; xgb_gpu_params['tree_method'] = 'gpu_hist'
        models_to_train = {
            'logistic_regression': (LogisticRegression(random_state=self.config.random_state, max_iter=self.config.max_iter, class_weight='balanced'), {'C': [0.1, 1.0], 'penalty': ['l2']}),
            'random_forest': (RandomForestClassifier(random_state=self.config.random_state, class_weight='balanced'), {'n_estimators': [100], 'max_depth': [10]}),
            'xgboost': (xgb.XGBClassifier(random_state=self.config.random_state, eval_metric='logloss', use_label_encoder=False, **xgb_gpu_params), {'n_estimators': [100], 'max_depth': [5], 'learning_rate': [0.1]}),
            'lightgbm': (LGBMClassifier(random_state=self.config.random_state, verbose=-1,), {'n_estimators': [100], 'max_depth': [5], 'learning_rate': [0.1]})
        }
        for name, (model, params) in models_to_train.items():
            grid_n_jobs = 1 if (name in ['lightgbm', 'xgboost'] and self.config.device == 'cuda') else self.config.n_jobs
            logger.info(f"Training {name}... (n_jobs={grid_n_jobs})"); grid = GridSearchCV(model, params, cv=tscv, scoring='roc_auc', n_jobs=grid_n_jobs)
            try: grid.fit(X_train, y_train); self.models[name] = grid.best_estimator_; logger.info(f"Best params for {name}: {grid.best_params_}")
            except Exception as e:
                logger.error(f"Could not train {name}. Error: {e}. Check library installations (especially for GPU).")
                if "cuML" in str(e) or "GPU" in str(e): logger.error(f"Hint: This might be due to an incorrect installation of the GPU-enabled version of {name}.")
                continue
    @timing_decorator
    @exception_handler
    def train_deep_model(self, X_train, y_train):
        """Train deep neural network with early stopping, using configured device."""
        logger.info(f"Training deep neural network on device: '{self.config.device.upper()}'"); X_t, X_v, y_t, y_v = self._create_validation_set(X_train, y_train)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_t.values), torch.FloatTensor(y_t.values).unsqueeze(1)), batch_size=256, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_v.values), torch.FloatTensor(y_v.values).unsqueeze(1)), batch_size=512)
        model = CreditRiskDNN(input_size=X_train.shape[1]).to(self.config.device); criterion = nn.BCELoss(); optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        best_val_loss = float('inf'); epochs_no_improve = 0
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                optimizer.zero_grad(); outputs = model(batch_X); loss = criterion(outputs, batch_y); loss.backward(); optimizer.step()
            model.eval(); val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.config.device), batch_y.to(self.config.device)
                    outputs = model(batch_X); val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)
            if (epoch + 1) % 10 == 0: logger.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss: best_val_loss = val_loss; epochs_no_improve = 0; torch.save(model.state_dict(), 'best_dnn_model.pth')
            else: epochs_no_improve += 1
            if epochs_no_improve >= self.config.early_stopping_rounds: logger.info(f"Early stopping at epoch {epoch+1}"); break
        model.load_state_dict(torch.load('best_dnn_model.pth')); self.models['deep_neural_network'] = model
    def _create_validation_set(self, X_train, y_train):
        val_split_idx = int(len(X_train) * (1 - self.config.validation_size)); return X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:], y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    @timing_decorator
    @exception_handler
    def evaluate_models(self, X_test, y_test) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation with business metrics."""
        logger.info("Evaluating all trained models...")
        for name, model in self.models.items():
            if name == 'deep_neural_network':
                model.eval()
                with torch.no_grad(): y_pred_proba = model(torch.FloatTensor(X_test.values).to(self.config.device)).cpu().numpy().flatten()
            else: y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > self.config.default_threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            business_cost = fp * self.config.cost_fp + fn * self.config.cost_fn; profit = (tp * self.config.profit_per_good_loan) - business_cost
            self.metrics[name] = {'roc_auc': roc_auc_score(y_test, y_pred_proba), 'avg_precision': average_precision_score(y_test, y_pred_proba), 'precision': precision_score(y_test, y_pred, zero_division=0), 'recall': recall_score(y_test, y_pred, zero_division=0), 'f1_score': f1_score(y_test, y_pred, zero_division=0), 'business_cost': business_cost, 'profit': profit, 'approval_rate': (tp + fp) / len(y_test) if len(y_test) > 0 else 0, 'y_pred_proba': y_pred_proba}
        if not self.metrics: logger.critical("No models were successfully trained or evaluated. Exiting."); return {}
        self.best_model_name = min(self.metrics.keys(), key=lambda x: self.metrics[x]['business_cost']); self.best_model = self.models[self.best_model_name]
        logger.info(f"Best model selected: {self.best_model_name} with business cost: {self.metrics[self.best_model_name]['business_cost']:.2f}"); return self.metrics

class ModelInterpreter:
    """Handles model interpretation using SHAP."""
    def __init__(self, model: Any, X_train: pd.DataFrame, model_name: str, config: EnhancedModelConfig):
        self.model = model; self.X_train = X_train; self.model_name = model_name; self.config = config; self.explainer = None; self.shap_values = None
        logger.info(f"Initializing SHAP explainer for model: {self.model_name}")
    @timing_decorator
    @exception_handler
    def explain(self):
        """Calculates SHAP values for the provided model and data, using the correct explainer."""
        # Use isinstance for robust type checking
        if isinstance(self.model, (LGBMClassifier, xgb.XGBClassifier, RandomForestClassifier)):
            logger.info("Using shap.TreeExplainer for tree-based model.")
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_train, check_additivity=False)
        elif isinstance(self.model, LogisticRegression):
            logger.info("Using shap.LinearExplainer for linear model.")
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
            self.shap_values = self.explainer.shap_values(self.X_train)
        elif isinstance(self.model, nn.Module):
            logger.info("Using shap.DeepExplainer for DNN model.")
            background = torch.FloatTensor(self.X_train.sample(100, random_state=self.config.random_state).values).to(self.config.device)
            to_explain = torch.FloatTensor(self.X_train.values).to(self.config.device)
            self.explainer = shap.DeepExplainer(self.model, background)
            self.shap_values = self.explainer.shap_values(to_explain)
        else:
            raise TypeError(f"Model type for '{self.model_name}' not supported by this SHAP interpreter.")
            
        # For classification, shap_values can be a list [class_0_values, class_1_values]
        # We are interested in the explanation for the positive class (default)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]

class InteractiveDashboardEngine:
    """Generates comprehensive interactive Plotly dashboards."""
    def __init__(self, trainer: 'ProductionModelTrainer'): self.trainer = trainer; self.metrics = trainer.metrics
    # In the class InteractiveDashboardEngine:
    # REPLACE the entire create_performance_dashboard method with this:

    @timing_decorator
    def create_performance_dashboard(self, X_test: pd.DataFrame, y_test: pd.Series, path: str = "performance_dashboard.html"):
        if not self.metrics:
            logger.warning("No metrics available to generate performance dashboard.")
            return

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'xy', 'colspan': 2}, None]],
            subplot_titles=('Key Performance Metrics (Test Set)', 'Business Impact (Test Set)', 'Receiver Operating Characteristic (ROC) Curves')
        )

        model_names = list(self.metrics.keys())
        roc_aucs = [m['roc_auc'] for m in self.metrics.values()]
        f1_scores = [m['f1_score'] for m in self.metrics.values()]
        business_costs = [m['business_cost'] for m in self.metrics.values()]
        profits = [m['profit'] for m in self.metrics.values()]

        # FIX: Text position is now set PER-TRACE on the bar charts only.
        fig.add_trace(go.Bar(name='ROC AUC', x=model_names, y=roc_aucs, text=[f'{v:.3f}' for v in roc_aucs], textposition='auto'), row=1, col=1)
        fig.add_trace(go.Bar(name='F1-Score', x=model_names, y=f1_scores, text=[f'{v:.3f}' for v in f1_scores], textposition='auto'), row=1, col=1)

        fig.add_trace(go.Bar(name='Business Cost', x=model_names, y=business_costs, text=[f'${v:,.0f}' for v in business_costs], textposition='auto'), row=1, col=2)
        fig.add_trace(go.Bar(name='Profit', x=model_names, y=profits, text=[f'${v:,.0f}' for v in profits], textposition='auto'), row=1, col=2)

        # These scatter plots do not have text labels, so no textposition is needed.
        for name, metric in self.metrics.items():
            fpr, tpr, _ = roc_curve(y_test, metric['y_pred_proba'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC {name} (AUC={metric["roc_auc"]:.3f})', mode='lines'), row=2, col=1)

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey'), name='Random Chance'), row=2, col=1)

        # The generic update_traces call that caused the error is GONE.
        fig.update_layout(title_text="<b>Model Performance and Business Impact Dashboard</b>", height=800, showlegend=True, legend_traceorder="reversed")
        fig.write_html(path)
        logger.info(f"Performance dashboard saved to {path}")
        
    @timing_decorator
    def create_interpretability_dashboard(self, interpreter: ModelInterpreter, path: str = "interpretability_dashboard.html"):
        """Creates a dashboard for model interpretability with SHAP plots."""
        if interpreter.shap_values is None: logger.warning("SHAP values not calculated. Skipping interpretability dashboard."); return
        shap_df = pd.DataFrame(interpreter.shap_values, columns=interpreter.X_train.columns); feature_importance = shap_df.abs().mean().sort_values(ascending=False).reset_index(); feature_importance.columns = ['Feature', 'Mean Absolute SHAP Value']
        fig = px.bar(feature_importance.head(20).sort_values(by='Mean Absolute SHAP Value', ascending=True), x='Mean Absolute SHAP Value', y='Feature', orientation='h', title=f"<b>Top 20 Feature Importances (SHAP) for '{interpreter.model_name}'</b>")
        fig.update_layout(height=800); fig.write_html(path); logger.info(f"Interpretability dashboard saved to {path}")

class ModelMonitor:
    """Monitors data drift and model performance degradation."""
    def __init__(self, training_data: pd.DataFrame, config: EnhancedModelConfig):
        self.training_data = training_data.copy(); self.config = config; self.numerical_cols = self.training_data.select_dtypes(include=np.number).columns; self.categorical_cols = self.training_data.select_dtypes(include=['object', 'category']).columns
    @timing_decorator
    def check_data_drift(self, new_data: pd.DataFrame) -> Dict:
        """Compares new data distribution to training data to detect drift."""
        drift_report = {'drift_detected': False, 'details': {}}; logger.info("Checking for data drift...")
        for col in self.numerical_cols:
            if col in new_data.columns:
                stat, p_val = ks_2samp(self.training_data[col].dropna(), new_data[col].dropna())
                if p_val < self.config.drift_threshold: drift_report['drift_detected'] = True; drift_report['details'][col] = {'type': 'numerical', 'p_value': p_val, 'test': 'KS'}
        for col in self.categorical_cols:
            if col in new_data.columns and self.training_data[col].nunique() > 1 and new_data[col].nunique() > 1:
                try:
                    contingency_table = pd.crosstab(self.training_data[col], new_data[col]); _, p_val, _, _ = chi2_contingency(contingency_table)
                    if p_val < self.config.drift_threshold: drift_report['drift_detected'] = True; drift_report['details'][col] = {'type': 'categorical', 'p_value': p_val, 'test': 'Chi-Squared'}
                except ValueError as e: logger.warning(f"Could not perform Chi-Squared test for column '{col}': {e}")
        if drift_report['drift_detected']: logger.warning(f"🚨 Data drift detected! Retraining may be necessary. Details: {drift_report['details']}")
        else: logger.info("✅ No significant data drift detected.")
        return drift_report

class ProductionDeploymentManager:
    """Handles model deployment, persistence, and prediction interface."""
    def __init__(self, config: EnhancedModelConfig): self.config = config; self.model_path = Path("production_models"); self.model_path.mkdir(exist_ok=True)
    @exception_handler
    def save_production_model(self, trainer: 'ProductionModelTrainer') -> str:
        """Saves the complete model pipeline (model, feature engineer, config)."""
        model_name = trainer.best_model_name; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); model_filename = f"credit_risk_model_{model_name}_{timestamp}.pkl"; model_filepath = self.model_path / model_filename
        model_package = {'model': trainer.best_model, 'model_name': model_name, 'feature_engineer': trainer.feature_engineer, 'config': self.config, 'training_metrics': trainer.metrics[model_name]}
        joblib.dump(model_package, model_filepath); logger.info(f"📦 Best model '{model_name}' and pipeline saved to: {model_filepath}"); return str(model_filepath)
    @exception_handler
    def load_production_model(self, model_filepath: str) -> Dict:
        """Loads a production model package."""
        logger.info(f"Loading model from: {model_filepath}"); return joblib.load(model_filepath)
    @exception_handler
    def predict_credit_risk(self, model_package: Dict, application_data: pd.DataFrame) -> Dict:
        """Makes credit risk predictions on new, raw application data."""
        model = model_package['model']; feature_engineer = model_package['feature_engineer']; config = model_package['config']; X_processed = feature_engineer.transform(application_data)
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad(): risk_scores = model(torch.FloatTensor(X_processed.values).to(config.device)).cpu().numpy().flatten()
        else: risk_scores = model.predict_proba(X_processed)[:, 1]
        decisions = []
        for score in risk_scores:
            if score > config.default_threshold: decision, risk_level = 'REJECT', 'HIGH'
            elif score > 0.3: decision, risk_level = 'REVIEW', 'MEDIUM'
            else: decision, risk_level = 'APPROVE', 'LOW'
            decisions.append({'risk_score': float(f'{score:.4f}'), 'decision': decision, 'risk_level': risk_level})
        return {'predictions': decisions}

def main():
    """Main execution function for the complete credit risk system."""
    logger.info("====== Starting Enhanced SME Credit Risk Assessment System ======")
    config = EnhancedModelConfig()
    # --- Data Generation & Validation ---
    data_gen = EnhancedSMEDataGenerator(config); df = data_gen.generate_dataset(); DataValidator.validate_dataframe(df)
    # --- Model Training and Evaluation ---
    trainer = ProductionModelTrainer(config); X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    trainer.train_baseline_models(X_train, y_train); trainer.train_deep_model(X_train, y_train); metrics = trainer.evaluate_models(X_test, y_test)
    # Exit if no models were trained
    if not trainer.best_model: logger.critical("No model was successfully trained. Aborting."); return
    # --- Model Interpretability ---
    interpreter = ModelInterpreter(trainer.best_model, X_train, trainer.best_model_name, config); interpreter.explain()
    # --- Visualization & Dashboards ---
    dashboard_engine = InteractiveDashboardEngine(trainer); dashboard_engine.create_performance_dashboard(X_test, y_test); dashboard_engine.create_interpretability_dashboard(interpreter)
    # --- Deployment ---
    deployment_manager = ProductionDeploymentManager(config); model_path = deployment_manager.save_production_model(trainer)
    # --- Prediction & Monitoring Demonstration ---
    loaded_model_package = deployment_manager.load_production_model(model_path)
    sample_applications = df[trainer.raw_feature_names].sample(5, random_state=config.random_state + 1)
    predictions = deployment_manager.predict_credit_risk(loaded_model_package, sample_applications)
    logger.info("="*25 + " SAMPLE PREDICTIONS " + "="*25)
    for i, pred in enumerate(predictions['predictions']): logger.info(f"Application {i+1}: Risk Score={pred['risk_score']:.4f}, Decision={pred['decision']}, Risk Level={pred['risk_level']}")
    logger.info("="*70)
    # Demonstrate monitoring
    monitor = ModelMonitor(df[trainer.raw_feature_names], config)
    new_data_sample = df[trainer.raw_feature_names].sample(1000, random_state=100)
    new_data_sample['annual_revenue'] *= 1.5; new_data_sample['credit_score'] -= 20
    drift_report = monitor.check_data_drift(new_data_sample)
    logger.info("====== Credit Risk Assessment System Completed Successfully! 🎉 ======")
    return metrics

if __name__ == "__main__":
    final_results = main()
    if final_results:
        logger.info("\n" + "="*20 + " Final Model Metrics Summary " + "="*20)
        results_df = pd.DataFrame(final_results).drop(index=['y_pred_proba'])
        logger.info("\n" + results_df.to_string())
        logger.info("="*65)
