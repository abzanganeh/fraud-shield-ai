"""
Timezone Pipeline for Fraud Detection

Provides timezone resolution and UTC-to-local time conversion for both:
- Batch processing (Spark DataFrames)
- Single-record inference (Python datetime)

Design: Thin pipeline class with dependency injection.
No file I/O or Spark session creation - all dependencies injected.
"""

import logging
import time
from typing import Tuple, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
DEFAULT_GRID_SIZE = 0.5

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore[assignment]

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, floor, broadcast, when, from_utc_timestamp
)


class TimezonePipeline:
    """
    Pipeline for timezone resolution and UTC-to-local conversion (batch Spark operations).
    
    Holds configuration (zip_ref_df, grid_size, column names) but does NOT
    handle file I/O or Spark session creation. All dependencies injected.
    """
    
    def __init__(
        self,
        zip_ref_df: DataFrame,
        grid_size: float = DEFAULT_GRID_SIZE,
        utc_timestamp_col: str = "trans_date_trans_time",
        customer_lat_col: str = "lat",
        customer_lon_col: str = "long",
        merchant_lat_col: str = "merch_lat",
        merchant_lon_col: str = "merch_long"
    ):
        """
        Initialize pipeline with timezone reference data and configuration.
        
        Args:
            zip_ref_df: Spark DataFrame with (lat_grid, lng_grid, timezone) columns
            grid_size: Grid resolution in degrees (default: 0.5 ~50km)
            utc_timestamp_col: Name of UTC timestamp column
            customer_lat_col: Customer latitude column name
            customer_lon_col: Customer longitude column name
            merchant_lat_col: Merchant latitude column name
            merchant_lon_col: Merchant longitude column name
        """
        self.zip_ref_df = zip_ref_df
        self.grid_size = grid_size
        self.utc_timestamp_col = utc_timestamp_col
        self.customer_lat_col = customer_lat_col
        self.customer_lon_col = customer_lon_col
        self.merchant_lat_col = merchant_lat_col
        self.merchant_lon_col = merchant_lon_col
        self._verbose = True

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        logger.log(level, msg)
        if self._verbose:
            print(msg)

    def _resolve_timezone_with_grid(
        self,
        df: DataFrame,
        lat_col: str,
        lon_col: str,
        entity_name: str = "locations"
    ) -> Tuple[DataFrame, Dict]:
        """
        Resolve timezones using grid-based lookup with nearest neighbor fallback.
        
        Args:
            df: Source DataFrame
            lat_col: Latitude column name
            lon_col: Longitude column name
            entity_name: For logging (e.g., "customers", "merchants")
        
        Returns:
            tuple: (timezone_df, metrics_dict)
        """
        start_time = time.time()
        
        self._log("=" * 80)
        self._log(f"RESOLVING TIMEZONES: {entity_name.upper()}")
        self._log("=" * 80)
        
        # Direct grid match
        timezone_df = (
            df.select(lat_col, lon_col).distinct()
            .withColumn("lat_grid", floor(col(lat_col) / self.grid_size))
            .withColumn("lng_grid", floor(col(lon_col) / self.grid_size))
            .join(broadcast(self.zip_ref_df), on=["lat_grid", "lng_grid"], how="left")
            .select(lat_col, lon_col, col("timezone"))
            .cache()
        )
        
        total = timezone_df.count()
        if total == 0:
            self._log("WARNING: Empty distinct set (no locations to resolve).", logging.WARNING)
            return timezone_df, {
                "entity": entity_name,
                "total": 0,
                "direct_matches": 0,
                "fallback_matches": 0,
                "final_coverage": 0,
                "direct_rate": 0.0,
                "final_rate": 0.0,
                "elapsed_seconds": time.time() - start_time,
            }
        direct = timezone_df.filter(col("timezone").isNotNull()).count()
        direct_rate = (direct / total) * 100
        
        self._log(f"Direct matches: {direct:,} / {total:,} ({direct_rate:.2f}%)")
        
        # Nearest neighbor fallback for NULLs
        null_count = total - direct
        resolved_fallback = 0
        
        if null_count > 0:
            self._log(f"Applying nearest neighbor for {null_count:,} locations...")
            null_locs = timezone_df.filter(col("timezone").isNull()).select(lat_col, lon_col)
            
            neighbors_found = None
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_tz = (
                        null_locs
                        .withColumn("lat_grid", floor(col(lat_col) / self.grid_size) + dx)
                        .withColumn("lng_grid", floor(col(lon_col) / self.grid_size) + dy)
                        .join(broadcast(self.zip_ref_df), on=["lat_grid", "lng_grid"], how="inner")
                        .select(lat_col, lon_col, col("timezone"))
                        .dropDuplicates([lat_col, lon_col])
                    )
                    if neighbors_found is None:
                        neighbors_found = neighbor_tz
                    else:
                        neighbors_found = neighbors_found.union(neighbor_tz).dropDuplicates([lat_col, lon_col])
            
            if neighbors_found is not None:
                timezone_df = (
                    timezone_df
                    .join(neighbors_found.withColumnRenamed("timezone", "fallback_tz"),
                         on=[lat_col, lon_col], how="left")
                    .withColumn("timezone", when(col("timezone").isNull(), col("fallback_tz")).otherwise(col("timezone")))
                    .drop("fallback_tz")
                    .cache()
                )
                resolved_fallback = timezone_df.filter(col("timezone").isNotNull()).count() - direct
                self._log(f"✓ Resolved {resolved_fallback:,} via nearest neighbor")
        
        final_coverage = timezone_df.filter(col("timezone").isNotNull()).count()
        final_rate = (final_coverage / total) * 100 if total > 0 else 0.0
        elapsed = time.time() - start_time
        
        self._log(f"Final coverage: {final_coverage:,} / {total:,} ({final_rate:.2f}%)")
        self._log(f"Completed in {elapsed:.1f}s")
        self._log("=" * 80)
        
        return timezone_df, {
            "entity": entity_name,
            "total": total,
            "direct_matches": direct,
            "fallback_matches": resolved_fallback,
            "final_coverage": final_coverage,
            "direct_rate": direct_rate,
            "final_rate": final_rate,
            "elapsed_seconds": elapsed
        }
    
    def add_customer_timezone(self, df: DataFrame) -> DataFrame:
        """
        Add customer_timezone and customer_timezone_valid columns to DataFrame.
        
        Args:
            df: Input DataFrame with customer lat/lon columns
        
        Returns:
            DataFrame with customer_timezone and customer_timezone_valid columns
        """
        if "customer_timezone" in df.columns:
            self._log("⚠️ customer_timezone already exists, skipping computation", logging.WARNING)
            return df
        
        customer_tz_df, metrics = self._resolve_timezone_with_grid(
            df=df,
            lat_col=self.customer_lat_col,
            lon_col=self.customer_lon_col,
            entity_name="customers"
        )
        
        # Join to main DataFrame
        df = df.join(
            customer_tz_df.withColumnRenamed("timezone", "customer_timezone"),
            on=[self.customer_lat_col, self.customer_lon_col],
            how="left"
        )
        
        # Validation column
        df = df.withColumn(
            "customer_timezone_valid",
            col("customer_timezone").isNotNull()
        )
        
        # Cleanup
        customer_tz_df.unpersist()
        
        self._log(f"✓ Customer timezone column added")
        self._log(f"  Coverage: {metrics['final_rate']:.2f}% ({metrics['final_coverage']:,} / {metrics['total']:,})")
        
        return df
    
    def add_merchant_timezone(self, df: DataFrame) -> DataFrame:
        """
        Add merchant_timezone and merchant_timezone_valid columns to DataFrame.
        
        Args:
            df: Input DataFrame with merchant lat/lon columns
        
        Returns:
            DataFrame with merchant_timezone and merchant_timezone_valid columns
        """
        if "merchant_timezone" in df.columns:
            self._log("⚠️ merchant_timezone already exists, skipping computation", logging.WARNING)
            return df
        
        merchant_tz_df, metrics = self._resolve_timezone_with_grid(
            df=df,
            lat_col=self.merchant_lat_col,
            lon_col=self.merchant_lon_col,
            entity_name="merchants"
        )
        
        # Join to main DataFrame
        df = df.join(
            merchant_tz_df.withColumnRenamed("timezone", "merchant_timezone"),
            on=[self.merchant_lat_col, self.merchant_lon_col],
            how="left"
        )
        
        # Validation column
        df = df.withColumn(
            "merchant_timezone_valid",
            col("merchant_timezone").isNotNull()
        )
        
        # Cleanup
        merchant_tz_df.unpersist()
        
        self._log(f"✓ Merchant timezone column added")
        self._log(f"  Coverage: {metrics['final_rate']:.2f}% ({metrics['final_coverage']:,} / {metrics['total']:,})")
        
        return df
    
    def add_merchant_local_time(self, df: DataFrame) -> DataFrame:
        """
        Convert UTC timestamps to merchant local time.
        
        Requires merchant_timezone column to exist (call add_merchant_timezone first).
        
        Args:
            df: Input DataFrame with UTC timestamp and merchant_timezone columns
        
        Returns:
            DataFrame with merchant_local_time column
        """
        if "merchant_local_time" in df.columns:
            self._log("⚠️ merchant_local_time already exists, skipping", logging.WARNING)
            return df
        
        if "merchant_timezone" not in df.columns:
            raise ValueError("merchant_timezone column required. Call add_merchant_timezone() first.")
        
        # Production fallback (EDA parity):
        # If merchant_timezone is NULL, fall back to UTC timestamp (guaranteed to exist).
        df = df.withColumn(
            "merchant_local_time",
            when(
                col("merchant_timezone").isNull(),
                col(self.utc_timestamp_col),
            ).otherwise(from_utc_timestamp(col(self.utc_timestamp_col), col("merchant_timezone"))),
        )
        
        self._log("✓ Merchant local time created")
        self._log("\nSample conversions (UTC → Merchant Local):")
        df.select(
            self.utc_timestamp_col,
            "merchant_timezone",
            "merchant_local_time"
        ).show(5, truncate=False)
        
        return df
    
    def add_customer_local_time(self, df: DataFrame) -> DataFrame:
        """
        Convert UTC timestamps to customer local time.
        
        Requires customer_timezone column to exist (call add_customer_timezone first).
        
        Args:
            df: Input DataFrame with UTC timestamp and customer_timezone columns
        
        Returns:
            DataFrame with customer_local_time column
        """
        if "customer_local_time" in df.columns:
            self._log("⚠️ customer_local_time already exists, skipping", logging.WARNING)
            return df
        
        if "customer_timezone" not in df.columns:
            raise ValueError("customer_timezone column required. Call add_customer_timezone() first.")
        
        # Production fallback (EDA parity):
        # If customer_timezone is NULL, fall back to UTC timestamp (guaranteed to exist).
        df = df.withColumn(
            "customer_local_time",
            when(
                col("customer_timezone").isNull(),
                col(self.utc_timestamp_col),
            ).otherwise(from_utc_timestamp(col(self.utc_timestamp_col), col("customer_timezone"))),
        )
        
        self._log("✓ Customer local time created")
        self._log("\nSample conversions (UTC → Customer Local):")
        df.select(
            self.utc_timestamp_col,
            "customer_timezone",
            "customer_local_time"
        ).show(5, truncate=False)
        
        return df
    
    def apply_full_pipeline(
        self,
        df: DataFrame,
        include_customer: bool = True,
        include_merchant: bool = True
    ) -> DataFrame:
        """
        Apply full timezone pipeline: resolve timezones and convert to local time.
        
        Args:
            df: Input DataFrame
            include_customer: Whether to process customer timezone/local time
            include_merchant: Whether to process merchant timezone/local time
        
        Returns:
            DataFrame with timezone and local time columns added
        """
        if include_merchant:
            df = self.add_merchant_timezone(df)
            df = self.add_merchant_local_time(df)
        
        if include_customer:
            df = self.add_customer_timezone(df)
            df = self.add_customer_local_time(df)
        
        return df


class SingleRecordTimezoneConverter:
    """
    Single-record timezone converter for production inference (Python-only, no Spark).
    
    For converting one transaction at a time in production services.
    """
    
    def __init__(
        self,
        grid_ref: Dict[Tuple[int, int], str],
        grid_size: float = DEFAULT_GRID_SIZE
    ):
        """
        Initialize with pre-loaded grid reference data.
        
        Args:
            grid_ref: Dict mapping (lat_grid, lng_grid) -> timezone string
            grid_size: Grid resolution in degrees (must match batch pipeline)
        """
        self.grid_ref = grid_ref
        self.grid_size = grid_size
    
    def lookup_timezone(self, lat: float, lon: float) -> Optional[str]:
        """
        Look up timezone for given latitude/longitude.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Timezone string (e.g., "America/New_York") or None if not found
        """
        lat_grid = int(lat / self.grid_size)
        lng_grid = int(lon / self.grid_size)
        
        # Direct lookup
        timezone = self.grid_ref.get((lat_grid, lng_grid))
        if timezone:
            return timezone
        
        # Nearest neighbor fallback
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                timezone = self.grid_ref.get((lat_grid + dx, lng_grid + dy))
                if timezone:
                    return timezone
        
        return None
    
    def convert_utc_to_local(
        self,
        utc_ts: datetime,
        timezone: Optional[str]
    ) -> datetime:
        """
        Convert UTC datetime to local timezone.
        
        Args:
            utc_ts: UTC datetime object
            timezone: Timezone string (e.g., "America/New_York"). If None (e.g. lookup failed), UTC is used.
        
        Returns:
            Local datetime object
        """
        if timezone is None:
            timezone = "UTC"
        if ZoneInfo is not None:
            tz = ZoneInfo(timezone)
            utc_tz = ZoneInfo("UTC")
        else:  # pragma: no cover
            try:
                import pytz
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Timezone conversion requires either Python 3.9+ (zoneinfo) "
                    "or the 'pytz' package."
                ) from exc
            tz = pytz.timezone(timezone)
            utc_tz = pytz.UTC

        if utc_ts.tzinfo is None:
            utc_ts = utc_ts.replace(tzinfo=utc_tz)

        return utc_ts.astimezone(tz)
