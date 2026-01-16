//! LOTSA (Large-scale Open Time Series Archive) data loader.
//!
//! Loads time series data from the Salesforce/lotsa_data dataset on Hugging Face.
//! Supports multiple dataset configurations (frequencies).

use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use crate::data::math_generator::MathProblem;

#[derive(Debug, Deserialize)]
struct HfRow {
    row: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct HfResponse {
    rows: Vec<HfRow>,
}

/// A single time series sample for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSample {
    /// Historical context values
    pub context: Vec<f32>,
    /// Target future values
    pub target: Vec<f32>,
    /// Dataset/config name
    pub dataset: String,
    /// Frequency string (e.g., "30min", "1h", "1d")
    pub freq: String,
}

/// Preprocessing methods for normalizing time series data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreprocessingMethod {
    /// No preprocessing (raw values)
    None,
    /// Z-score normalization using mean and std
    ZScore,
    /// Robust scaling using median and IQR (more resistant to outliers)
    #[default]
    RobustScale,
    /// Log transform then Z-score (for positive, heavy-tailed data)
    LogZScore,
    /// Min-max scaling to [0, 1]
    MinMax,
    /// Box-Cox like transform: sign(x) * log(1 + |x|)
    SignedLog,
}

impl TimeSeriesSample {
    /// Apply preprocessing to normalize the sample.
    /// Returns a new sample with normalized values.
    pub fn preprocess(&self, method: PreprocessingMethod) -> Self {
        match method {
            PreprocessingMethod::None => self.clone(),
            PreprocessingMethod::ZScore => self.zscore_normalize(),
            PreprocessingMethod::RobustScale => self.robust_scale(),
            PreprocessingMethod::LogZScore => self.log_zscore(),
            PreprocessingMethod::MinMax => self.minmax_scale(),
            PreprocessingMethod::SignedLog => self.signed_log(),
        }
    }

    /// Z-score normalization: (x - mean) / std
    fn zscore_normalize(&self) -> Self {
        let all_values: Vec<f32> = self.context.iter()
            .chain(self.target.iter())
            .copied()
            .collect();

        let n = all_values.len() as f32;
        if n == 0.0 {
            return self.clone();
        }

        let mean: f32 = all_values.iter().sum::<f32>() / n;
        let variance: f32 = all_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / n;
        let std = variance.sqrt().max(1e-8);

        Self {
            context: self.context.iter().map(|x| (x - mean) / std).collect(),
            target: self.target.iter().map(|x| (x - mean) / std).collect(),
            dataset: self.dataset.clone(),
            freq: self.freq.clone(),
        }
    }

    /// Robust scaling: (x - median) / IQR
    fn robust_scale(&self) -> Self {
        let mut all_values: Vec<f32> = self.context.iter()
            .chain(self.target.iter())
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if all_values.is_empty() {
            return self.clone();
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = all_values.len();
        let median = if n % 2 == 0 {
            (all_values[n/2 - 1] + all_values[n/2]) / 2.0
        } else {
            all_values[n/2]
        };

        // IQR: Q3 - Q1
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let q1 = all_values[q1_idx];
        let q3 = all_values[q3_idx.min(n - 1)];
        let iqr = (q3 - q1).max(1e-8);

        Self {
            context: self.context.iter()
                .map(|x| if x.is_finite() { (x - median) / iqr } else { 0.0 })
                .collect(),
            target: self.target.iter()
                .map(|x| if x.is_finite() { (x - median) / iqr } else { 0.0 })
                .collect(),
            dataset: self.dataset.clone(),
            freq: self.freq.clone(),
        }
    }

    /// Log transform then Z-score (for positive heavy-tailed data)
    fn log_zscore(&self) -> Self {
        // First apply log1p transform
        let log_context: Vec<f32> = self.context.iter()
            .map(|&x| if x > 0.0 { (1.0 + x).ln() } else { -(1.0 + x.abs()).ln() })
            .collect();
        let log_target: Vec<f32> = self.target.iter()
            .map(|&x| if x > 0.0 { (1.0 + x).ln() } else { -(1.0 + x.abs()).ln() })
            .collect();

        // Then apply Z-score
        let temp = Self {
            context: log_context,
            target: log_target,
            dataset: self.dataset.clone(),
            freq: self.freq.clone(),
        };
        temp.zscore_normalize()
    }

    /// Min-max scaling to [0, 1]
    fn minmax_scale(&self) -> Self {
        let all_values: Vec<f32> = self.context.iter()
            .chain(self.target.iter())
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if all_values.is_empty() {
            return self.clone();
        }

        let min_val = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = (max_val - min_val).max(1e-8);

        Self {
            context: self.context.iter()
                .map(|x| if x.is_finite() { (x - min_val) / range } else { 0.5 })
                .collect(),
            target: self.target.iter()
                .map(|x| if x.is_finite() { (x - min_val) / range } else { 0.5 })
                .collect(),
            dataset: self.dataset.clone(),
            freq: self.freq.clone(),
        }
    }

    /// Signed log transform: sign(x) * log(1 + |x|)
    fn signed_log(&self) -> Self {
        fn transform(x: f32) -> f32 {
            if !x.is_finite() { return 0.0; }
            x.signum() * (1.0 + x.abs()).ln()
        }

        let transformed = Self {
            context: self.context.iter().map(|&x| transform(x)).collect(),
            target: self.target.iter().map(|&x| transform(x)).collect(),
            dataset: self.dataset.clone(),
            freq: self.freq.clone(),
        };
        // Follow with Z-score for additional standardization
        transformed.zscore_normalize()
    }
}

/// Available LOTSA dataset configurations.
/// Config names match the actual Hugging Face dataset: Salesforce/lotsa_data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LotsaConfig {
    // Transportation
    BeijingSubway30Min,
    HzMetro,
    LoopSeattle,
    LosLoop,
    Pems03,
    Pems04,
    Pems07,
    Pems08,
    PemsBay,
    QTraffic,
    ShMetro,
    SzTaxi,
    TrafficHourly,
    TrafficWeekly,
    // Energy & Electricity
    AustralianElectricityDemand,
    ElecDemand,
    Elf,
    Covid19Energy,
    // Weather & Climate
    BeijingAirQuality,
    ChinaAirQuality,
    Weather,
    Era5,
    Cmip6,
    Borealis,
    // Cloud & Computing
    AlibabaClusterTrace2018,
    AzureVmTraces2017,
    BorgClusterData2011,
    // Buildings
    Bdg2Bear,
    Bdg2Fox,
    Bdg2Panther,
    Bdg2Rat,
    Buildings900k,
    // Finance & Economics
    Bitcoin,
    FredMd,
    // Retail & Sales
    FavoritaSales,
    M4Daily,
    M4Hourly,
    M4Monthly,
    M4Quarterly,
    M4Weekly,
    M4Yearly,
    M5,
    MonashM3Monthly,
    MonashM3Quarterly,
    MonashM3Yearly,
    // Health
    CdcFluviewIlinet,
    CovidDeaths,
    CovidMobility,
    Hospital,
    // Other
    Nn5Daily,
    Nn5Weekly,
    PedestrianCounts,
    TaxiHourly,
    UberTlcDaily,
    UberTlcHourly,
}

impl LotsaConfig {
    /// Get the Hugging Face config name.
    pub fn config_name(&self) -> &'static str {
        match self {
            // Transportation
            LotsaConfig::BeijingSubway30Min => "BEIJING_SUBWAY_30MIN",
            LotsaConfig::HzMetro => "HZMETRO",
            LotsaConfig::LoopSeattle => "LOOP_SEATTLE",
            LotsaConfig::LosLoop => "LOS_LOOP",
            LotsaConfig::Pems03 => "PEMS03",
            LotsaConfig::Pems04 => "PEMS04",
            LotsaConfig::Pems07 => "PEMS07",
            LotsaConfig::Pems08 => "PEMS08",
            LotsaConfig::PemsBay => "PEMS_BAY",
            LotsaConfig::QTraffic => "Q-TRAFFIC",
            LotsaConfig::ShMetro => "SHMETRO",
            LotsaConfig::SzTaxi => "SZ_TAXI",
            LotsaConfig::TrafficHourly => "traffic_hourly",
            LotsaConfig::TrafficWeekly => "traffic_weekly",
            // Energy & Electricity
            LotsaConfig::AustralianElectricityDemand => "australian_electricity_demand",
            LotsaConfig::ElecDemand => "elecdemand",
            LotsaConfig::Elf => "elf",
            LotsaConfig::Covid19Energy => "covid19_energy",
            // Weather & Climate
            LotsaConfig::BeijingAirQuality => "beijing_air_quality",
            LotsaConfig::ChinaAirQuality => "china_air_quality",
            LotsaConfig::Weather => "weather",
            LotsaConfig::Era5 => "era5",
            LotsaConfig::Cmip6 => "cmip6",
            LotsaConfig::Borealis => "borealis",
            // Cloud & Computing
            LotsaConfig::AlibabaClusterTrace2018 => "alibaba_cluster_trace_2018",
            LotsaConfig::AzureVmTraces2017 => "azure_vm_traces_2017",
            LotsaConfig::BorgClusterData2011 => "borg_cluster_data_2011",
            // Buildings
            LotsaConfig::Bdg2Bear => "bdg-2_bear",
            LotsaConfig::Bdg2Fox => "bdg-2_fox",
            LotsaConfig::Bdg2Panther => "bdg-2_panther",
            LotsaConfig::Bdg2Rat => "bdg-2_rat",
            LotsaConfig::Buildings900k => "buildings_900k",
            // Finance & Economics
            LotsaConfig::Bitcoin => "bitcoin_with_missing",
            LotsaConfig::FredMd => "fred_md",
            // Retail & Sales
            LotsaConfig::FavoritaSales => "favorita_sales",
            LotsaConfig::M4Daily => "m4_daily",
            LotsaConfig::M4Hourly => "m4_hourly",
            LotsaConfig::M4Monthly => "m4_monthly",
            LotsaConfig::M4Quarterly => "m4_quarterly",
            LotsaConfig::M4Weekly => "m4_weekly",
            LotsaConfig::M4Yearly => "m4_yearly",
            LotsaConfig::M5 => "m5",
            LotsaConfig::MonashM3Monthly => "monash_m3_monthly",
            LotsaConfig::MonashM3Quarterly => "monash_m3_quarterly",
            LotsaConfig::MonashM3Yearly => "monash_m3_yearly",
            // Health
            LotsaConfig::CdcFluviewIlinet => "cdc_fluview_ilinet",
            LotsaConfig::CovidDeaths => "covid_deaths",
            LotsaConfig::CovidMobility => "covid_mobility",
            LotsaConfig::Hospital => "hospital",
            // Other
            LotsaConfig::Nn5Daily => "nn5_daily_with_missing",
            LotsaConfig::Nn5Weekly => "nn5_weekly",
            LotsaConfig::PedestrianCounts => "pedestrian_counts",
            LotsaConfig::TaxiHourly => "taxi_30min",
            LotsaConfig::UberTlcDaily => "uber_tlc_daily",
            LotsaConfig::UberTlcHourly => "uber_tlc_hourly",
        }
    }

    /// Get the frequency string.
    pub fn freq(&self) -> &'static str {
        match self {
            LotsaConfig::BeijingSubway30Min | LotsaConfig::TaxiHourly => "30min",
            LotsaConfig::HzMetro | LotsaConfig::LoopSeattle | LotsaConfig::LosLoop |
            LotsaConfig::Pems03 | LotsaConfig::Pems04 | LotsaConfig::Pems07 |
            LotsaConfig::Pems08 | LotsaConfig::PemsBay | LotsaConfig::QTraffic |
            LotsaConfig::ShMetro | LotsaConfig::SzTaxi | LotsaConfig::TrafficHourly |
            LotsaConfig::AustralianElectricityDemand | LotsaConfig::ElecDemand |
            LotsaConfig::Elf | LotsaConfig::Covid19Energy | LotsaConfig::BeijingAirQuality |
            LotsaConfig::ChinaAirQuality | LotsaConfig::Weather | LotsaConfig::Era5 |
            LotsaConfig::Cmip6 | LotsaConfig::Borealis | LotsaConfig::AlibabaClusterTrace2018 |
            LotsaConfig::AzureVmTraces2017 | LotsaConfig::BorgClusterData2011 |
            LotsaConfig::Bdg2Bear | LotsaConfig::Bdg2Fox | LotsaConfig::Bdg2Panther |
            LotsaConfig::Bdg2Rat | LotsaConfig::Buildings900k | LotsaConfig::M4Hourly |
            LotsaConfig::UberTlcHourly => "1h",
            LotsaConfig::TrafficWeekly | LotsaConfig::M4Weekly | LotsaConfig::Nn5Weekly => "1w",
            LotsaConfig::M4Daily | LotsaConfig::Nn5Daily | LotsaConfig::UberTlcDaily => "1d",
            LotsaConfig::M4Monthly | LotsaConfig::MonashM3Monthly | LotsaConfig::FredMd => "1mo",
            LotsaConfig::M4Quarterly | LotsaConfig::MonashM3Quarterly => "1q",
            LotsaConfig::M4Yearly | LotsaConfig::MonashM3Yearly => "1y",
            LotsaConfig::Bitcoin | LotsaConfig::FavoritaSales | LotsaConfig::M5 |
            LotsaConfig::CdcFluviewIlinet | LotsaConfig::CovidDeaths | LotsaConfig::CovidMobility |
            LotsaConfig::Hospital | LotsaConfig::PedestrianCounts => "1d",
        }
    }

    /// Get a curated set of diverse configs for multi-dataset training.
    /// These are verified to work with the LOTSA API and provide good coverage.
    pub fn all() -> Vec<LotsaConfig> {
        vec![
            // Transportation (diverse frequencies)
            LotsaConfig::BeijingSubway30Min,
            LotsaConfig::Pems03,
            LotsaConfig::Pems04,
            LotsaConfig::Pems07,
            LotsaConfig::Pems08,
            LotsaConfig::TrafficHourly,
            // Energy
            LotsaConfig::AustralianElectricityDemand,
            // Weather
            LotsaConfig::BeijingAirQuality,
            LotsaConfig::ChinaAirQuality,
            LotsaConfig::Weather,
            // Cloud
            LotsaConfig::AlibabaClusterTrace2018,
            // Buildings
            LotsaConfig::Bdg2Bear,
            LotsaConfig::Bdg2Fox,
            // M4 Competition (diverse frequencies)
            LotsaConfig::M4Hourly,
            LotsaConfig::M4Daily,
            LotsaConfig::M4Weekly,
            // Health
            LotsaConfig::Hospital,
            // Other
            LotsaConfig::Nn5Weekly,
            LotsaConfig::PedestrianCounts,
        ]
    }

    /// Get a small set of configs for quick testing.
    pub fn quick() -> Vec<LotsaConfig> {
        vec![
            LotsaConfig::BeijingSubway30Min,
            LotsaConfig::Pems03,
            LotsaConfig::TrafficHourly,
            LotsaConfig::M4Hourly,
        ]
    }

    /// Get all transportation-related configs.
    pub fn transportation() -> Vec<LotsaConfig> {
        vec![
            LotsaConfig::BeijingSubway30Min,
            LotsaConfig::HzMetro,
            LotsaConfig::LoopSeattle,
            LotsaConfig::LosLoop,
            LotsaConfig::Pems03,
            LotsaConfig::Pems04,
            LotsaConfig::Pems07,
            LotsaConfig::Pems08,
            LotsaConfig::PemsBay,
            LotsaConfig::QTraffic,
            LotsaConfig::ShMetro,
            LotsaConfig::SzTaxi,
            LotsaConfig::TrafficHourly,
            LotsaConfig::TrafficWeekly,
        ]
    }

    /// Get all M4 competition configs.
    pub fn m4() -> Vec<LotsaConfig> {
        vec![
            LotsaConfig::M4Daily,
            LotsaConfig::M4Hourly,
            LotsaConfig::M4Monthly,
            LotsaConfig::M4Quarterly,
            LotsaConfig::M4Weekly,
            LotsaConfig::M4Yearly,
        ]
    }
}

/// LOTSA dataset loader.
pub struct LotsaLoader {
    client: Client,
    context_len: usize,
    horizon: usize,
}

impl Default for LotsaLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl LotsaLoader {
    /// Create a new loader with default context length (512) and horizon (96).
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            context_len: 512,
            horizon: 96,
        }
    }

    /// Create a loader with custom context and horizon.
    pub fn with_params(context_len: usize, horizon: usize) -> Self {
        Self {
            client: Client::new(),
            context_len,
            horizon,
        }
    }

    /// Load time series samples from a specific LOTSA config.
    pub async fn load_time_series(
        &self,
        config: LotsaConfig,
        num_samples: usize,
        offset: usize,
    ) -> Result<Vec<TimeSeriesSample>> {
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset=Salesforce/lotsa_data&config={}&split=train&offset={}&length={}",
            config.config_name(),
            offset,
            num_samples
        );

        let resp = self.client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to fetch LOTSA data from {}: {}",
                config.config_name(),
                resp.status()
            ));
        }

        let json_body: HfResponse = resp.json().await?;
        let mut samples = Vec::new();

        for item in json_body.rows {
            let row = item.row;
            if let Some(target) = row.get("target").and_then(|v| v.as_array()) {
                // Extract time series values
                let series: Vec<f32> = target
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .chain(
                        target
                            .iter()
                            .filter_map(|v| v.as_array())
                            .flat_map(|arr| arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32))),
                    )
                    .collect();

                // Create sliding windows for training samples
                let min_len = self.context_len + self.horizon;
                if series.len() >= min_len {
                    // Create multiple samples from a single long series
                    let num_windows = (series.len() - min_len) / (self.horizon / 2).max(1) + 1;
                    let num_windows = num_windows.min(10); // Limit per series

                    for w in 0..num_windows {
                        let start = w * (self.horizon / 2).max(1);
                        if start + min_len > series.len() {
                            break;
                        }

                        let context = series[start..start + self.context_len].to_vec();
                        let target = series[start + self.context_len..start + min_len].to_vec();

                        samples.push(TimeSeriesSample {
                            context,
                            target,
                            dataset: config.config_name().to_string(),
                            freq: config.freq().to_string(),
                        });
                    }
                }
            }
        }

        Ok(samples)
    }

    /// Load from multiple configs for universal forecasting.
    pub async fn load_multi_dataset(
        &self,
        configs: &[LotsaConfig],
        samples_per_config: usize,
    ) -> Result<Vec<TimeSeriesSample>> {
        let mut all_samples = Vec::new();

        for config in configs {
            match self.load_time_series(*config, samples_per_config, 0).await {
                Ok(samples) => {
                    println!(
                        "Loaded {} samples from {}",
                        samples.len(),
                        config.config_name()
                    );
                    all_samples.extend(samples);
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load {}: {}", config.config_name(), e);
                }
            }
        }

        Ok(all_samples)
    }

    /// Load samples and convert to old MathProblem format for backward compatibility.
    pub async fn load_samples(&self, num_samples: usize) -> Result<Vec<MathProblem>> {
        let samples = self
            .load_time_series(LotsaConfig::BeijingSubway30Min, num_samples, 0)
            .await?;

        let problems = samples
            .into_iter()
            .map(|s| {
                let context_str = s
                    .context
                    .iter()
                    .rev()
                    .take(50)
                    .rev()
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(", ");

                let target_str = s
                    .target
                    .iter()
                    .take(3)
                    .map(|f| format!("{:.2}", f))
                    .collect::<Vec<_>>()
                    .join(", ");

                MathProblem {
                    question: format!(
                        "Analyze the following time series sequence: [{}]. Predict the next 3 values.",
                        context_str
                    ),
                    answer: target_str,
                    solution: "Predicted based on trend.".to_string(),
                    category: "lotsa".to_string(),
                    operation: "forecasting".to_string(),
                    difficulty: "real_world".to_string(),
                }
            })
            .collect();

        Ok(problems)
    }

    /// Save samples to JSONL file.
    pub fn save_to_jsonl(samples: &[TimeSeriesSample], path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        for sample in samples {
            let json = serde_json::to_string(sample)?;
            writeln!(writer, "{}", json)?;
        }

        Ok(())
    }

    /// Load samples from JSONL file.
    pub fn load_from_jsonl(path: &std::path::Path) -> Result<Vec<TimeSeriesSample>> {
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let mut samples = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let sample: TimeSeriesSample = serde_json::from_str(&line)?;
                samples.push(sample);
            }
        }

        Ok(samples)
    }
}
