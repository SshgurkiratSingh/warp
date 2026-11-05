use anyhow::{Context as _, Result};
use async_stream::stream;
use aws_config::BehaviorVersion;
use aws_sdk_bedrock::Client as BedrockClient;
use aws_sdk_bedrock::types::{InferenceType, ModelModality};
use aws_sdk_bedrockruntime::Client as BedrockRuntimeClient;
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, ConverseStreamOutput, Message,
};
use futures::Stream;
use serde::{Deserialize, Serialize};

/// Configuration parameters for establishing a direct AWS Bedrock connection.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockClientConfig {
    pub profile: Option<String>,
    pub region: Option<String>,
}

/// A foundation model summary fetched directly from AWS Bedrock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BedrockModelSummary {
    pub model_id: String,
    pub model_name: String,
    pub provider_name: String,
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub response_streaming_supported: bool,
    pub inference_types_supported: Vec<String>,
    pub lifecycle_status: String,
}

impl BedrockModelSummary {
    pub fn supports_text_input(&self) -> bool {
        self.input_modalities.iter().any(|m| m.eq_ignore_ascii_case("TEXT"))
    }

    pub fn supports_text_output(&self) -> bool {
        self.output_modalities.iter().any(|m| m.eq_ignore_ascii_case("TEXT"))
    }

    pub fn supports_on_demand(&self) -> bool {
        self.inference_types_supported
            .iter()
            .any(|t| t.eq_ignore_ascii_case("ON_DEMAND"))
    }

    pub fn is_active(&self) -> bool {
        self.lifecycle_status.eq_ignore_ascii_case("ACTIVE")
    }
}

/// Detailed verification status of whether a model can be used.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelUsability {
    /// Model is active, supports text generation, and the AWS account has invocation permissions.
    Usable {
        streaming: bool,
    },
    /// Model does not support text input or text output (e.g. image generation or embeddings only).
    UnsupportedModality {
        input_modalities: Vec<String>,
        output_modalities: Vec<String>,
    },
    /// Model is deprecated or marked legacy by AWS.
    LegacyStatus {
        status: String,
    },
    /// Model does not support on-demand inference (provisioned throughput required).
    RequiresProvisionedThroughput,
    /// Model access has not been granted to this AWS account in the AWS Bedrock console.
    AccessDenied {
        reason: String,
    },
    /// Model ID not found in the current region.
    NotFound {
        reason: String,
    },
    /// Request failed validation or model requires specific configuration.
    ValidationFailed {
        reason: String,
    },
    /// Account or region rate limit exceeded.
    Throttled {
        reason: String,
    },
    /// Other error occurred when verifying model access.
    VerificationFailed {
        reason: String,
    },
}

impl ModelUsability {
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Usable { .. })
    }

    pub fn failure_reason(&self) -> Option<String> {
        match self {
            Self::Usable { .. } => None,
            Self::UnsupportedModality {
                input_modalities,
                output_modalities,
            } => Some(format!(
                "Unsupported modalities: input [{}] / output [{}]",
                input_modalities.join(", "),
                output_modalities.join(", ")
            )),
            Self::LegacyStatus { status } => {
                Some(format!("Model lifecycle status is {status} (not active)"))
            }
            Self::RequiresProvisionedThroughput => Some(
                "Model only supports provisioned throughput; on-demand inference is unavailable"
                    .to_string(),
            ),
            Self::AccessDenied { reason } => Some(format!(
                "Access denied: model access not granted in AWS Bedrock console ({reason})"
            )),
            Self::NotFound { reason } => Some(format!("Model not found in region: {reason}")),
            Self::ValidationFailed { reason } => Some(format!("Validation failed: {reason}")),
            Self::Throttled { reason } => Some(format!("Request throttled: {reason}")),
            Self::VerificationFailed { reason } => {
                Some(format!("Verification call failed: {reason}"))
            }
        }
    }
}

/// A foundation model with its verified usability outcome.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedBedrockModel {
    pub summary: BedrockModelSummary,
    pub usability: ModelUsability,
}

/// Direct client for AWS Bedrock control plane and runtime inference.
#[derive(Clone)]
pub struct BedrockDirectClient {
    bedrock: BedrockClient,
    runtime: BedrockRuntimeClient,
    region: String,
}

impl BedrockDirectClient {
    /// Constructs a direct client from `BedrockClientConfig` using the AWS credential chain.
    pub async fn from_config(config: BedrockClientConfig) -> Result<Self> {
        let region_provider = aws_config::meta::region::RegionProviderChain::first_try(
            config.region.clone().map(aws_config::Region::new),
        )
        .or_default_provider();

        let mut loader =
            aws_config::defaults(BehaviorVersion::latest()).region(region_provider);

        if let Some(profile) = config.profile.filter(|p| !p.trim().is_empty()) {
            loader = loader.profile_name(profile);
        }

        let sdk_config = loader.load().await;
        Self::from_sdk_config(&sdk_config)
    }

    /// Constructs a direct client from an already loaded `aws_config::SdkConfig`.
    pub fn from_sdk_config(sdk_config: &aws_config::SdkConfig) -> Result<Self> {
        let region = sdk_config
            .region()
            .map(|r| r.as_ref().to_string())
            .unwrap_or_else(|| "us-east-1".to_string());

        let bedrock = BedrockClient::new(sdk_config);
        let runtime = BedrockRuntimeClient::new(sdk_config);

        Ok(Self {
            bedrock,
            runtime,
            region,
        })
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    /// Fetches the foundation models list strictly from AWS Bedrock.
    pub async fn list_foundation_models_from_aws(&self) -> Result<Vec<BedrockModelSummary>> {
        let response = self
            .bedrock
            .list_foundation_models()
            .send()
            .await
            .context("Failed to list foundation models directly from AWS Bedrock")?;

        let summaries = response
            .model_summaries()
            .iter()
            .map(|summary| {
                let model_id = summary.model_id().to_string();
                let model_name = summary.model_name().map(str::to_string).unwrap_or_else(|| model_id.clone());
                let provider_name = summary.provider_name().map(str::to_string).unwrap_or_default();

                let input_modalities = summary
                    .input_modalities()
                    .iter()
                    .map(|m| match m {
                        ModelModality::Text => "TEXT".to_string(),
                        ModelModality::Image => "IMAGE".to_string(),
                        ModelModality::Embedding => "EMBEDDING".to_string(),
                        other => other.as_str().to_string(),
                    })
                    .collect();

                let output_modalities = summary
                    .output_modalities()
                    .iter()
                    .map(|m| match m {
                        ModelModality::Text => "TEXT".to_string(),
                        ModelModality::Image => "IMAGE".to_string(),
                        ModelModality::Embedding => "EMBEDDING".to_string(),
                        other => other.as_str().to_string(),
                    })
                    .collect();

                let response_streaming_supported =
                    summary.response_streaming_supported().unwrap_or(false);

                let inference_types_supported = summary
                    .inference_types_supported()
                    .iter()
                    .map(|t| match t {
                        InferenceType::OnDemand => "ON_DEMAND".to_string(),
                        InferenceType::Provisioned => "PROVISIONED".to_string(),
                        other => other.as_str().to_string(),
                    })
                    .collect();

                let lifecycle_status = summary
                    .model_lifecycle()
                    .map(|l| l.status().as_str().to_string())
                    .unwrap_or_else(|| "ACTIVE".to_string());

                BedrockModelSummary {
                    model_id,
                    model_name,
                    provider_name,
                    input_modalities,
                    output_modalities,
                    response_streaming_supported,
                    inference_types_supported,
                    lifecycle_status,
                }
            })
            .collect();

        Ok(summaries)
    }

    /// Evaluates static capabilities to determine if the model meets baseline requirements.
    pub fn evaluate_model_capabilities(summary: &BedrockModelSummary) -> Option<ModelUsability> {
        if !summary.supports_text_input() || !summary.supports_text_output() {
            return Some(ModelUsability::UnsupportedModality {
                input_modalities: summary.input_modalities.clone(),
                output_modalities: summary.output_modalities.clone(),
            });
        }

        if !summary.is_active() {
            return Some(ModelUsability::LegacyStatus {
                status: summary.lifecycle_status.clone(),
            });
        }

        if !summary.supports_on_demand() {
            return Some(ModelUsability::RequiresProvisionedThroughput);
        }

        None
    }

    /// Verifies whether a foundation model is accessible and usable by this AWS account.
    ///
    /// Performs static qualification checks first, then dispatches a minimal Converse probe
    /// directly to AWS Bedrock Runtime to verify that model access is granted.
    pub async fn verify_model_usability(&self, summary: &BedrockModelSummary) -> ModelUsability {
        if let Some(unusable_status) = Self::evaluate_model_capabilities(summary) {
            return unusable_status;
        }

        self.probe_model_invocation(&summary.model_id, summary.response_streaming_supported)
            .await
    }

    /// Probes AWS Bedrock Runtime with a minimal 1-token test prompt using the Converse API.
    async fn probe_model_invocation(&self, model_id: &str, streaming: bool) -> ModelUsability {
        let test_message = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text("hi".to_string()))
            .build()
            .expect("Valid test message construction");

        let result = self
            .runtime
            .converse()
            .model_id(model_id)
            .messages(test_message)
            .send()
            .await;

        match result {
            Ok(_) => ModelUsability::Usable { streaming },
            Err(sdk_err) => {
                let service_error = sdk_err.as_service_error();
                let error_message = sdk_err.to_string();

                if let Some(err) = service_error {
                    if err.is_access_denied_exception() {
                        return ModelUsability::AccessDenied {
                            reason: error_message,
                        };
                    }
                    if err.is_resource_not_found_exception() {
                        return ModelUsability::NotFound {
                            reason: error_message,
                        };
                    }
                    if err.is_validation_exception() {
                        return ModelUsability::ValidationFailed {
                            reason: error_message,
                        };
                    }
                    if err.is_throttling_exception() {
                        return ModelUsability::Throttled {
                            reason: error_message,
                        };
                    }
                }

                if error_message.contains("AccessDenied") || error_message.contains("not have access") {
                    ModelUsability::AccessDenied {
                        reason: error_message,
                    }
                } else if error_message.contains("ResourceNotFound") {
                    ModelUsability::NotFound {
                        reason: error_message,
                    }
                } else {
                    ModelUsability::VerificationFailed {
                        reason: error_message,
                    }
                }
            }
        }
    }

    /// Fetches all foundation models from AWS and verifies each model's usability status.
    pub async fn verify_all_models(&self) -> Result<Vec<VerifiedBedrockModel>> {
        let summaries = self.list_foundation_models_from_aws().await?;
        let mut verified = Vec::with_capacity(summaries.len());

        for summary in summaries {
            let usability = self.verify_model_usability(&summary).await;
            verified.push(VerifiedBedrockModel { summary, usability });
        }

        Ok(verified)
    }

    /// Invokes a model directly via Bedrock Runtime Converse API.
    pub async fn converse_direct(
        &self,
        model_id: &str,
        system_prompt: Option<&str>,
        user_message: &str,
        max_tokens: Option<i32>,
    ) -> Result<String> {
        let message = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(user_message.to_string()))
            .build()
            .context("Failed to build user message")?;

        let mut request = self
            .runtime
            .converse()
            .model_id(model_id)
            .messages(message);

        if let Some(system) = system_prompt {
            request = request.system(aws_sdk_bedrockruntime::types::SystemContentBlock::Text(
                system.to_string(),
            ));
        }

        if let Some(tokens) = max_tokens {
            let config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                .max_tokens(tokens)
                .build();
            request = request.inference_config(config);
        }

        let response = request
            .send()
            .await
            .context("Direct Bedrock converse invocation failed")?;

        let output = response
            .output()
            .context("Converse response missing output")?;

        let text = match output {
            aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => msg
                .content()
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };

        Ok(text)
    }

    /// Streams model generation directly via Bedrock Runtime ConverseStream API.
    pub async fn converse_stream_direct(
        &self,
        model_id: &str,
        system_prompt: Option<&str>,
        user_message: &str,
        max_tokens: Option<i32>,
    ) -> Result<impl Stream<Item = Result<String>>> {
        let message = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(user_message.to_string()))
            .build()
            .context("Failed to build user message for streaming")?;

        let mut request = self
            .runtime
            .converse_stream()
            .model_id(model_id)
            .messages(message);

        if let Some(system) = system_prompt {
            request = request.system(aws_sdk_bedrockruntime::types::SystemContentBlock::Text(
                system.to_string(),
            ));
        }

        if let Some(tokens) = max_tokens {
            let config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                .max_tokens(tokens)
                .build();
            request = request.inference_config(config);
        }

        let response = request
            .send()
            .await
            .context("Direct Bedrock converse_stream invocation failed")?;

        let mut receiver = response.stream;
        let stream = stream! {
            while let Ok(Some(output)) = receiver.recv().await {
                if let ConverseStreamOutput::ContentBlockDelta(delta_event) = output {
                    if let Some(delta) = delta_event.delta() {
                        if let aws_sdk_bedrockruntime::types::ContentBlockDelta::Text(t) = delta {
                            yield Ok(t.clone());
                        }
                    }
                }
            }
        };

        Ok(stream)
    }
}

#[cfg(test)]
#[path = "direct_bedrock_tests.rs"]
mod tests;
