use super::*;

#[test]
fn test_bedrock_model_summary_capability_checks() {
    let text_model = BedrockModelSummary {
        model_id: "anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
        model_name: "Claude 3.5 Sonnet v2".to_string(),
        provider_name: "Anthropic".to_string(),
        input_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
        output_modalities: vec!["TEXT".to_string()],
        response_streaming_supported: true,
        inference_types_supported: vec!["ON_DEMAND".to_string()],
        lifecycle_status: "ACTIVE".to_string(),
    };

    assert!(text_model.supports_text_input());
    assert!(text_model.supports_text_output());
    assert!(text_model.supports_on_demand());
    assert!(text_model.is_active());
    assert!(BedrockDirectClient::evaluate_model_capabilities(&text_model).is_none());
}

#[test]
fn test_embedding_model_unsupported_modality() {
    let embedding_model = BedrockModelSummary {
        model_id: "amazon.titan-embed-text-v2:0".to_string(),
        model_name: "Titan Text Embeddings v2".to_string(),
        provider_name: "Amazon".to_string(),
        input_modalities: vec!["TEXT".to_string()],
        output_modalities: vec!["EMBEDDING".to_string()],
        response_streaming_supported: false,
        inference_types_supported: vec!["ON_DEMAND".to_string()],
        lifecycle_status: "ACTIVE".to_string(),
    };

    assert!(embedding_model.supports_text_input());
    assert!(!embedding_model.supports_text_output());

    let status = BedrockDirectClient::evaluate_model_capabilities(&embedding_model);
    assert_eq!(
        status,
        Some(ModelUsability::UnsupportedModality {
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["EMBEDDING".to_string()],
        })
    );
    assert!(!status.unwrap().is_usable());
}

#[test]
fn test_image_generation_model_unsupported_modality() {
    let image_model = BedrockModelSummary {
        model_id: "amazon.titan-image-generator-v2:0".to_string(),
        model_name: "Titan Image Generator v2".to_string(),
        provider_name: "Amazon".to_string(),
        input_modalities: vec!["TEXT".to_string(), "IMAGE".to_string()],
        output_modalities: vec!["IMAGE".to_string()],
        response_streaming_supported: false,
        inference_types_supported: vec!["ON_DEMAND".to_string()],
        lifecycle_status: "ACTIVE".to_string(),
    };

    assert!(!image_model.supports_text_output());
    let status = BedrockDirectClient::evaluate_model_capabilities(&image_model);
    assert!(matches!(
        status,
        Some(ModelUsability::UnsupportedModality { .. })
    ));
}

#[test]
fn test_legacy_model_lifecycle_status() {
    let legacy_model = BedrockModelSummary {
        model_id: "anthropic.claude-v1".to_string(),
        model_name: "Claude v1".to_string(),
        provider_name: "Anthropic".to_string(),
        input_modalities: vec!["TEXT".to_string()],
        output_modalities: vec!["TEXT".to_string()],
        response_streaming_supported: true,
        inference_types_supported: vec!["ON_DEMAND".to_string()],
        lifecycle_status: "LEGACY".to_string(),
    };

    let status = BedrockDirectClient::evaluate_model_capabilities(&legacy_model);
    assert_eq!(
        status,
        Some(ModelUsability::LegacyStatus {
            status: "LEGACY".to_string()
        })
    );
}

#[test]
fn test_provisioned_only_model_detection() {
    let provisioned_model = BedrockModelSummary {
        model_id: "custom-provisioned-model".to_string(),
        model_name: "Custom Model".to_string(),
        provider_name: "Custom".to_string(),
        input_modalities: vec!["TEXT".to_string()],
        output_modalities: vec!["TEXT".to_string()],
        response_streaming_supported: true,
        inference_types_supported: vec!["PROVISIONED".to_string()],
        lifecycle_status: "ACTIVE".to_string(),
    };

    let status = BedrockDirectClient::evaluate_model_capabilities(&provisioned_model);
    assert_eq!(status, Some(ModelUsability::RequiresProvisionedThroughput));
}

#[test]
fn test_model_usability_failure_reasons() {
    let access_denied = ModelUsability::AccessDenied {
        reason: "You do not have access to this model".to_string(),
    };
    assert!(!access_denied.is_usable());
    assert!(
        access_denied
            .failure_reason()
            .unwrap()
            .contains("Access denied")
    );

    let usable = ModelUsability::Usable { streaming: true };
    assert!(usable.is_usable());
    assert!(usable.failure_reason().is_none());
}

#[test]
fn test_verified_bedrock_model_serialization_roundtrip() {
    let verified = VerifiedBedrockModel {
        summary: BedrockModelSummary {
            model_id: "meta.llama3-3-70b-instruct-v1:0".to_string(),
            model_name: "Llama 3.3 70B Instruct".to_string(),
            provider_name: "Meta".to_string(),
            input_modalities: vec!["TEXT".to_string()],
            output_modalities: vec!["TEXT".to_string()],
            response_streaming_supported: true,
            inference_types_supported: vec!["ON_DEMAND".to_string()],
            lifecycle_status: "ACTIVE".to_string(),
        },
        usability: ModelUsability::Usable { streaming: true },
    };

    let json = serde_json::to_string(&verified).expect("Serialization succeeds");
    let deserialized: VerifiedBedrockModel =
        serde_json::from_str(&json).expect("Deserialization succeeds");
    assert_eq!(verified, deserialized);
}
