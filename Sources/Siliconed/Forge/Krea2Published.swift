import Foundation

/// **The three small files of `krea/Krea-2-Turbo` the engine reads, as published** — the only bytes
/// of that repository an installation still needs, and the Hub serves them behind a gate (a token, a
/// license accepted): a user of the app has neither. Their weights come from ungated repositories
/// (`Installer.dit`, `Installer.encoder`); these three are written here **verbatim**, byte for byte,
/// at revision `98e0fe11`. A test hashes each literal as git does (`blob <n>\0` + bytes, SHA-1) and
/// finds the blob id the Hub lists for the file at that revision: the text here *is* the file.
///
///     transformer/config.json    → the DiT map's header `config` (`ForgeDiT`, `Krea2DiT.Config`)
///     text_encoder/config.json   → its `text_config`, the encoder map's header (`ForgeText`)
///     model_index.json           → `text_encoder_select_layers`, the twelve layers drawn (`Krea2Text`)
///
/// Nothing else of them is read. What a port by analogy would miss: the encoder's `config.json` is
/// **not** `Qwen/Qwen3-VL-4B-Instruct`'s (same values, but `rope_theta` under `rope_parameters` and
/// `pad_token_id: null`, transformers 5's serialization): the map's header copies the published
/// `text_config` whole, and only this one gives the map forged from Krea's own repository, to the byte.
package enum Krea2Published {
    /// The revision these bytes are from, and each file's git blob id there (`/api/models/…/tree`).
    package static let revision = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"
    package static let blobs: [(path: String, blob: String, text: String)] = [
        ("transformer/config.json", "a1bbdd4203a18e23ee90aa3b0c9fca5cff0b72ef", transformerConfigText),
        ("text_encoder/config.json", "259091957835d33e698926946f7a86189240ae30", encoderConfigText),
        ("model_index.json", "889dbf53a26b78303ec0d9782c9bd3a1a13e2483", modelIndexText),
    ]

    /// `transformer/config.json`, parsed in its order (the DiT map's header copies it).
    package static var transformerConfig: OrderedJSON { parse(transformerConfigText) }
    /// `text_encoder/config.json`, parsed in its order (`ForgeText` keeps its `text_config`).
    package static var encoderConfig: OrderedJSON { parse(encoderConfigText) }
    /// `model_index.json`, parsed in its order.
    package static var modelIndex: OrderedJSON { parse(modelIndexText) }

    private static func parse(_ text: String) -> OrderedJSON {
        // The literals are fixed and tested: a parse failure is a programming error.
        try! OrderedJSON.parse(Data(text.utf8))
    }

    static let transformerConfigText = #"""
        {
          "_class_name": "Krea2Transformer2DModel",
          "_diffusers_version": "0.39.0.dev0",
          "attention_head_dim": 128,
          "axes_dims_rope": [
            32,
            48,
            48
          ],
          "in_channels": 64,
          "intermediate_size": 16384,
          "norm_eps": 1e-05,
          "num_attention_heads": 48,
          "num_key_value_heads": 12,
          "num_layers": 28,
          "num_layerwise_text_blocks": 2,
          "num_refiner_text_blocks": 2,
          "num_text_layers": 12,
          "rope_theta": 1000.0,
          "text_hidden_dim": 2560,
          "text_intermediate_size": 6912,
          "text_num_attention_heads": 20,
          "text_num_key_value_heads": 20,
          "timestep_embed_dim": 256
        }

        """#

    static let encoderConfigText = #"""
        {
          "architectures": [
            "Qwen3VLModel"
          ],
          "dtype": "bfloat16",
          "image_token_id": 151655,
          "model_type": "qwen3_vl",
          "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "dtype": "bfloat16",
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "initializer_range": 0.02,
            "intermediate_size": 9728,
            "max_position_embeddings": 262144,
            "model_type": "qwen3_vl_text",
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "pad_token_id": null,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [
                24,
                20,
                20
              ],
              "rope_theta": 5000000,
              "rope_type": "default"
            },
            "tie_word_embeddings": true,
            "use_cache": true,
            "vocab_size": 151936
          },
          "tie_word_embeddings": true,
          "transformers_version": "5.2.0",
          "video_token_id": 151656,
          "vision_config": {
            "deepstack_visual_indexes": [
              5,
              11,
              17
            ],
            "depth": 24,
            "dtype": "bfloat16",
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "model_type": "qwen3_vl",
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2
          },
          "vision_end_token_id": 151653,
          "vision_start_token_id": 151652
        }

        """#

    static let modelIndexText = #"""
        {
          "_class_name": "Krea2Pipeline",
          "_diffusers_version": "0.39.0.dev0",
          "scheduler": [
            "diffusers",
            "FlowMatchEulerDiscreteScheduler"
          ],
          "text_encoder": [
            "transformers",
            "Qwen3VLModel"
          ],
          "text_encoder_select_layers": [
            2,
            5,
            8,
            11,
            14,
            17,
            20,
            23,
            26,
            29,
            32,
            35
          ],
          "tokenizer": [
            "transformers",
            "Qwen2Tokenizer"
          ],
          "transformer": [
            "diffusers",
            "Krea2Transformer2DModel"
          ],
          "vae": [
            "diffusers",
            "AutoencoderKLQwenImage"
          ],
          "is_distilled": true,
          "patch_size": 2
        }
        """#
}
