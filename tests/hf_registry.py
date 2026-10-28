"""
Registry of the models of the parser robustness sweep.

Each entry describes how to load a model and which kind of dummy inputs to
build for it. The local entries use the factories of hf_local_models, the
others are loaded from the Hugging Face hub with the given auto class.
"""

# Import Modules
from dataclasses import dataclass, field
from typing import Callable

from torch import Tensor, nn

from tests.hf_local_models import LOCAL_BUILDERS, LocalBuilder


@dataclass
class ModelSpec:
    """
    One model of the robustness sweep.

    Args:
        name (str): The short name used for the output folder.
        category (str): The architecture family (Text, Vision, ...).
        repo (str): The Hugging Face repository id, or "" for a local model.
        loader (str): The auto class name (AutoModel, AutoModelForCausalLM,
            ...) or "local" for the local builders.
        input_kind (str): The dummy input kind (text, encdec, vision,
            audio_raw, audio_spec, speech_spec, clip, layout, detr,
            vision_encdec, vits, local).
        builder (Callable | None): For local models, the factory returning
            (model, inputs).
        note (str): An optional note about the model.
    """

    # The short name and the category of the model
    name: str
    category: str

    # The repository to load, empty for a local model
    repo: str

    # The loader to use
    loader: str

    # The kind of dummy inputs to build
    input_kind: str

    # The optional local builder
    builder: Callable[[], tuple[nn.Module, dict[str, Tensor]]] | None = field(
        default=None
    )

    # An optional note about the model
    note: str = ""


def _spec(
    name: str,
    category: str,
    repo: str,
    loader: str,
    input_kind: str,
    note: str = ""
) -> ModelSpec:
    """
    Build a Hugging Face model spec.

    Args:
        name (str): The short name used for the output folder.
        category (str): The architecture family of the model.
        repo (str): The Hugging Face repository id.
        loader (str): The auto class name of the model.
        input_kind (str): The dummy input kind of the model.
        note (str): An optional note about the model.

    Returns:
        ModelSpec: The registry entry of the model.
    """

    # The registry entry of the Hugging Face model
    return ModelSpec(
        name=name,
        category=category,
        repo=repo,
        loader=loader,
        input_kind=input_kind,
        note=note
    )


def _local_spec(name: str, category: str, note: str = "") -> ModelSpec:
    """
    Build a local model spec from its factory.

    Args:
        name (str): The short name used for the output folder.
        category (str): The architecture family of the model.
        note (str): An optional note about the model.

    Returns:
        ModelSpec: The registry entry of the local model.
    """

    # The registry entry of the local model
    return ModelSpec(
        name=name,
        category=category,
        repo="",
        loader="local",
        input_kind="local",
        builder=LOCAL_BUILDERS[name],
        note=note
    )


# The models of the robustness sweep, by category
MODEL_SPECS: list[ModelSpec] = [
    # --- Text encoders ---
    _spec(
        "bert_tiny",
        "Text",
        "hf-internal-testing/tiny-random-BertModel",
        "AutoModel",
        "text"
    ),
    _spec(
        "bert_l4_h256",
        "Text",
        "google/bert_uncased_L-4_H-256_A-4",
        "AutoModel",
        "text"
    ),
    _spec(
        "bert_l2_h128",
        "Text",
        "google/bert_uncased_L-2_H-128_A-2",
        "AutoModel",
        "text"
    ),
    _spec(
        "minilm_l6",
        "Text",
        "sentence-transformers/all-MiniLM-L6-v2",
        "AutoModel",
        "text"
    ),
    _spec(
        "distilbert",
        "Text",
        "distilbert/distilbert-base-uncased",
        "AutoModel",
        "text"
    ),
    _spec("albert_base", "Text", "albert/albert-base-v2", "AutoModel", "text"),
    _spec(
        "roberta_base",
        "Text",
        "FacebookAI/roberta-base",
        "AutoModel",
        "text"
    ),
    _spec(
        "deberta_v3_xsmall",
        "Text",
        "microsoft/deberta-v3-xsmall",
        "AutoModel",
        "text"
    ),
    _spec(
        "electra_small",
        "Text",
        "google/electra-small-discriminator",
        "AutoModel",
        "text"
    ),
    _spec(
        "squeezebert",
        "Text",
        "squeezebert/squeezebert-uncased",
        "AutoModel",
        "text"
    ),
    _spec(
        "mobilebert",
        "Text",
        "google/mobilebert-uncased",
        "AutoModel",
        "text"
    ),
    _spec(
        "layoutlm_base",
        "Text",
        "microsoft/layoutlm-base-uncased",
        "AutoModel",
        "layout",
        note="text plus bounding boxes"
    ),

    # --- Text decoders ---
    _spec(
        "tiny_gpt2",
        "Text",
        "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        "AutoModelForCausalLM",
        "text"
    ),
    _spec(
        "distilgpt2",
        "Text",
        "distilbert/distilgpt2",
        "AutoModelForCausalLM",
        "text"
    ),
    _spec("gpt_neo_125m", "Text", "EleutherAI/gpt-neo-125m",
          "AutoModelForCausalLM", "text"),
    _spec("opt_125m", "Text", "facebook/opt-125m",
          "AutoModelForCausalLM", "text"),
    _spec("pythia_70m", "Text", "EleutherAI/pythia-70m",
          "AutoModelForCausalLM", "text"),
    _spec("qwen2_0_5b", "Text", "Qwen/Qwen2.5-0.5B",
          "AutoModelForCausalLM", "text"),
    _spec("smollm2_135m", "Text", "HuggingFaceTB/SmolLM2-135M",
          "AutoModelForCausalLM", "text"),
    _spec("bloom_560m", "Text", "bigscience/bloom-560m",
          "AutoModelForCausalLM", "text"),
    _spec("tiny_starcoder_py", "Text", "bigcode/tiny_starcoder_py",
          "AutoModelForCausalLM", "text"),
    _spec("mamba_130m", "Text", "state-spaces/mamba-130m-hf",
          "AutoModelForCausalLM", "text", note="SSM recurrence"),

    # --- Text encoder decoders ---
    _spec("t5_small", "Text", "t5-small", "AutoModelForSeq2SeqLM", "encdec"),
    _spec(
        "flan_t5_small",
        "Text",
        "google/flan-t5-small",
        "AutoModelForSeq2SeqLM",
        "encdec"
    ),
    _spec(
        "bart_base",
        "Text",
        "facebook/bart-base",
        "AutoModelForSeq2SeqLM",
        "encdec"
    ),
    _spec(
        "opus_mt_en_de",
        "Text",
        "Helsinki-NLP/opus-mt-en-de",
        "AutoModelForSeq2SeqLM",
        "encdec"
    ),
    _spec(
        "switch_base_8",
        "Text",
        "google/switch-base-8",
        "AutoModelForSeq2SeqLM",
        "encdec",
        note="mixture of experts"
    ),

    # --- Vision ---
    _spec(
        "deit_tiny",
        "Vision",
        "facebook/deit-tiny-patch16-224",
        "AutoModel",
        "vision"
    ),
    _spec(
        "vit_base",
        "Vision",
        "google/vit-base-patch16-224",
        "AutoModel",
        "vision"
    ),
    _spec("resnet_18", "Vision", "microsoft/resnet-18", "AutoModel", "vision"),
    _spec(
        "mobilenet_v2",
        "Vision",
        "google/mobilenet_v2_0.35_96",
        "AutoModel",
        "vision"
    ),
    _spec("mit_b0", "Vision", "nvidia/mit-b0", "AutoModel", "vision",
          note="segformer encoder"),
    _spec(
        "mobilevit_small",
        "Vision",
        "apple/mobilevit-small",
        "AutoModel",
        "vision"
    ),
    _spec("levit_128s", "Vision", "facebook/levit-128s", "AutoModel",
          "vision"),
    _spec(
        "convnext_tiny",
        "Vision",
        "facebook/convnext-tiny-224",
        "AutoModel",
        "vision"
    ),
    _spec(
        "beit_base",
        "Vision",
        "microsoft/beit-base-patch16-224",
        "AutoModel",
        "vision"
    ),
    _spec(
        "detr_resnet_50",
        "Vision",
        "facebook/detr-resnet-50",
        "AutoModelForObjectDetection",
        "processor",
        note="object detection head"
    ),

    # --- Audio ---
    _spec(
        "wav2vec2_base",
        "Audio",
        "facebook/wav2vec2-base",
        "AutoModel",
        "audio_raw"
    ),
    _spec(
        "wav2vec2_base_960h",
        "Audio",
        "facebook/wav2vec2-base-960h",
        "AutoModelForCTC",
        "audio_raw",
        note="speech to text CTC head"
    ),
    _spec(
        "hubert_base",
        "Audio",
        "facebook/hubert-base-ls960",
        "AutoModel",
        "audio_raw"
    ),
    _spec(
        "ast_speech_commands",
        "Audio",
        "MIT/ast-finetuned-speech-commands-v2",
        "AutoModel",
        "audio_processor",
        note="audio spectrogram transformer"
    ),
    _spec(
        "whisper_tiny",
        "Audio",
        "openai/whisper-tiny",
        "AutoModelForSpeechSeq2Seq",
        "speech_spec",
        note="speech to text encoder decoder"
    ),
    _spec(
        "s2t_small",
        "Audio",
        "facebook/s2t-small-librispeech-asr",
        "AutoModelForSpeechSeq2Seq",
        "s2t_spec",
        note="speech to text encoder decoder"
    ),
    _spec(
        "mms_tts_eng",
        "Audio",
        "facebook/mms-tts-eng",
        "AutoModel",
        "vits",
        note="text to speech with random noise"
    ),

    # --- Multimodal ---
    _spec(
        "clip_base",
        "Multimodal",
        "openai/clip-vit-base-patch32",
        "AutoModel",
        "clip"
    ),
    _spec(
        "blip_caption",
        "Multimodal",
        "Salesforce/blip-image-captioning-base",
        "BlipForConditionalGeneration",
        "clip",
        note="image captioning encoder decoder"
    ),
    _spec(
        "trocr_small_handwritten",
        "Multimodal",
        "microsoft/trocr-small-handwritten",
        "VisionEncoderDecoderModel",
        "vision_encdec"
    ),
    _spec(
        "vit_gpt2_caption",
        "Multimodal",
        "nlpconnect/vit-gpt2-image-captioning",
        "VisionEncoderDecoderModel",
        "vision_encdec"
    ),

    # --- Local architectures ---
    _local_spec("local_mlp", "MLP"),
    _local_spec("local_lstm", "RNN", note="lstm recurrence"),
    _local_spec("local_rnn", "RNN", note="vanilla rnn recurrence"),
    _local_spec("local_gru_seq2seq", "RNN", note="gru recurrence"),
    _local_spec("local_bilstm_attention", "RNN", note="bidirectional + attention"),
    _local_spec("local_resnet_cnn", "CNN", note="residual convolutions"),
    _local_spec("local_conv_autoencoder", "AutoEncoder"),
    _local_spec("local_gan_generator", "GAN"),
    _local_spec("local_gan_discriminator", "GAN"),
    _local_spec("local_gcn", "GNN", note="dense adjacency propagation")
]
