# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# GPU family of target platform. Supported values: tegra, non-tegra
riva_target_gpu_family="tegra"

# Name of tegra platform that is being used. Supported tegra platforms: orin, xavier
riva_tegra_platform="orin"

# Enable or Disable Riva Services
# For any language other than en-US: service_enabled_nlp must be set to false
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_tts=true
service_enabled_nmt=false

# Configure translation services
# Text-to-Text translation (T2T):
# - service_enabled_nmt must be set to true
# - Uncomment desired model for source and target languages in models_nmt field
# Speech-to-Text translation (S2T):
# - service_enabled_asr, service_enabled_nmt must be set to true
# - Set language code of input speech in the asr_language_code field
# - Uncomment desired model for source and target languages in models_nmt field
# Speech-to-Speech translation (S2S):
# - service_enabled_asr, service_enabled_nmt, service_enabled_tts must be set to true
# - Set language code of input speech in the asr_language_code field
# - Uncomment desired model for source and target languages in models_nmt field
# - Set language code of output speech in the tts_language_code field

# Enable Riva Enterprise
# If enrolled in Enterprise, enable Riva Enterprise by setting configuration
# here. You must explicitly acknowledge you have read and agree to the EULA.
RIVA_API_KEY="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"
# RIVA_API_NGC_ORG=<ngc organization>
RIVA_EULA=accept

# Language code to fetch ASR models of a specific language
# Supported language codes: ar-AR, en-US, en-GB, de-DE, es-ES, es-US, fr-FR, hi-IN, it-IT, ja-JP, ru-RU, ko-KR, pt-BR, zh-CN, nl-NL, nl-BE, es-en-US, ja-en-JP
# For multiple languages enter space separated language codes.
asr_language_code=("zh-CN")

# ASR acoustic model architecture
# Supported values are: conformer, parakeet (en-US)
asr_acoustic_model=("conformer")

# ASR acoustic model architecture variant
# Supported values for the architecture are:
# conformer: xl (en-US + amd64 only), unified(de-DE, ja-JP and zh-CN only), ml_cs(es-en-US only), unified_ml_cs(ja-en-JP only)
# parakeet: 0.6b (en-US)
# For the default model, keep the field empty
asr_acoustic_model_variant=("")

# ASR decoder type to be used
# If you'd like to use greedy decoder for ASR instead of flashlight/os2s decoder then set the below $use_asr_greedy_decoder to true
use_asr_greedy_decoder=false

# Language code to fetch TTS models of a specific language
# Supported language codes: en-US, es-ES, es-US, it-IT, de-DE, zh-CN
# For multiple languages enter space separated language codes
tts_language_code=("zh-CN")

# Specify one or more GPUs to use
# specifying more than one GPU is currently an experimental feature, and may result in undefined behaviours.
gpus_to_use="device=0"

# Specify the encryption key to use to deploy models
MODEL_DEPLOY_KEY="tlt_encode"

# Locations to use for storing models artifacts
#
# If an absolute path is specified, the data will be written to that location
# Otherwise, a Docker volume will be used (default).
#
# riva_init.sh will create a `rmir` and `models` directory in the volume or
# path specified.
#
# RMIR ($riva_model_loc/rmir)
# Riva uses an intermediate representation (RMIR) for models
# that are ready to deploy but not yet fully optimized for deployment. Pretrained
# versions can be obtained from NGC (by specifying NGC models below) and will be
# downloaded to $riva_model_loc/rmir by `riva_init.sh`
#
# Custom models produced by NeMo or TLT and prepared using riva-build
# may also be copied manually to this location $(riva_model_loc/rmir).
#
# Models ($riva_model_loc/models)
# During the riva_init process, the RMIR files in $riva_model_loc/rmir
# are inspected and optimized for deployment. The optimized versions are
# stored in $riva_model_loc/models. The riva server exclusively uses these
# optimized versions.
riva_model_loc="riva-model-repo"

if [[ $riva_target_gpu_family == "tegra" ]]; then
    riva_model_loc="`pwd`/model_repository"
fi

# The default RMIRs are downloaded from NGC by default in the above $riva_rmir_loc directory
# If you'd like to skip the download from NGC and use the existing RMIRs in the $riva_rmir_loc
# then set the below $use_existing_rmirs flag to true. You can also deploy your set of custom
# RMIRs by keeping them in the riva_rmir_loc dir and use this quickstart script with the
# below flag to deploy them all together.
use_existing_rmirs=false

# Ports to expose for Riva services
riva_speech_api_port="50051"

# NGC orgs
riva_ngc_org="nvidia"
riva_ngc_team="riva"
riva_ngc_image_version="2.15.0"
riva_ngc_model_version="2.15.0"

# Pre-built models listed below will be downloaded from NGC. If models already exist in $riva-rmir
# then models can be commented out to skip download from NGC

########## ASR MODELS ##########

validate_asr_config() {
  if [[ ${asr_acoustic_model} == "parakeet" ]]; then
    if [[ ${asr_acoustic_model_variant} != "0.6b" ]]; then
      echo "Only 0.6b variant is available for Parakeet model"
      exit 1
    fi

    if [[ ${lang_code} != "en-US" ]]; then
      echo "Parakeet-0.6b is only available for language code en-US"
      exit 1
    fi
  elif [[ ${asr_acoustic_model} == "conformer" ]]; then
    if [[ ${asr_acoustic_model_variant} != "unified" && ${asr_acoustic_model_variant} != "ml_cs" && ${asr_acoustic_model_variant} != "unified_ml_cs" && ${asr_acoustic_model_variant} != "xl" && ${asr_acoustic_model_variant} != "" ]]; then
      echo "Valid variants for Conformer are: unified, ml_cs and unified_ml_cs."
      exit 1
    fi

    if [[ ${asr_acoustic_model_variant} == "unified" && ${lang_code} != "de-DE" && ${lang_code} != "ja-JP" && ${lang_code} != "zh-CN" ]]; then
      echo "Unified Conformer acoustic model is only available for language code de-DE, ja-JP and zh-CN."
      exit 1
    fi

    if [[ ${asr_acoustic_model_variant} == "ml_cs" && ${lang_code} != "es-en-US" ]]; then
      echo "Multilingual Code Switch Conformer acoustic model is only available for language code es-en-US."
      exit 1
    fi

    if [[ ${asr_acoustic_model_variant} == "unified_ml_cs" && ${lang_code} != "ja-en-JP" ]]; then
      echo "Unified Multilingual Code Switch Conformer acoustic model is only available for language code ja-en-JP."
      exit 1
    fi

    if [[ ${asr_acoustic_model_variant} == "xl" && ${lang_code} != "en-US" ]]; then
      echo "Conformer-XL acoustic model is only available for language code en-US."
      exit 1
    fi
  else
    echo "Invalid acoustic model architecture, valid acoustic model architectures are conformer and parakeet"
    exit 1
  fi
  if [[ $riva_target_gpu_family  == "tegra" ]]; then
    if [[ ${asr_acoustic_model} == "conformer" && \
        ${asr_acoustic_model_variant} == "xl" ]]; then
        echo "Conformer-XL model not available for arm64 architecture"
        exit 1
    fi
  fi
}


models_asr=()

for lang_code in ${asr_language_code[@]}; do
    validate_asr_config
    modified_lang_code="${lang_code//-/_}"
    modified_lang_code=${modified_lang_code,,}

    decoder=""
    if [ "$use_asr_greedy_decoder" = true ]; then
      decoder="_gre"
    fi
    modified_asr_acoustic_model_variant=""
    if [[ "$asr_acoustic_model_variant" != "" ]]; then
      modified_asr_acoustic_model_variant="_${asr_acoustic_model_variant}"
    fi
    if [[ "$modified_asr_acoustic_model_variant" == "_0.6b" ]]; then
      modified_asr_acoustic_model_variant="_0-6b"
    fi

    if [[ $riva_target_gpu_family  == "tegra" ]]; then
      models_asr+=(
      ### Streaming w/ CPU decoder, best latency configuration
          "${riva_ngc_org}/${riva_ngc_team}/models_asr_${asr_acoustic_model}${modified_asr_acoustic_model_variant}_${modified_lang_code}_str:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"

      ### Offline w/ CPU decoder
      #    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}${modified_asr_acoustic_model_variant}_${modified_lang_code}_ofl${decoder}:${riva_ngc_model_version}"
      )
    else
      models_asr+=(
      ### Streaming w/ CPU decoder, best latency configuration
          "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}${modified_asr_acoustic_model_variant}_${modified_lang_code}_str${decoder}:${riva_ngc_model_version}"

      ### Streaming w/ CPU decoder, best throughput configuration
      #    "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}${modified_asr_acoustic_model_variant}_${modified_lang_code}_str_thr${decoder}:${riva_ngc_model_version}"

      ### Offline w/ CPU decoder
          "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}${modified_asr_acoustic_model_variant}_${modified_lang_code}_ofl${decoder}:${riva_ngc_model_version}"
      )
    fi

    ### Punctuation model
    if [[ ${modified_asr_acoustic_model_variant} != "_unified" && ${modified_asr_acoustic_model_variant} != "_unified_ml_cs" ]]; then
      pnc_lang=$(echo $modified_lang_code | cut -d "_" -f 1)
      pnc_region=${modified_lang_code##*_}
      modified_lang_code=${pnc_lang}_${pnc_region}
      if [[ $riva_target_gpu_family == "tegra" ]]; then
        if [[ "$lang_code" == "en-US" ]]; then
          models_asr+=(
              "${riva_ngc_org}/${riva_ngc_team}/models_nlp_punctuation_bert_base_${modified_lang_code}:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
          #    "${riva_ngc_org}/${riva_ngc_team}/models_nlp_punctuation_bert_large_${modified_lang_code}:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
          )
        else
          models_asr+=(
              "${riva_ngc_org}/${riva_ngc_team}/models_nlp_punctuation_bert_base_${modified_lang_code}:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
          )
        fi
      else
        if [[ "$lang_code" == "en-US" ]]; then
          models_asr+=(
              "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base_${modified_lang_code}:${riva_ngc_model_version}"
          #    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_large_${modified_lang_code}:${riva_ngc_model_version}"
          )
        else
          models_asr+=(
              "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base_${modified_lang_code}:${riva_ngc_model_version}"
          )
        fi
      fi
    fi
done

### Speaker diarization model
models_asr+=(
#    "${riva_ngc_org}/${riva_ngc_team}/rmir_diarizer_offline:${riva_ngc_model_version}"
)

########## NLP MODELS ##########

if [[ $riva_target_gpu_family == "tegra" ]]; then
  models_nlp=(
  ### Bert base Punctuation model
      "${riva_ngc_org}/${riva_ngc_team}/models_nlp_punctuation_bert_base_en_us:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
  #    "${riva_ngc_org}/${riva_ngc_team}/models_nlp_punctuation_bert_large_en_us:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
  )
else
  models_nlp=(
  ### Bert base Punctuation model
      "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base_en_us:${riva_ngc_model_version}"
  #    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_large_en_us:${riva_ngc_model_version}"
  )
fi

########## TTS MODELS ##########

models_tts=()

for lang_code in ${tts_language_code[@]}; do
  modified_lang_code="${lang_code//-/_}"
  modified_lang_code=${modified_lang_code,,}

  if [[ $riva_target_gpu_family == "tegra" ]]; then
    if [[ ${lang_code} == "en-US" ]]; then
      models_tts+=(
      ### These models have been trained with energy conditioning and use the International Phonetic Alphabet (IPA) for inference and training.
          "${riva_ngc_org}/${riva_ngc_team}/models_tts_fastpitch_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
      #    "${riva_ngc_org}/${riva_ngc_team}/models_tts_radtts_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"

      ### This is a beta multi-speaker model with new features like emotion mixing and use the International Phonetic Alphabet (IPA) for inference and training.
      #    "${riva_ngc_org}/${riva_ngc_team}/models_tts_radttspp_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"

      ### This is a zero shot model for synthesizing speech using audio prompt input, require access to ea-riva-tts NGC org for using it
      #    "gjaugwraudqz/rmir_tts_pflow_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"

      ### This model uses the ARPABET for inference and training.
      #    "${riva_ngc_org}/${riva_ngc_team}/models_tts_fastpitch_hifigan_${modified_lang_code}:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
      )
    elif [[ ${lang_code} == "zh-CN" || ${lang_code} == "es-US" ]]; then
      models_tts+=(
      ### This model is multi-speaker with emotion and and use the International Phonetic Alphabet (IPA) for inference and training.
          "${riva_ngc_org}/${riva_ngc_team}/models_tts_fastpitch_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
      )
    else
      ### These models are single-speaker and use the International Phonetic Alphabet (IPA) for inference and training.
      if [[ ${lang_code} != "de-DE" ]]; then
        models_tts+=(
            "${riva_ngc_org}/${riva_ngc_team}/models_tts_fastpitch_hifigan_${modified_lang_code}_f_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
        )
      fi
      models_tts+=(
          "${riva_ngc_org}/${riva_ngc_team}/models_tts_fastpitch_hifigan_${modified_lang_code}_m_ipa:${riva_ngc_model_version}-${riva_target_gpu_family}-${riva_tegra_platform}"
      )
    fi
  else
    if [[ ${lang_code} == "en-US" ]]; then
      models_tts+=(
      ### These models have been trained with energy conditioning and use the International Phonetic Alphabet (IPA) for inference and training.
          "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"
      #    "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_radtts_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"

      ### This is a beta multi-speaker model with new features like emotion mixing and use the International Phonetic Alphabet (IPA) for inference and training.
      #    "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_radttspp_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"

      ### This is a zero shot model for synthesizing speech using audio prompt input, require special access to ea-riva-tts NGC org for using it
      #    "gjaugwraudqz/rmir_tts_pflow_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"

      ### This model uses the ARPABET for inference and training.
      #    "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_${modified_lang_code}:${riva_ngc_model_version}"
      )
    elif [[ ${lang_code} == "zh-CN" || ${lang_code} == "es-US" ]]; then
      models_tts+=(
      ### This model is multi-speaker with emotion and and use the International Phonetic Alphabet (IPA) for inference and training.
          "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_${modified_lang_code}_ipa:${riva_ngc_model_version}"
      )
    else
      ### These models are single-speaker and use the International Phonetic Alphabet (IPA) for inference and training.
      if [[ ${lang_code} != "de-DE" ]]; then
        models_tts+=(
            "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_${modified_lang_code}_f_ipa:${riva_ngc_model_version}"
        )
      fi
      models_tts+=(
          "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_${modified_lang_code}_m_ipa:${riva_ngc_model_version}"
      )
    fi
  fi
done

######### NMT models ###############

# Models follow Source language _ One or more target languages model architecture
# Source or target language "any" means the model supports 32 languages mentioned in docs.
# e.g., rmir_nmt_de_en_24x6 is a German to English 24x6 bilingual model
# and rmir_megatronnmt_en_any_500m is a English to 32 languages megatron model

models_nmt=(
  ###### Bilingual models
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_de_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_es_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_zh_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_ru_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_fr_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_de_en_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_es_en_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_ru_en_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_zh_en_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_fr_en_24x6:${riva_ngc_model_version}"

  ###### Multilingual models
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_deesfr_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_en_deesfr_12x2:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_deesfr_en_24x6:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_deesfr_en_12x2:${riva_ngc_model_version}"

  ###### Megatron models
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_megatronnmt_any_en_500m:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_megatronnmt_en_any_500m:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_megatron_1b_any_en:${riva_ngc_model_version}"
  #"${riva_ngc_org}/${riva_ngc_team}/rmir_nmt_megatron_1b_en_any:${riva_ngc_model_version}"

)

NGC_TARGET=${riva_ngc_org}
if [[ ! -z ${riva_ngc_team} ]]; then
  NGC_TARGET="${NGC_TARGET}/${riva_ngc_team}"
else
  team="\"\""
fi

# Specify paths to SSL Key and Certificate files to use TLS/SSL Credentials for a secured connection.
# If either are empty, an insecure connection will be used.
# Stored within container at /ssl/servert.crt and /ssl/server.key
# Optional, one can also specify a root certificate, stored within container at /ssl/root_server.crt
ssl_server_cert=""
ssl_server_key=""
ssl_root_cert=""

# define Docker images required to run Riva
image_speech_api="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}"

# define Docker images required to setup Riva
image_init_speech="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}-servicemaker"

# daemon names
riva_daemon_speech="riva-speech"
if [[ $riva_target_gpu_family != "tegra" ]]; then
    riva_daemon_client="riva-client"
fi
