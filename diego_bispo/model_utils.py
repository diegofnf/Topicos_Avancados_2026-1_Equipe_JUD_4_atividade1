import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import HF_CACHE_DIR


def load_model(model_name: str, cache_dir: str = HF_CACHE_DIR):
    """
    Carrega modelo e tokenizer com quantização 4bit (NF4).

    - load_in_4bit reduz Llama 8B de ~16 GB para ~4 GB
    - nf4 é mais preciso que int4 para distribuições de pesos neurais
    - compute_dtype=float16 faz os cálculos em 16bit mesmo com pesos em 4bit
    - double_quant quantiza também as constantes de quantização (~0.4 GB a menos)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Modelos como Llama não definem pad_token por padrão.
    # Usar eos_token como fallback é a convenção padrão da comunidade.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        # Delega ao accelerate a alocação de camadas entre GPUs/RAM.
        # Essencial para rodar modelos grandes no Colab sem gerenciar manualmente.
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def unload(model, tokenizer):
    """
    Libera VRAM completamente.

    Sequência necessária:
    1. to("cpu") — move tensores da GPU para RAM
    2. del        — remove referências Python
    3. gc.collect() — limpa referências circulares
    4. empty_cache() — devolve memória GPU ao sistema operacional
    """
    model.to("cpu")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def gerar_texto(
    model,
    tokenizer,
    prompt: str,
    sample: bool = True,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
) -> str:
    """
    Gera texto a partir de um prompt.

    - sample=True  → candidato (criativo), usa temperature
    - sample=False → juiz/curador (determinístico), ignora temperature

    Estratégia de montagem do prompt:
    1. Tenta apply_chat_template com system role separado (Llama, Qwen)
    2. Se falhar, incorpora system_prompt no conteúdo do usuário (Gemma)
    3. Se falhar, usa texto plano como fallback final
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages_com_system = []
    if system_prompt:
        messages_com_system.append({"role": "system", "content": system_prompt.strip()})
    messages_com_system.append({"role": "user", "content": prompt.strip()})

    messages_sem_system = [
        {
            "role": "user",
            "content": f"{system_prompt.strip()}\n\n{prompt.strip()}" if system_prompt else prompt.strip(),
        }
    ]

    input_ids = None
    attention_mask = None

    for messages in [messages_com_system, messages_sem_system]:
        try:
            result = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            # apply_chat_template pode retornar tensor ou BatchEncoding
            # dependendo da versão do transformers
            if hasattr(result, "input_ids"):
                input_ids = result.input_ids.to(device)
                attention_mask = (
                    result.attention_mask.to(device)
                    if hasattr(result, "attention_mask")
                    else torch.ones_like(input_ids)
                )
            else:
                input_ids = result.to(device)
                attention_mask = torch.ones_like(input_ids)
            break
        except Exception as e:
            print(f"[aviso] apply_chat_template falhou ({e}), tentando fallback.")

    if input_ids is None:
        texto = f"{system_prompt.strip()}\n\n{prompt.strip()}" if system_prompt else prompt.strip()
        encoded = tokenizer(texto, return_tensors="pt").to(device)
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # 1.3 para candidato: penaliza repetições mais agressivamente.
        # 1.1 para juiz/curador: evita loops sem distorcer termos jurídicos.
        # sample=True  → (criatividade controlada), apenas para as respostas discursivas.
        # sample=False → (resposta exata), para os demais.
        "repetition_penalty": 1.3 if sample else 1.1,
    }

    # temperature só faz sentido com do_sample=True.
    # Quando do_sample=False, anula temperature e top_p do generation_config.json
    # para evitar conflito e warning de flags inválidas.
    if sample:
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["temperature"] = None
        generation_kwargs["top_p"] = None

    outputs = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()