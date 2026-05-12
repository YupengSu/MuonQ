from .models import GPT, GPTConfig, LlamaModel, LlamaConfig


def load_model(config, device):
    model_type = config.get('model_type', 'gpt') 

    if model_type == 'llama':
        llamaconfig = LlamaConfig()
        llamaconfig.hidden_size = config['n_embd']
        llamaconfig.num_hidden_layers = config['n_layer']
        llamaconfig.num_attention_heads = config['n_head']
        llamaconfig.num_key_value_heads = config.get('n_kv_head', config['n_head'])
        llamaconfig.vocab_size = config['vocab_size']
        llamaconfig.max_position_embeddings = config.get('max_position_embeddings', config.get('block_size', 4096))
        if 'intermediate_size' in config:
            llamaconfig.intermediate_size = config['intermediate_size']
        if 'rope_theta' in config:
            llamaconfig.rope_theta = config['rope_theta']
        if 'rms_norm_eps' in config:
            llamaconfig.rms_norm_eps = config['rms_norm_eps']
        model = LlamaModel(llamaconfig, device, flash_attention=config['flash_attention'])
    else:
        gptconfig = GPTConfig()
        gptconfig.n_embd = config['n_embd']
        gptconfig.n_layer = config['n_layer']
        gptconfig.n_head = config['n_head']
        gptconfig.vocab_size = config['vocab_size']
        gptconfig.block_size = config['block_size']
        model = GPT(gptconfig, device, flash_attention=config['flash_attention'])
    return model

