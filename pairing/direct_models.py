"""Participant selection for new, versioned adaptive runs (never legacy resumes)."""
from pairing.context_history import POLICY


def create_participant(instructions, directory, config, *, token_counter=None):
    """Build a client; caller owns simulator/trial loop and campaign authorization.

    Required config: provider, model, context_window, max_output_tokens.
    No model or output-limit substitutions. No inference is launched here.
    """
    required={'provider','model','context_window','max_output_tokens'}
    missing=required-set(config)
    if missing:raise ValueError('Missing participant configuration: '+', '.join(sorted(missing)))
    if 'thinking_display' in config and config['provider']!='anthropic':
        raise ValueError('thinking_display is supported only by the native Anthropic client')
    if config.get('thinking_display') not in (None,'summarized'):
        raise ValueError('thinking_display supports only None or summarized')
    common=dict(context_window=config['context_window'],token_counter=token_counter,
                effort=config.get('effort','high'))
    if config['provider']=='openai':
        from pairing.openai_model import OpenAILearningModel
        return OpenAILearningModel(instructions,directory,model=config['model'],
            max_output_tokens=config['max_output_tokens'],
            reasoning_summary=config.get('reasoning_summary','detailed'),**common)
    if config['provider']=='anthropic':
        from pairing.anthropic_model import AnthropicLearningModel,MODEL
        if config['model']!=MODEL:raise ValueError('Anthropic client requires exact '+MODEL)
        return AnthropicLearningModel(instructions,directory,context_policy=POLICY,
            compact_at=min(760_000,config['context_window']-1),
            max_tokens=config['max_output_tokens'],workspace_id=config.get('workspace_id'),
            workspace_name=config.get('workspace_name'),
            cache_ttl=config.get('cache_ttl','5m'),thinking_display=config.get('thinking_display'),ledger_path=config.get('ledger_path'),
            spend_limit_usd=config.get('spend_limit_usd',195.0),**common)
    if config['provider']=='google':
        from pairing.google_model import GoogleLearningModel
        return GoogleLearningModel(instructions,directory,model=config['model'],
            max_output_tokens=config['max_output_tokens'],ledger_path=config.get('ledger_path'),
            spend_limit_usd=config.get('spend_limit_usd',125.0),
            input_price_per_million=config.get('input_price_per_million',2.0),
            output_price_per_million=config.get('output_price_per_million',10.0),**common)
    if config['provider']=='openrouter':
        from pairing.openrouter_model import OpenRouterLearningModel
        return OpenRouterLearningModel(instructions,directory,model=config['model'],
            max_output_tokens=config['max_output_tokens'],provider=config.get('upstream_provider'),
            context_policy=config.get('context_policy',POLICY),ledger_path=config.get('ledger_path'),
            spend_limit_usd=config.get('spend_limit_usd',150.0),
            input_price_per_million=config.get('input_price_per_million',2.0),
            output_price_per_million=config.get('output_price_per_million',10.0),**common)
    raise ValueError('Unsupported direct participant provider')
