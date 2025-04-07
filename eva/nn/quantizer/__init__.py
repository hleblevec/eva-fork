def lsq_quantization(model, q_config):
    from .lsqplus_quantize_V2 import prepare as lsqplusprepareV2 

    lsqplusprepareV2(
        model,
        inplace=True,
        a_bits=q_config["a_bit"],
        w_bits=q_config["w_bit"],
        all_positive=q_config["all_positive"],
        quant_inference=q_config["quant_inference"],
        per_channel=q_config["per_channel"],
        batch_init=q_config["batch_init"],
    ) 
    return model