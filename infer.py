# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# import os
# import sys
# import time

# import fire

# import torch

# from accelerate.utils import is_xpu_available
# # from llama_cookbook.inference.model_utils import load_model, load_peft_model

# from llama_cookbook.inference.safety_utils import AgentType, get_safety_checker
# from transformers import AutoTokenizer


# def main(
#     model_name = r"C:\Users\dell\.llama\checkpoints\Llama3.2-1B-Instruct-int4-spinquant-eo8\gguf-model-filename.gguf",
#     peft_model: str = None,
#     quantization: str = None, # Options: 4bit, 8bit
#     max_new_tokens=100,  # The maximum numbers of tokens to generate
#     prompt_file: str = None,
#     seed: int = 42,  # seed value for reproducibility
#     do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
#     min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
#     use_cache: bool = True,  # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
#     top_p: float = 1.0,  # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
#     temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
#     top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
#     repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
#     length_penalty: int = 1,  # [optional] Exponential penalty to the length that is used with beam-based generation.
#     enable_azure_content_safety: bool = False,  # Enable safety check with Azure content safety api
#     enable_sensitive_topics: bool = False,  # Enable check for sensitive topics using AuditNLG APIs
#     enable_salesforce_content_safety: bool = True,  # Enable safety check with Salesforce safety flan t5
#     enable_llamaguard_content_safety: bool = False,
#     max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
#     use_fast_kernels: bool = False,  # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
#     share_gradio: bool = False,  # Enable endpoint creation for gradio.live
#     **kwargs,
# ):
#     # Set the seeds for reproducibility
#     if is_xpu_available():
#         torch.xpu.manual_seed(seed)
#     else:
#         torch.cuda.manual_seed(seed)
#     torch.manual_seed(seed)

#     model = load_model(model_name, quantization, use_fast_kernels, **kwargs)
#     if peft_model:
#         model = load_peft_model(model, peft_model)

#     model.eval()

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     def inference(
#         user_prompt,
#         temperature,
#         top_p,
#         top_k,
#         max_new_tokens,
#         **kwargs,
#     ):
#         safety_checker = get_safety_checker(
#             enable_azure_content_safety,
#             enable_sensitive_topics,
#             enable_salesforce_content_safety,
#             enable_llamaguard_content_safety,
#         )

#         # Safety check of the user prompt
#         safety_results = [check(user_prompt) for check in safety_checker]
#         are_safe = all([r[1] for r in safety_results])
#         if are_safe:
#             print("User prompt deemed safe.")
#             print(f"User prompt:\n{user_prompt}")
#         else:
#             print("User prompt deemed unsafe.")
#             for method, is_safe, report in safety_results:
#                 if not is_safe:
#                     print(method)
#                     print(report)
#             print("Skipping the inference as the prompt is not safe.")
#             return  # Exit the program with an error status

#         batch = tokenizer(
#             user_prompt,
#             truncation=True,
#             max_length=max_padding_length,
#             return_tensors="pt",
#         )
#         if is_xpu_available():
#             batch = {k: v.to("xpu") for k, v in batch.items()}
#         else:
#             batch = {k: v.to("cuda") for k, v in batch.items()}

#         start = time.perf_counter()
#         with torch.no_grad():
#             outputs = model.generate(
#                 **batch,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=do_sample,
#                 top_p=top_p,
#                 temperature=temperature,
#                 min_length=min_length,
#                 use_cache=use_cache,
#                 top_k=top_k,
#                 repetition_penalty=repetition_penalty,
#                 length_penalty=length_penalty,
#                 **kwargs,
#             )
#         e2e_inference_time = (time.perf_counter() - start) * 1000
#         print(f"the inference time is {e2e_inference_time} ms")
#         output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#         # Safety check of the model output
#         safety_results = [
#             check(output_text, agent_type=AgentType.AGENT, user_prompt=user_prompt)
#             for check in safety_checker
#         ]
#         are_safe = all([r[1] for r in safety_results])
#         if are_safe:
#             print("User input and model output deemed safe.")
#             print(f"Model output:\n{output_text}")
#             return output_text
#         else:
#             print("Model output deemed unsafe.")
#             for method, is_safe, report in safety_results:
#                 if not is_safe:
#                     print(method)
#                     print(report)
#             return None

#     if prompt_file is not None:
#         assert os.path.exists(
#             prompt_file
#         ), f"Provided Prompt file does not exist {prompt_file}"
#         with open(prompt_file, "r") as f:
#             user_prompt = "\n".join(f.readlines())
#         inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
#     elif not sys.stdin.isatty():
#         user_prompt = "\n".join(sys.stdin.readlines())
#         inference(user_prompt, temperature, top_p, top_k, max_new_tokens)
#     else:
#         try:
#             import gradio as gr
#         except ImportError:
#             raise ImportError("This part of the recipe requires gradio. Please run `pip install gradio`")
            
#         gr.Interface(
#             fn=inference,
#             inputs=[
#                 gr.components.Textbox(
#                     lines=9,
#                     label="User Prompt",
#                     placeholder="none",
#                 ),
#                 gr.components.Slider(
#                     minimum=0, maximum=1, value=1.0, label="Temperature"
#                 ),
#                 gr.components.Slider(minimum=0, maximum=1, value=1.0, label="Top p"),
#                 gr.components.Slider(
#                     minimum=0, maximum=100, step=1, value=50, label="Top k"
#                 ),
#                 gr.components.Slider(
#                     minimum=1, maximum=2000, step=1, value=200, label="Max tokens"
#                 ),
#             ],
#             outputs=[
#                 gr.components.Textbox(
#                     lines=5,
#                     label="Output",
#                 )
#             ],
#             title="Meta Llama3 Playground",
#             description="https://github.com/meta-llama/llama-cookbook",
#         ).queue().launch(server_name="0.0.0.0", share=share_gradio)


# if __name__ == "__main__":
#     fire.Fire(main)