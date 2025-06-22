import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

# Configuration
MODEL_ID = "saish-shetty/SmolLM2-1.7B-pre-trained"
DEVICE = "cpu"  # CPU-only for HF Spaces
MAX_LENGTH = 512

def load_model():
    """Load model and tokenizer"""
    print(f"Loading model: {MODEL_ID}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,  # Changed from bfloat16 to float32 for CPU
            device_map=None,  # Changed from "cpu" to None
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Added for better CPU performance
        )
        
        # Move to CPU explicitly
        model = model.to('cpu')
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Model loaded successfully!")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# Load model globally
model, tokenizer = load_model()

def generate_text(
    prompt, 
    max_new_tokens=100, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=50, 
    repetition_penalty=1.1
):
    """Generate text based on input prompt and parameters"""
    
    if not prompt.strip():
        return "Please enter a prompt to generate text."
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        
        # Generate
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()
        
        # Calculate stats
        tokens_generated = len(tokenizer.encode(new_text, add_special_tokens=False))
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        # Format output
        result = f"{new_text}\n\n"
        result += f"📊 **Generation Stats:**\n"
        result += f"• Tokens generated: {tokens_generated}\n"
        result += f"• Time taken: {generation_time:.2f}s\n"
        result += f"• Speed: {tokens_per_second:.1f} tokens/sec"
        
        return result
        
    except Exception as e:
        return f"Error generating text: {str(e)}"

def get_model_info():
    """Return model information"""
    info = f"""
    # 🤖 SmolLM2-1.7B Pre-trained from Scratch on Cosmopedia-v2
    
    **Base Model:** HuggingFaceTB/SmolLM2-1.7B  
    **Training Dataset:** Cosmopedia-v2 (1B tokens subset)  
    **Training Steps:** 30,000 steps  
    **Final Loss:** 3.7547  
    **Parameters:** 1.7B  
    
    **Training Configuration:**
    - Batch Size: 1 per device
    - Gradient Accumulation: 16 steps  
    - Learning Rate: 2e-5
    - Sequence Length: 2048 tokens
    - Optimizer: 8-bit AdamW
    - Mixed Precision: bf16
    
    This model was pre-trained from scratch on educational content from Cosmopedia-v2, 
    making it particularly good at explaining concepts, answering questions, 
    and generating educational content.
    """
    return info

# Example prompts for the interface
example_prompts = [
    "What is the meaning of life and how do different philosophical traditions approach this question?",
    "Explain the concept of free will versus determinism in philosophy.",
    "What are the main arguments for and against the existence of objective moral truths?",
    "How do different cultures define happiness and what can we learn from these perspectives?",
    "What is consciousness and why is it considered one of philosophy's hardest problems?",
    "Discuss the relationship between knowledge and belief in epistemology.",
    "What are the ethical implications of artificial intelligence in modern society?",
    "How do ancient philosophical teachings remain relevant in contemporary life?",
]

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="SmolLM2-1.7B Cosmopedia Demo",
        theme=gr.themes.Soft(),
        css="""
        .model-info { 
            background-color: transparent !important; 
            color: var(--body-text-color) !important;
            padding: 20px; 
            border-radius: 10px; 
            margin: 10px 0; 
        }
        """
    ) as demo:
        
        # Header
        gr.Markdown("# 🌟 SmolLM2-1.7B Cosmopedia Chat")
        gr.Markdown("*Pre-trained From Scratch on educational content for thoughtful conversations*")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                prompt_input = gr.Textbox(
                    label="💭 Your Prompt",
                    placeholder="Ask a philosophical question or request an explanation...",
                    lines=3,
                    value=""
                )
                
                # Generation parameters
                with gr.Accordion("🔧 Generation Parameters", open=False):
                    max_tokens = gr.Slider(
                        minimum=10,
                        maximum=300,
                        value=100,
                        step=10,
                        label="Max New Tokens"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (creativity)"
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (nucleus sampling)"
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-k"
                    )
                    repetition_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.1,
                        step=0.05,
                        label="Repetition Penalty"
                    )
                
                # Generate button
                generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")
                
                # Example prompts
                gr.Markdown("### 💡 Example Prompts:")
                example_buttons = []
                for i, example in enumerate(example_prompts):
                    btn = gr.Button(f"📚 {example[:60]}..." if len(example) > 60 else f"📚 {example}", size="sm")
                    example_buttons.append((btn, example))
            
            with gr.Column(scale=2):
                # Output section
                output_text = gr.Textbox(
                    label="🤖 Generated Response",
                    lines=15,
                    max_lines=20,
                    show_copy_button=True
                )
                
                # Clear button
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
        
        # Model info section
        with gr.Accordion("ℹ️ Model Information", open=False):
            model_info = gr.Markdown(get_model_info())
        
        # Event handlers
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_input, max_tokens, temperature, top_p, top_k, repetition_penalty],
            outputs=output_text
        )
        
        # Example button handlers
        for btn, example in example_buttons:
            btn.click(
                lambda example=example: example,
                outputs=prompt_input
            )
        
        clear_btn.click(
            lambda: ("", ""),
            outputs=[prompt_input, output_text]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Note:** This model runs on CPU for demo purposes. For faster inference, use GPU deployment.
        
        🔗 **Model:** [saish-shetty/SmolLM2-1.7B-pre-trained](https://huggingface.co/saish-shetty/SmolLM2-1.7B-pre-trained)
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )