"""Wrapper for TFLite conditioners model."""
import tensorflow as tf
import mlx.core as mx
import numpy as np

class TFLiteConditioners:
    """Use the working TFLite conditioners model for text + time conditioning."""
    
    def __init__(self, model_path, tokenizer):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.tokenizer = tokenizer
        
    def __call__(self, prompt, seconds_total):
        """Generate conditioning for the given prompt and duration.
        
        Args:
            prompt: Text prompt
            seconds_total: Duration in seconds
            
        Returns:
            cross_attn_cond: (1, 65, 768) - Text + time tokens for cross-attention
            global_cond: (1, 768) - Global conditioning (time embedding)
        """
        # Tokenize (TFLite expects max_length=128)
        tokens = self.tokenizer(prompt, return_tensors='np', padding='max_length', 
                                max_length=128, truncation=True)
        
        # Set inputs
        self.interpreter.set_tensor(0, tokens['input_ids'].astype(np.int64))
        self.interpreter.set_tensor(1, tokens['attention_mask'].astype(np.int64))
        self.interpreter.set_tensor(2, np.array([seconds_total], dtype=np.float32))
        
        # Run
        self.interpreter.invoke()
        
        # Get outputs
        output_details = self.interpreter.get_output_details()
        cross_attn = self.interpreter.get_tensor(output_details[0]['index'])  # (1, 65, 768)
        global_cond = self.interpreter.get_tensor(output_details[2]['index'])  # (1, 768)
        
        # Convert to MLX
        return mx.array(cross_attn), mx.array(global_cond)

