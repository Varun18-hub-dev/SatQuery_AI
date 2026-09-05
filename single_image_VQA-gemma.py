import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class PaliGemmaVQAAdapter:
    """Unified Adapter for Single-Image Visual Question Answering (Track A)."""
    
    def __init__(self, model_id: str = "google/paligemma-3b-pt-224"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def predict(self, image_input, query: str) -> dict:
        """
        Executes VQA on an input image and query.
        
        Args:
            image_input: File path (str) or PIL Image instance.
            query: Natural language question regarding the scene.
            
        Returns:
            dict: Standardized adapter contract {text, evidence_type, spatial_evidence, confidence}
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        prompt = f"<image>question en {query}"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=60,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode generated answer
        generated_tokens = outputs.sequences[0]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)
        answer = decoded[len(prompt.replace("<image>", "")) :].strip()

        # Compute average softmax confidence across generated output tokens
        logits = torch.stack(outputs.scores, dim=1)  # shape: (1, seq_len, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        token_probs = probs[0, torch.arange(len(outputs.scores)), generated_tokens[-len(outputs.scores):]]
        avg_confidence = float(token_probs.mean().item())

        return {
            "text": answer,
            "evidence_type": "none",
            "spatial_evidence": None,
            "confidence": round(avg_confidence, 4)
        }