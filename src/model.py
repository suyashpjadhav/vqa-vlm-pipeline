"""
Vision-Language Model (VLM) initialization and inference wrapper for VQA.
"""

from typing import Dict, Any, List, Optional, Union
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


class VQAClassifier:
    """
    VQA Classifier wrapping Salesforce/blip-vqa-base using BlipProcessor and BlipForQuestionAnswering.
    """
    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: Optional[str] = None
    ) -> None:
        """
        Initializes BlipProcessor and model, mapping to GPU if available, else CPU.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"Loading BlipProcessor and BlipForQuestionAnswering '{model_name}' on device '{self.device}'...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def answer_question(self, image: Union[Image.Image, str], question: str) -> str:
        """
        Preprocesses inputs via BlipProcessor, generates output tokens, and decodes them into text.

        Args:
            image (Union[Image.Image, str]): PIL Image object or image file path.
            question (str): Question string regarding the image.

        Returns:
            str: Decoded answer text.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, "convert") and image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_new_tokens=50)

        answer = self.processor.decode(output_tokens[0], skip_special_tokens=True).strip()
        return answer

    def predict(self, image: Union[Image.Image, str], question: str) -> str:
        """
        Alias for answer_question.
        """
        return self.answer_question(image, question)

    def batch_predict(self, images: List[Union[Image.Image, str]], questions: List[str]) -> List[str]:
        """
        Generates answers for a batch of images and questions.
        """
        pil_images = [
            Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
            for img in images
        ]

        inputs = self.processor(images=pil_images, text=questions, padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_new_tokens=50)

        answers = self.processor.batch_decode(output_tokens, skip_special_tokens=True)
        return [ans.strip() for ans in answers]


# Alias for backward compatibility
VQAQuestionAnsweringModel = VQAClassifier

